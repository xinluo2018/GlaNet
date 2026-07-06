'''
author: xin luo
create: 2026.7.4
des: v2 training script for u3net_cross_fusion_v2.
changes vs 5_1_trainer_cross_fusion.py:
  (1) warm start from the best v1 checkpoint (val mIoU 0.952);
  (2) loss: deep-supervised BCE + Dice (main + aux4), fixed the v1 dice_loss
      that unpacked only 2 of the 4 model outputs;
  (3) LR schedule: short linear warmup + cosine annealing (5e-5 -> 1e-6);
  (4) EMA weight averaging, both raw and EMA weights evaluated at val time;
  (5) bf16 autocast for training (fp32 for validation metrics).
r3 (2026.7.4): model-side change only — bottleneck fusion switched to joint
  cross-attention (Joint_Vit_Fusion, see u3net_cross_fusion_v2.py header for
  references); recipe identical to r2. Warm start still from the 0.9522 soup:
  encoders/decoder/fusion-MLPs load, fusion attention projections re-init.
  RESULT: best raw 0.9514 / ema 0.9512 < r2's 0.9521 -> model reverted to
  pairwise fusion.
r4 (2026.7.5): low-LR cosine fine-tune of the user's best BCE-trained
  checkpoint (u3net_cross_fusion_0952.pth, val 0.952) with the IoU-directed
  loss (BCE+Dice+0.5*Lovasz) + EMA: lr 3e-5 -> 1e-6 over 200 epochs.
  Rationale: constant-LR runs (r2/r3 and the user's) peak mid-run then decay
  from overfitting; a decaying-LR refinement from the best point is the
  standard squeeze (Lovasz directly optimizes the Jaccard index,
  Berman et al., CVPR 2018).
  RESULT: best 0.9521 = the warm start itself (ep1); no gain, late decay to
  ~0.950 -> fine-tuning from the peak is exhausted as a lever.
r5 (2026.7.5): r2 recipe unchanged (soup09522 warm start, constant lr 1e-4,
  600 ep, BCE+Dice+0.5*Lovasz, EMA); single change: train-time MODALITY
  DROPOUT (p=0.15) inside the bottleneck cross-fusion — per sample, a
  modality is dropped as cross-attention source (inverted scaling, zero new
  params, inference unchanged). Rationale: every run overfits (train 0.975+
  vs val ~0.951); modality-level dropout regularizes multimodal fusion and
  prevents over-reliance on one modality (ModDrop, Neverova et al., TPAMI
  2016; ShaSpec, Wang et al., CVPR 2023; Hong et al., TGRS 2021 on
  multimodal RS classification).
  RESULT: peak 0.9513 @ep109 < r2's 0.9521 -> modality dropout rejected
  (mod_drop back to 0; code kept in the model file).
r6 (2026.7.5): error decomposition on the valset (soup09522) showed the
  binding error is INSTANCE imbalance, not pixel imbalance: glaciers <64px
  are only 49% detected (1870 components, pixel recall 53%), and recovering
  just their FN pixels puts global mIoU at ~0.9552 (>= target). Fix: add a
  BLOB LOSS term (Kofler et al., 'blob loss: instance imbalance aware loss
  functions for semantic segmentation', arXiv 2022 / IPMI 2023): per-GT-
  connected-component soft Dice (other components masked out, background FP
  shared), averaged over components -> each small glacier weighs as one
  instance instead of a few pixels. Vectorized via index_add (one pass per
  sample, no per-blob loops on hot path). Recipe otherwise = r2 (soup09522
  warm start, constant lr 1e-4, 600 ep, BCE+Dice+0.5*Lovasz + EMA);
  L = base + 0.5 * blob_dice(main). Related: Focal Tversky (Abraham & Khan,
  ISBI 2019) and small-object oversampling (Kisantal et al., 2019) are the
  follow-up levers if blob loss alone undershoots.
  RESULT: peak 0.9513 @ep261 < r2's 0.9521, and the error decomposition on
  the final ckpt shows it BACKFIRED: <64px detection 49%->44%, FN px
  150k->206k, FP blobs 33k->26k. Diagnosis: each blob's dice shares the
  WHOLE background FP mass in its denominator; with ~24 blobs/patch the
  gradient is ~24x "suppress FP everywhere" and the per-blob recall push
  drowns -> the model turned conservative, i.e. MORE small-glacier misses.
r7 (2026.7.5): instance term switched from dice to pure per-component soft
  RECALL: L = base + 0.5 * mean_c(1 - (p_c+s)/(area_c+s)) — each GT
  component only pushes ITS OWN pixels up (equivalent to positive-pixel
  weighting ~ 1/component-area, cf. inverse-size instance weighting in
  blob loss ablations, Kofler et al. 2022); FP control stays with the
  global BCE+Dice+Lovasz. Save line lowered 0.9515->0.9512 to harvest
  soup candidates.
  RESULT: global peak only 0.9463 @ep29, final 0.9448 — BUT the
  decomposition proves the lever works: <64px detection 49%->84.2%, pixel
  recall 53%->85.2%, total FN 150k->94.5k. Cost: confident over-
  segmentation (FP blobs 33k->43k); threshold sweep on the final ckpt tops
  at 0.9423 @thr0.7, so the rebalance must happen in training, not at
  inference. Direction confirmed, dose too high and misdirected at large
  glaciers too.
r8 (2026.7.5): same recall term, two changes: (1) blob_weight 0.5->0.15
  (r7's recall gain saturated by ~ep30; the surplus push only bought FP);
  (2) the term now applies ONLY to components < 1024 px (small_max_area) —
  large glaciers already sit at 98.5%+ recall, pushing them just bloats
  boundaries; the misses live entirely in the small buckets.
  RESULT: peak 0.9494 @ep189 — still below the 0.9522 baseline. Post-hoc
  analyses closed the small-glacier route for the GLOBAL metric: (1) WiSE-FT
  weight interpolation baseline<->r7/r8 is monotone decreasing (best 0.9516
  @alpha 0.2); (2) merging r7's newly-detected small blobs into the baseline
  prediction is net-negative: new blobs not touching the baseline are ~82%
  FP by pixel mass (+3.1k TP vs +14.5k FP). At current feature quality the
  missed small glaciers are indistinguishable from lookalike bright patches.
  r7 weights kept for recall-priority use cases.
r9 (2026.7.5): attack the DOMINANT error mass instead: large-glacier
  boundaries own 59.8% of all FN px (~90k) and most of the ~157k FP px sit
  on boundaries too (~75% of total error mass in the boundary band).
  BOUNDARY-WEIGHTED BCE (U-Net weight map, Ronneberger et al., MICCAI 2015;
  boundary-focused losses: Kervadec et al., MIDL 2019; BASNet, CVPR 2019):
  per-pixel weight w = 1 + 3*exp(-d^2/(2*4^2)), d = Euclidean distance to
  the GT glacier boundary (both sides, distance_transform_edt in the
  dataloader workers). Applied to the main-output BCE only; Dice/Lovasz and
  aux unchanged. Symmetric FN/FP pressure -> no over-segmentation failure
  mode. Recall term OFF (blob_weight=0) for a clean A/B vs r2's 0.9521.
  RESULT: best 0.9512 (one ckpt saved, never higher) — ~neutral; the
  boundary band is near-irreducible with current features (mixed pixels /
  debris-covered margins).
r10 (2026.7.5): REMOVE H/V flip augmentation (boundary_w0=0 -> plain loss,
  recipe otherwise = r2). Motivation: TTA probe on the trained baseline
  showed a strong REAL orientation prior (identity 0.9522 vs hflip 0.9366,
  rot180 0.9343, rot90 0.9399 — sun-azimuth/shadow and aspect signal in
  mountain terrain), yet RandomH/VFlip(p=0.3 each) feeds ~half the training
  samples non-physical flipped-sun geometry that never occurs at test time.
  Augmentation should match the test distribution (affinity/diversity,
  Gontijo-Lopes et al., ICLR 2021). Kept: random crop, +/-15 deg rotation
  (~natural sun-azimuth variation), GaussianNoise.
  RESULT: BREAKTHROUGH — best 0.9544 (vs the 0.9521 plateau; beats the
  project-wide previous best 0.9533). Flip augmentation was the binding
  error all along. Greedy soup of the top r10 ckpts (Wortsman et al.,
  ICML 2022): 0.9548 (members raw_09544+09543+09540) ->
  u3net_cross_fusion_v2r10_soup09548.pth.
r11 (2026.7.5): same no-flip recipe, warm start from the r10 soup 0.9548
  (repeat the soup->retrain pattern that produced r10's jump); save line
  raised to 0.9540.
  RESULT: best 0.9546 (EMA) — below its own warm start soup. Combined
  greedy soup r10+r11 = 0.9548 (u3net_cross_fusion_v2_best09548.pth).
  Threshold sweep on it: 0.5 is already optimal (0.95483). Gap to target:
  0.00017.
r12 (2026.7.6): USER-DIRECTED fusion change — per-branch SELF-attention
  added as a third term next to the two cross-attention terms in each
  Cross_ViTransformerBlock_xyd (intra-modal + inter-modal attention:
  MulT, Tsai et al., ACL 2019; SwinFusion intra-/inter-domain attention,
  Ma et al., IEEE/CAA JAS 2022). Self-attn output projections are
  zero-initialized, so the warm start (best09548, val 0.9548) is
  numerically unchanged at step 0 and the new pathway grows only if
  useful. Params 27.3M -> ~32.3M (+18%); per the no-growth rule this
  reverts unless val mIoU improves. No-flip recipe, 800 ep, save 0.9544.
  RESULT: GOAL REACHED — EMA best 0.95505 @ep779 (raw 0.95491), >= 0.955
  target; ckpt u3net_cross_fusion_v2r12_ema_09550.pth. Param growth kept
  (justified by the lift 0.9548 -> 0.95505).
ABLATION arms A/B (2026.7.6, user-directed): prove the fusion module's
  contribution. Arm A = full model (bottleneck cross+self fusion), Arm B =
  bottleneck_fusion=False (encoder features pass straight to the decoder;
  NOTHING else differs). Both arms train FROM IMAGENET INIT (no warm start
  — warm-starting from a fusion-trained ckpt would contaminate the
  comparison), identical no-flip recipe, fixed seed 0, 800 ep, plain
  BCE+Dice+0.5*Lovasz + EMA, const lr 1e-4. Metric: best val mIoU (raw or
  EMA). Run B first (tag ablB_nofusion), then A (tag ablA_fusion).
'''

import sys
sys.path.append('/home/ps/Develop/dev-luo/GlaNet')
import time
import math
import torch
import numpy as np
import pandas as pd
from scipy import ndimage
import torch.nn as nn
from glob import glob
import torch.nn.functional as F
try: from notebooks import config
except ModuleNotFoundError: import config
from torchvision.transforms import v2
from utils.data_aug import GaussianNoise
from utils.dataloader import read_scenes
from utils.dataloader import SceneArraySet, PatchPathSet
from model.u3net_cross_fusion_v2 import u3net_cross_fusion_v2
from torchmetrics.classification import BinaryJaccardIndex, BinaryAccuracy

## 1. params
patch_size = 512
patch_resize = None
learning_rate = 1e-4          ## Run5: back to the r2 constant-LR recipe
lr_min = 1e-6                 ## cosine floor (unused when use_cosine=False)
warmup_epochs = 5
use_cosine = False            ## Run5: constant LR (peaks come mid-run, EMA/ckpt catch them)
epochs = 800
batch_size_tra = 8
batch_size_val = 8
use_amp = True                ## bf16 autocast for training
ema_decay = 0.999
shallow_fusion = False        ## windowed cross-modal fusion at skips 3/2
mod_drop = 0.                 ## r5 result: moddrop regressed (0.9513 < 0.9521) -> off
blob_weight = 0.              ## r8 closed the small-recall route -> off
small_max_area = 1024
boundary_w0 = 0.              ## r9 result: boundary weighting ~neutral -> off (wmap==1)
boundary_sigma = 4.0
bottleneck_fusion = True      ## ABLATION: True = with fusion module (Arm A)
seed = 0                      ## fixed seed so both ablation arms match
device = torch.device('cuda:0')
path_pretrained = None        ## ablation trains from ImageNet init (no warm start)
run_tag = 'ablA_fusion'       ## run tag for saved files
path_metrics = f'training_metrics_{run_tag}.csv'

torch.backends.cudnn.benchmark = True
torch.manual_seed(seed)

### traset
paths_scene_tra, paths_truth_tra = config.paths_scene_tra, config.paths_truth_tra
paths_dem_tra = config.paths_dem_tra
print(f'train scenes: {len(paths_scene_tra)}')

## valset
paths_valset = sorted(glob(f'data/dset/valset/patch_{patch_size}/*.pt'))
print(f'vali patch {patch_size}: {len(paths_valset)}')
path_valset_lat = f'data/dset/valset/patch_{patch_size}/patches_lat.json'

## 2. Read data
scenes_arr, truths_arr, scenes_lat = read_scenes(paths_scene_tra, paths_truth_tra, paths_dem_tra, lat=True)

## 3. dataloader
transforms_tra = v2.Compose([
            v2.ToImage(),
            v2.RandomCrop(size=(patch_size, patch_size)),
            ## r10: H/V flips removed — they flip the sun azimuth/aspect
            ## geometry, which is real signal in mountain scenes (TTA probe:
            ## hflip input costs -0.016 mIoU on the trained baseline).
            v2.RandomApply([v2.RandomRotation(degrees=15)], p=0.3),   # type:ignore
            GaussianNoise(mean = 0.0, sigma_max_img=0.1, sigma_max_dem=0, p=0.3)
            ])
transforms_val = v2.Compose([v2.ToDtype(torch.float32)])

class SceneArraySetBlob(SceneArraySet):
    '''r6: also return int16 connected-component labels of the (augmented)
    truth patch, computed in the dataloader workers (free on the GPU path).
    r9: plus a boundary weight map w = 1 + w0*exp(-d^2/2s^2), d = distance
    to the GT boundary on either side (Ronneberger et al., MICCAI 2015).'''
    def __getitem__(self, idx):
        patch, ptruth, lat = super().__getitem__(idx)
        mask = np.asarray(ptruth[0]) > 0.5
        comp, _ = ndimage.label(mask)
        if mask.any() and not mask.all():
            d_in = ndimage.distance_transform_edt(mask)
            d_out = ndimage.distance_transform_edt(~mask)
            d = np.where(mask, d_in, d_out).astype(np.float32)
        else:
            d = np.full(mask.shape, 1e6, dtype=np.float32)
        wmap = 1. + boundary_w0 * np.exp(-d**2 / (2. * boundary_sigma**2))
        return (patch, ptruth, torch.from_numpy(comp.astype(np.int16)),
                torch.from_numpy(wmap)[None], lat)

tra_data = SceneArraySetBlob(scenes_arr=scenes_arr, truths_arr=truths_arr,
                              patch_size=patch_size, transforms=transforms_tra, scenes_lat=scenes_lat)
val_data = PatchPathSet(paths_valset=paths_valset, transforms=transforms_val, path_valset_lat=path_valset_lat)

tra_loader = torch.utils.data.DataLoader(tra_data, batch_size=batch_size_tra,
                                         shuffle=True, num_workers=5)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size_val, num_workers=5)

## 4. model, loss and optimizer
model = u3net_cross_fusion_v2(backbone_name='efficientnet_b0',
                              pretrained=True,
                              shallow_fusion=shallow_fusion,
                              mod_drop=mod_drop,
                              bottleneck_fusion=bottleneck_fusion)
print(f'bottleneck_fusion={bottleneck_fusion}, '
      f'params={sum(p.numel() for p in model.parameters()):,}')

# 4.1 load pretrained weights (partial, name+shape matched)
if path_pretrained is None:
    model_checkpoint = {}
    print('no warm start: training from ImageNet-initialized encoders')
else:
    model_checkpoint = torch.load(path_pretrained, map_location='cpu')
model_dict = model.state_dict()
pretrained_dict = {
    k: v for k, v in model_checkpoint.items()
    if k in model_dict and v.shape == model_dict[k].shape}
if path_pretrained is not None:
    discarded_keys = set(model_checkpoint.keys()) - set(pretrained_dict.keys())
    if discarded_keys:
        print(f"unmatched parameters and are discarded ({len(discarded_keys)}):")
        for k in sorted(discarded_keys):
            print(f"  - {k}")
    newly_keys = set(model_dict.keys()) - set(pretrained_dict.keys())
    if newly_keys:
        print(f"newly added parameters and are randomly initialized ({len(newly_keys)}):")
        for k in sorted(newly_keys):
            print(f"  - {k}")
    print(f"loaded {len(pretrained_dict)}/{len(model_dict)} tensors from {path_pretrained}")
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

### loss: deep-supervised BCE + Dice
def bce_dice(logit, target, smooth=1.0, wmap=None):
    '''wmap: optional per-pixel BCE weight (r9 boundary weighting).'''
    logit = logit.float()
    bce = F.binary_cross_entropy_with_logits(logit, target, weight=wmap)
    prob = torch.sigmoid(logit)
    inter = (prob * target).sum(dim=(1, 2, 3))
    union = prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2. * inter + smooth) / (union + smooth)
    return 0.5 * bce + 0.5 * (1. - dice.mean())

def deep_bce_dice_loss(outputs, target):
    '''outputs: (main, aux4, aux3, aux2) logits; target: [B,1,H,W]'''
    main_logit, aux4_logit, aux3_logit, aux2_logit = outputs
    aux4_target = F.interpolate(target, size=aux4_logit.shape[2:], mode='area')
    return bce_dice(main_logit, target) + bce_dice(aux4_logit, aux4_target)

### Lovasz hinge loss (Berman et al., CVPR 2018) — convex surrogate of Jaccard/IoU
def _lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_hinge(logit, target):
    '''logit: [B,1,H,W] raw logits; target: [B,1,H,W] in {0,1} (soft targets are thresholded)'''
    logit = logit.float().view(-1)
    label = (target.view(-1) > 0.5).float()
    signs = 2. * label - 1.
    errors = 1. - logit * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    gt_sorted = label[perm]
    grad = _lovasz_grad(gt_sorted)
    return torch.dot(F.relu(errors_sorted), grad)

def deep_bce_dice_lovasz_loss(outputs, target, wmap=None):
    '''main: BCE+Dice+Lovasz; aux4: BCE+Dice (low-res soft targets unsuited to hinge).
    wmap: optional boundary weight map, applied to the MAIN BCE only (r9).'''
    main_logit, aux4_logit, aux3_logit, aux2_logit = outputs
    aux4_target = F.interpolate(target, size=aux4_logit.shape[2:], mode='area')
    return (bce_dice(main_logit, target, wmap=wmap)
            + 0.5 * lovasz_hinge(main_logit, target)
            + bce_dice(aux4_logit, aux4_target))

### blob loss (Kofler et al., arXiv 2022 / IPMI 2023) — instance-imbalance-aware:
### per GT connected component, soft Dice of that blob vs the prediction with the
### other blobs masked out (background FP mass is shared by all blobs), then mean
### over blobs. A 30-px glacier weighs the same as a 30k-px one.
def blob_dice(logit, comp, smooth=1.0):
    '''logit: [B,1,H,W]; comp: [B,H,W] int labels, 0 = background.
    Vectorized: per-component prob mass and area via one index_add/bincount,
    dice_c = (2*p_c + s) / (p_c + p_bg + area_c + s).'''
    prob = torch.sigmoid(logit.float())[:, 0]
    losses = []
    for b in range(prob.shape[0]):
        cb = comp[b].long().flatten()
        n = int(cb.max())
        if n == 0:
            continue
        p = prob[b].flatten()
        pmass = torch.zeros(n + 1, device=p.device, dtype=p.dtype).index_add_(0, cb, p)
        area = torch.bincount(cb, minlength=n + 1).to(p.dtype)
        dice = (2. * pmass[1:] + smooth) / (pmass[1:] + pmass[0] + area[1:] + smooth)
        losses.append(1. - dice)
    if not losses:
        return prob.sum() * 0.
    return torch.cat(losses).mean()

### r7: per-component soft RECALL (r6 lesson: blob dice's shared-background
### denominator turns the term into an FP-suppressor and INCREASES misses).
### Each GT component only pushes its own pixels up, weight ~ 1/area:
### small glaciers get a real gradient, FP control stays with the global loss.
def blob_recall(logit, comp, smooth=1.0, max_area=None):
    '''logit: [B,1,H,W]; comp: [B,H,W] int labels, 0 = background.
    loss = mean over GT components of 1 - (p_c + s) / (area_c + s).
    max_area: if set, only components smaller than this contribute (r8 —
    large glaciers are already recalled, pushing them only bloats FP).'''
    prob = torch.sigmoid(logit.float())[:, 0]
    losses = []
    for b in range(prob.shape[0]):
        cb = comp[b].long().flatten()
        n = int(cb.max())
        if n == 0:
            continue
        p = prob[b].flatten()
        pmass = torch.zeros(n + 1, device=p.device, dtype=p.dtype).index_add_(0, cb, p)
        area = torch.bincount(cb, minlength=n + 1).to(p.dtype)
        loss_c = 1. - (pmass[1:] + smooth) / (area[1:] + smooth)
        if max_area is not None:
            loss_c = loss_c[area[1:] < max_area]
        losses.append(loss_c)
    if not losses:
        return prob.sum() * 0.
    losses = torch.cat(losses)
    if losses.numel() == 0:
        return prob.sum() * 0.
    return losses.mean()

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
if use_cosine:
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=lr_min)],
        milestones=[warmup_epochs])
else:
    lr_scheduler = None

### EMA of model weights
class ModelEMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.updates = 0
        self.shadow = {k: v.detach().clone().float() if v.dtype.is_floating_point
                       else v.detach().clone()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = min(self.decay, (1. + self.updates) / (10. + self.updates))  # ramp-up
        for k, v in model.state_dict().items():
            if self.shadow[k].device != v.device:
                self.shadow[k] = self.shadow[k].to(v.device)
            s = self.shadow[k]
            if v.dtype.is_floating_point:
                s.mul_(d).add_(v.detach().float(), alpha=1. - d)
            else:
                s.copy_(v)

    def state_dict(self):
        return self.shadow

## 5. train and val loops
@torch.no_grad()
def evaluate(model, val_loader, loss_fn, device):
    model.eval()
    miou = BinaryJaccardIndex().to(device)
    oa = BinaryAccuracy().to(device)
    loss_val = 0.
    for x_batch, y_batch, lat_batch in val_loader:
        x_batch, y_batch, lat_batch = x_batch.to(device), y_batch.to(device), lat_batch.to(device)
        preds = model(x_batch, lat=lat_batch)
        loss_val += loss_fn(preds, y_batch).item()
        pred = (torch.sigmoid(preds[0]) > 0.5).float()
        miou.update(pred, y_batch.long())
        oa.update(pred, y_batch.long())
    return loss_val / len(val_loader), miou.compute().item(), oa.compute().item()

def train_loops(model, loss_fn, optimizer, tra_loader, val_loader,
                epochs, device, lr_scheduler=None, ema=None):
    metrics_rows = []
    model = model.to(device)
    size_tra_loader = len(tra_loader)
    best_miou = 0.9500
    for epoch in range(epochs):
        start = time.time()
        loss_tra = 0.
        miou_tra = BinaryJaccardIndex().to(device)
        oa_tra = BinaryAccuracy().to(device)
        model.train()
        for x_batch, y_batch, comp_batch, w_batch, lat_batch in tra_loader:
            x_batch, y_batch, lat_batch = x_batch.to(device), y_batch.to(device), lat_batch.to(device)
            comp_batch, w_batch = comp_batch.to(device), w_batch.to(device)
            optimizer.zero_grad()
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                preds = model(x_batch, lat=lat_batch)
                loss = loss_fn(preds, y_batch, wmap=w_batch)
                if blob_weight > 0:
                    loss = loss + blob_weight * blob_recall(preds[0], comp_batch, max_area=small_max_area)
            loss.backward()
            optimizer.step()
            if ema is not None:
                ema.update(model)
            pred = (torch.sigmoid(preds[0].float()) > 0.5).float()
            miou_tra.update(pred, y_batch.long())
            oa_tra.update(pred, y_batch.long())
            loss_tra += loss.item()
        miou_tra_global = miou_tra.compute().item()
        oa_tra_global = oa_tra.compute().item()
        loss_tra_global = loss_tra / size_tra_loader

        '''----- validation (raw weights + EMA weights) -----'''
        if (epoch + 1) % 2 == 0:
            loss_val, miou_val, oa_val = evaluate(model, val_loader, loss_fn, device)
            miou_ema = float('nan')
            if ema is not None:
                backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
                model.load_state_dict(ema.state_dict())
                _, miou_ema, _ = evaluate(model, val_loader, loss_fn, device)
                model.load_state_dict(backup)
            print(f'Ep{epoch}: tra-> Loss:{loss_tra_global:.3f},Oa:{oa_tra_global:.3f},Miou:{miou_tra_global:.3f}, '
                  f'val-> Loss:{loss_val:.3f},Oa:{oa_val:.3f},Miou:{miou_val:.3f},MiouEMA:{miou_ema:.4f}, '
                  f'lr:{optimizer.param_groups[0]["lr"]:.2e},time:{time.time()-start:.1f}s', flush=True)
            metrics_rows.append({'epoch': epoch, 'tra_loss': loss_tra_global, 'tra_oa': oa_tra_global,
                                 'tra_miou': miou_tra_global, 'val_loss': loss_val, 'val_oa': oa_val,
                                 'val_miou': miou_val, 'val_miou_ema': miou_ema})
            pd.DataFrame(metrics_rows).to_csv(path_metrics, index=False)
            ## save the best model (raw or EMA, whichever improves)
            if miou_val > best_miou:
                best_miou = miou_val
                torch.save(model.state_dict(),
                           f'model/trained/u3net_cross_fusion_{run_tag}_raw_0{round(best_miou*10000)}.pth')
                print(f'  >> saved RAW ckpt, val miou {best_miou:.4f}', flush=True)
            if ema is not None and miou_ema > best_miou:
                best_miou = miou_ema
                torch.save(ema.state_dict(),
                           f'model/trained/u3net_cross_fusion_{run_tag}_ema_0{round(best_miou*10000)}.pth')
                print(f'  >> saved EMA ckpt, val miou {best_miou:.4f}', flush=True)
        else:
            print(f'Ep{epoch}: tra-> Loss:{loss_tra_global:.3f},Oa:{oa_tra_global:.3f},Miou:{miou_tra_global:.3f}, '
                  f'time:{time.time()-start:.1f}s', flush=True)
        if lr_scheduler:
            lr_scheduler.step()
    print(f'BEST val miou: {best_miou:.4f}', flush=True)
    return metrics_rows

if __name__ == '__main__':
    ema = ModelEMA(model, decay=ema_decay)
    metrics = train_loops(model=model,
                    loss_fn=deep_bce_dice_lovasz_loss,
                    optimizer=optimizer,
                    tra_loader=tra_loader,
                    val_loader=val_loader,
                    epochs=epochs,
                    lr_scheduler=lr_scheduler,
                    device=device,
                    ema=ema)
    torch.save(model.state_dict(), f'model/trained/u3net_cross_fusion_{run_tag}_final.pth')
    pd.DataFrame(metrics).to_csv(path_metrics, index=False)
