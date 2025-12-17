''' 
author: xin luo, 
created: 2023.4.2
modified: 2025.12.10
Configure parameters for the model training
'''

import torch.nn as nn
from glob import glob

#### ------------- Model -------------
model_name = 'deeplabv3plus'  ### option: unet, deeplabv3plus, deeplabv3plus_mobilev2
num_dataloader_worker = 10
#### ------------- Training parameters -------------
patch_size = 768    ###  
num_epoch = 200

lr = 0.0005                   ## if use lr_scheduler;
batch_size_tra = 4            ## 
batch_size_val = 4            ## 
loss_bce = nn.BCELoss()       ## selected for binary classification
### --- path to save
path_weights_save = 'model/trained/patch_' + str(patch_size) + '/' + model_name+'_weights.pth'
path_metrics_save = 'model/trained/patch_' + str(patch_size) + '/' + model_name + '_metrics.csv'
### --- number of bands of the image.
num_bands = 7

#### ------------- Directories/files -------------
dir_dset = 'data/dset/'
dir_scene = 'data/dset/scene/'
dir_truth = 'data/dset/truth/'
dir_dem = 'data/dset/dem/'
paths_truth = sorted(glob('data/dset/truth/*.tif'))
paths_truth_vec = sorted(glob('data/dset/truth/*.gpkg'))
paths_scene = [path.replace('truth','scene') for path in paths_truth]
paths_dem = [path.replace('.tif', '_dem.tif').replace('truth','dem') for path in paths_truth]

## training/validation split
ids_scene = [path.split('/')[-1].split('.')[0] for path in paths_truth_vec] 
ids_scene_val = ids_scene[::4]  ## every 4th scene for validation
ids_scene_tra = sorted(list(set(ids_scene) - set(ids_scene_val)))
## traset
paths_scene_tra = [dir_scene+id+'_nor.tif' for id in ids_scene_tra]
paths_dem_tra = [dir_dem+id+'_dem_nor.tif' for id in ids_scene_tra]
paths_truth_tra = [dir_truth+id+'.tif' for id in ids_scene_tra] 
## valset
dir_valset = 'data/dset/valset/patch_'+str(patch_size)



# ## ------------- Data tranform/augmentation -------------
# transforms_tra = [
#         colorjitter(prob=0.25, alpha=0.05, beta=0.05),    # numpy-based, !!!beta should be small（防止过度变换） 颜色变换
#         rotate(prob=0.25),           # numpy-based 旋转
#         flip(prob=0.25),             # numpy-based 翻转
#         numpy2tensor(),              # numpy转tensor
#         torch_noise(prob=0.25, std_min=0, std_max=0.1),      # tensor-based 噪声变换
#             ]


