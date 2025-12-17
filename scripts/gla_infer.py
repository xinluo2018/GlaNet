''' 
author: xin luo
create: 2025.12.15
des: glacier mappping from remote sensing images using the trained GlaNet model.
'''
import torch
import numpy as np
import rasterio as rio
from model import unet
from notebooks import config
from utils.img2patch import img2patch

## glacier inference 
def gla_infer(model, img, patch_size=1024):
    ### image to patches
    imgPat_obj = img2patch(img=img, patch_size=patch_size, edge_overlay = 40)
    patch_arr_list = imgPat_obj.toPatch()
    ## 1. channel first, and 2. numpy array to torch tensor
    patch_list = [torch.from_numpy(patch.transpose(2,0,1)).float() for patch in patch_arr_list]  
    with torch.no_grad():  ## save 
        presult_list = [model(patch[np.newaxis, :]) for patch in patch_list]
    ## 1. channel last, and 2.torch tensor to numpy array
    presult_arr_list = [np.squeeze(patch.detach().numpy().transpose(0,2,3,1), axis = 0) 
                                    for patch in presult_list ]       
    ## patch to image
    gla_pred_prob = imgPat_obj.toImage(presult_arr_list)
    gla_pred_cla = np.where(gla_pred_prob>0.5, 1, 0)
    gla_pred_cla = gla_pred_cla.astype(np.uint8)
    return gla_pred_cla

if __name__ == '__main__':
    ## 1. load model
    patch_size = 512
    path_trained_model = 'model/trained/patch_' + str(patch_size) + '/unet_weights.pth'
    model = unet(num_bands=7)
    model.load_state_dict(torch.load(path_trained_model, weights_only=True))
    model.eval();  ##  

    ## 2. load data 
    for id in range(len(config.paths_scene_val)):
        path_scene = config.paths_scene_val[id]
        path_dem = config.paths_dem_val[id]
        path_truth = config.paths_truth_val[id]
        scene_id = path_truth.split('/')[-1].split('.')[0]
        print('scene id:', scene_id)
        with rio.open(path_truth) as truth_rio:
            truth_arr = truth_rio.read(1)  # (H,W)
            profile_truth = truth_rio.profile
        with rio.open(path_scene) as scene_rio, rio.open(path_dem) as dem_rio:
            scene_dem_arr = np.concatenate([scene_rio.read(), dem_rio.read()], axis=0)
            scene_dem_arr = scene_dem_arr.transpose((1, 2, 0))  # (H,W,C)

        ## 3. inference
        gla_pred_cla = gla_infer(model, scene_dem_arr, patch_size=patch_size)

        ## 4. write the result to path
        path_gla_pred_cla = config.dir_result + scene_id + \
                                '_pred_cla_p' + str(patch_size) + \
                                '.tif'   ## path to save result
        with rio.open(path_gla_pred_cla, 'w', **profile_truth) as dst:
            dst.write(gla_pred_cla[:,:,0], 1)  # write to the first band
        print('results saved to:', path_gla_pred_cla)
