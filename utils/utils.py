'''
author: xin luo
create: 2025.12.07
des: functions for data processing
'''


import rasterio as rio 
import numpy as np

def read_scenes(scene_paths, truth_paths, dem_paths):
  paths_zip = zip(scene_paths, truth_paths, dem_paths)
  scenes_arr = []
  truths_arr = []
  for scene_path, truth_path, dem_path in paths_zip:
      ## 1. read scene and truth images
      with rio.open(scene_path) as src:
          scene_arr = src.read().transpose((1, 2, 0))  # (H, W, C)
      with rio.open(truth_path) as truth_src:
          truth_arr = truth_src.read(1)  # (H, W)
      ## 2. read dem
      with rio.open(dem_path) as dem_src:
          dem_arr = dem_src.read(1)  # (H, W)
      dem_arr = dem_arr[:, :, np.newaxis]  # expand to (H, W, 1)
      scene_arr = np.concatenate([scene_arr, dem_arr], axis=-1)  # (H, W, C+1)
      scenes_arr.append(scene_arr)
      truths_arr.append(truth_arr)
  return scenes_arr, truths_arr