'''
author: xin luo
create: 2025.12.07
des: functions for data processing
'''

import numpy as np
import rasterio as rio 
from rasterio.warp import transform_bounds

def get_lat(img_rio):
    wgs84_bounds = transform_bounds(src_crs=img_rio.crs, 
                                    dst_crs='EPSG:4326', 
                                    left=img_rio.bounds.left, 
                                    bottom=img_rio.bounds.bottom, 
                                    right=img_rio.bounds.right, 
                                    top=img_rio.bounds.top) 
    lat_deg = (wgs84_bounds[1] + wgs84_bounds[3]) / 2
    return lat_deg

def read_scenes(scene_paths, truth_paths, dem_paths):
  paths_zip = zip(scene_paths, truth_paths, dem_paths)
  scenes_arr = []
  truths_arr = []
  scenes_lat = []
  for scene_path, truth_path, dem_path in paths_zip:
      ## 1. read scene and truth images
      with rio.open(scene_path) as src:
        scene_arr = src.read().transpose((1, 2, 0))  # (H, W, C)
        scene_lat = get_lat(src) 
      with rio.open(truth_path) as truth_src:
        truth_arr = truth_src.read(1)  # (H, W)
      ## 2. read dem
      with rio.open(dem_path) as dem_src:
        dem_arr = dem_src.read(1)  # (H, W)
      dem_arr = dem_arr[:, :, np.newaxis]  # expand to (H, W, 1)
      scene_arr = np.concatenate([scene_arr, dem_arr], axis=-1)  # (H, W, C+1)
      scenes_arr.append(scene_arr)
      truths_arr.append(truth_arr)
      scenes_lat.append(scene_lat)
  return scenes_arr, truths_arr, scenes_lat
