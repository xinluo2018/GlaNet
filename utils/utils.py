'''
author: xin luo
create: 2025.12.07, modify: 2026.3.2
des: functions for data processing
'''

import numpy as np
import rasterio as rio 
from rasterio.warp import transform_bounds

## delete
def get_lat(img_rio):
    wgs84_bounds = transform_bounds(src_crs=img_rio.crs, 
                                    dst_crs='EPSG:4326', 
                                    left=img_rio.bounds.left, 
                                    bottom=img_rio.bounds.bottom, 
                                    right=img_rio.bounds.right, 
                                    top=img_rio.bounds.top) 
    lat_deg = (wgs84_bounds[1] + wgs84_bounds[3]) / 2
    return lat_deg
