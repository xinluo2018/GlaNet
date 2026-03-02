'''
author: xin luo, 
created: 2025.11.30
des: configuration file
'''
from glob import glob
## directories/files
dir_scene = 'data/dset/scene/scene_nor/'
dir_dem = 'data/dset/dem/dem_nor/' 
dir_truth = 'data/dset/truth/truth_tif/' 
dir_result = 'data/result/'
paths_truth = sorted(glob('data/dset/truth/truth_tif/*.tif'))
paths_scene = sorted(glob('data/dset/scene/scene_nor/*.tif'))
paths_dem = sorted(glob('data/dset/dem/dem_nor/*.tif'))

## training/validation split
ids_dset = list(range(len(paths_truth)))
ids_val = ids_dset[::5]  ## every 5th scene for validation
ids_tra = sorted(list(set(ids_dset) - set(ids_val)))
## traset
paths_scene_tra = [paths_scene[i] for i in ids_tra]
paths_truth_tra = [paths_truth[i] for i in ids_tra]
paths_dem_tra = [paths_dem[i] for i in ids_tra]
## valset(scene)
paths_scene_val = [paths_scene[i] for i in ids_val]
paths_truth_val = [paths_truth[i] for i in ids_val] 
paths_dem_val = [paths_dem[i] for i in ids_val]

# ## max and min values for different satellites' scenes (obtained from notebooks/2_dset_check.ipynb)
# max_scenes = {'l5': 65454.0, 'l7': 56297.0, 
#               'l8': 65439.0, 'l9': 65453.0, 's2': 19312.0} 
# min_scenes = {'l5': 4891.0, 'l7': 6719.0,  
#               'l8': 1.0, 'l9': 1.0, 's2': 1.0}
### scale and offset are given from GEE platform.  
scale = {'l5': 2.75e-05, 'l7': 2.75e-05,
                  'l8': 2.75e-05, 'l9': 2.75e-05, 's2': 0.0001} 
offset = {'l5': -0.2, 'l7': -0.2, 
                  'l8': -0.2, 'l9': -0.2, 's2': 0} 
max_dem = 8848.0  # highest point on Earth: Mount Everest
min_dem = -420.0  # lowest point on Earth: Dead Sea Shore 


