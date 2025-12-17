## author: xin luo
## create: 2025.11.21
## des: data preprocessing


import random
import numpy as np

class Crop:
    '''randomly crop corresponding to specific patch size'''
    def __init__(self, size=(256, 256)):
        self.size = size
    def __call__(self, image, truth):
        '''size: (height, width)'''
        start_h = random.randint(0, truth.shape[0]-self.size[0])
        start_w = random.randint(0, truth.shape[1]-self.size[1])
        patch = image[:,start_h:start_h+self.size[0],start_w:start_w+self.size[1]]
        truth = truth[start_h:start_h+self.size[0], start_w:start_w+self.size[1]]
        return patch, truth

class Normalize:
    '''
    des: normalize each band to [0, 1] according to given max and min values
    '''
    def __init__(self, max_bands, min_bands):
        '''        
        max_bands: list of max values for each band or single value for all bands
        min_bands: list of min values for each band or single value for all bands
        '''
        self.max = max_bands  
        self.min = min_bands          
    def __call__(self, image):
        if isinstance(self.max, (int, float)):
            self.max = [self.max] * image.shape[-1]  
        if isinstance(self.min, (int, float)):
            self.min = [self.min] * image.shape[-1]         
        normalized = []
        for b in range(image.shape[-1]):  
            band = image[:, :, b].astype(float)   ### 
            band_norm = (band - self.min[b]) / (self.max[b] - self.min[b] + 1e-6)
            normalized.append(band_norm)  
        normalized = np.stack(normalized, axis=-1)  
        return np.clip(normalized, 0.0, 1.0)  


def normalize_scene(image, max_bands, min_bands):
    '''
    des: normalize each band of the image to [0, 1]
    args:
        image: H*W*C, numpy array
        max_bands: list of max values for each band (or single value for all bands)
        min_bands: list of min values for each band (or single value for all bands)
    '''
    if isinstance(max_bands, (int, float)):
        max_bands = [max_bands] * image.shape[-1]  
    if isinstance(min_bands, (int, float)):
        min_bands = [min_bands] * image.shape[-1]         
    ## normalization
    min_bands_arr = np.array(min_bands).reshape(1, 1, -1)
    max_bands_arr = np.array(max_bands).reshape(1, 1, -1)
    normalized = (image.astype(float) - min_bands_arr) \
                     / (max_bands_arr - min_bands_arr + 1e-6)
    return np.clip(normalized, 0.0, 1.0)  