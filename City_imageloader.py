# -*- coding: utf-8 -*-
"""
Created on Sat May 21 10:17:03 2022

@author: Jean-
"""
# Packages
import numpy as np
import matplotlib.pyplot as plt
import torch
import glob


class CityscapeDataset(object):
    '''
    load_shapes()
    load_image()
    load_mask()
    '''

    def __init__(self,root ,  subset,as_tensor = True):
        
        self.subset = subset
        self.root = root
        #self.data = torchvision.datasets.Cityscapes(root,split=subset, mode='fine', target_type=['instance'], transform=None)
        self.img_paths = glob.glob(root + 'leftImg8bit/' + subset + '/*/*_leftImg8bit.png')
        self.img_paths.sort()
        self.as_tensor = as_tensor
        print('num_images: ',len(self.img_paths))
        
    def __len__(self):
       return 10 #len(self.img_paths)
   
    def __getitem__(self, index):

        img =plt.imread(self.img_paths[index])
        #img = torch.as_tensor(img, dtype=torch.uint8)
            
           
        return img