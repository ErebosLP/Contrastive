# -*- coding: utf-8 -*-
"""
Created on Sat May 21 10:17:03 2022

@author: Jean-
"""
# Packages
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import torch
import glob
from PIL import Image  


class CityscapeDataset(object):
    '''
    load_shapes()
    load_image()
    load_mask()
    '''

    def __init__(self,root ,  subset,transform,as_tensor = True,):
        
        self.subset = subset
        self.root = root
        self.img_paths = glob.glob(root + 'leftImg8bit/' + subset + '/*/*_leftImg8bit.png')
        self.img_paths.sort()
        self.as_tensor = as_tensor
        print('num_images: ',len(self.img_paths))
        self.transform = transform
    def __len__(self):
       return 2 #len(self.img_paths)
   
    def __getitem__(self, index):

        #img =plt.imread(self.img_paths[index])
        img = Image.open(self.img_paths[index])
        img2 = Image.open(self.img_paths[index-1])
        img = T.Resize((256,256))(img) 
        img2 = T.Resize((256,256))(img2) 
        #img = T.ToTensor()(img)
        
        #img = self.tensorize(img)
        #img = torch.as_tensor(img, dtype=torch.uint8)
            
           
        return self.transform(img,img2)