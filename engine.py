
import torch
import random
from PIL import Image, ImageOps, ImageFilter
import utils
import numpy as np
import torchvision
import torchvision.transforms as T





def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, learning_rate, lambd = 0.0051):

    #in abhängigkeit der epoche wird learning rate und hard ratio verkleinert
    def adjust_lr(optimizer, ep ,learning_rate): # function just exist, if top method is run
        
        if ep < 3:# TODO: 10 # warming up
            lr = learning_rate

        elif ep < 30:
            lr = learning_rate * 3 / 10

        elif ep < 55:
            lr = learning_rate / 10

        elif ep < 80:
            lr = learning_rate / 100 * 5

        elif ep < 160:
            lr = learning_rate / 100

        else: # selbstständig hinzugefügt und ander schrittwete abgeändert
            lr = learning_rate / 1000 * 5

       
        for p in optimizer.param_groups:
            p['lr'] = lr

        return lr
    
    def augmentation(image):
        l = range(10)
        aug = random.sample(l,2)
        aug = [0,0]
        if aug[0] == 0:
            transform = T.GaussianBlur(kernel_size=(7, 13), sigma=(0.1,0.2))
            Img_view1 = image
        # elif aug[0] == 1:
        # elif aug[0] == 2:
        # elif aug[0] == 3:
        # elif aug[0] == 4:
        # elif aug[0] == 5:
        # elif aug[0] == 6:
        # elif aug[0] == 7:
        # elif aug[0] == 8:
        # elif aug[0] == 9:
            
        if aug[1] == 0:
            transform = T.GaussianBlur(kernel_size=(7, 13), sigma=(0.1,0.2))
            Img_view2 = image
        # elif aug[1] == 1:
        # elif aug[1] == 2:
        # elif aug[1] == 3:
        # elif aug[1] == 4:
        # elif aug[1] == 5:
        # elif aug[1] == 6:
        # elif aug[1] == 7:
        # elif aug[1] == 8:
        # elif aug[1] == 9:
        return Img_view1, Img_view2
    
    def off_diagonal(x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
            # import ipdb
            # ipdb.set_trace()
    losses = 0
    for images in metric_logger.log_every(data_loader, print_freq, header):
        #for image in images:
        images_view1, images_view2 = augmentation(images)

        
        # compute embeddings
        import ipdb
        ipdb.set_trace()
        
        z_view1 = model(images_view1) # NxD
        z_view2 = model(images_view2) # NxD
        
        # normalize repr. along the batch dimension
        z_view1_norm = (z_view1 - z_view1.mean(0)) / z_view1.std(0) # NxD
        z_view2_norm = (z_view2 - z_view2.mean(0)) / z_view2.std(0) # NxD
        
        # cross-correlation matrix
        c = z_view1_norm.T @ z_view2_norm # DxD
        
        
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + lambd * off_diag
        losses += loss
        
        rate = adjust_lr(optimizer, epoch + 1 ,learning_rate) # einstellen von Learning rate (in abhängigkeit von der Epoche) 

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        optimizer.zero_grad()

        metric_logger.update(loss=losses)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return losses #  das habe ich hinzugefügt für tensorboard
