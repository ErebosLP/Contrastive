
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
    
    
    def off_diagonal(x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    import ipdb

    
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    losses = 0

    for images in metric_logger.log_every(data_loader, print_freq, header):

        images_view1 = torch.unsqueeze(images[0][0].to(device),0)
        images_view2 = torch.unsqueeze(images[0][1].to(device),0)

        z_view1 = model.forward(images_view1) # NxD
        z_view2 = model.forward(images_view2) # NxD

        # normalize repr. along the batch dimension
        z_view1_norm = (z_view1 - z_view1.mean()) / z_view1.std() # NxD
        z_view2_norm = (z_view2 - z_view2.mean()) / z_view2.std() # NxD
        
        
        # cross-correlation matrix
        c = z_view1_norm.T @ z_view2_norm # DxD
        import ipdb
        ipdb.set_trace()
        
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + lambd * off_diag
        #losses += loss

        rate = adjust_lr(optimizer, epoch + 1 ,learning_rate) # einstellen von Learning rate (in abhängigkeit von der Epoche) 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #optimizer.zero_grad()
        
        
        metric_logger.update(loss=loss)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return loss #  das habe ich hinzugefügt für tensorboard
