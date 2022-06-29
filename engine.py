
import torch
import random
from PIL import Image, ImageOps, ImageFilter
import utils
import numpy as np
import torchvision
import torchvision.transforms as T




def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scheduler, lambd = 0.0051):

    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    weakly_correlated_loss = 0.0

    for images in metric_logger.log_every(data_loader, print_freq, header):

        images_view1 = torch.unsqueeze(images[0][0].to(device),0)
        images_view2 = torch.unsqueeze(images[0][1].to(device),0)

        z_view1 = model.forward(images_view1) # NxD
        z_view1 = torch.squeeze(torch.squeeze(z_view1,2),2)
        
        z_view2 = model.forward(images_view2) # NxD
        z_view2 = torch.squeeze(torch.squeeze(z_view2,2),2)
        
        # normalize repr. along the batch dimension
        z_view1_norm = (z_view1 - z_view1.mean()) / z_view1.std() # NxD
        z_view2_norm = (z_view2 - z_view2.mean()) / z_view2.std() # NxD
        
        #D = z_view1.shape[1]
        
        # cross-correlation matrix
        cross_weakly = z_view1_norm @ z_view2_norm.T # DxD

        #cross_matrix = cross_weakly.repeat_interleave(D,0).repeat_interleave(D,1)
        
        #identity_stacked_matrix = torch.eye(D).cuda()
        
        c_diff= cross_weakly - 1 #cross_matrix - identity_stacked_matrix
        #c_diff[(identity_stacked_matrix == 0.)] *= self.lambda_param

        weakly_correlated_loss = c_diff #.sum() # / D
        

        optimizer.zero_grad()
        weakly_correlated_loss.backward()
        optimizer.step()        
        
        metric_logger.update(loss=weakly_correlated_loss)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    scheduler.step()
    return weakly_correlated_loss 
