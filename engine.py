
import torch
import random
from PIL import Image, ImageOps, ImageFilter
import utils
import numpy as np
import torchvision
import torchvision.transforms as T




def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scheduler, lambd = 0.0051):
    torch.cuda.empty_cache()
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    

    for images in metric_logger.log_every(data_loader, print_freq, header):
        weakly_correlated_loss = 0.0
        for i in range(images.shape[0]):

            images_view1 = torch.unsqueeze(images[i][0].to(device),0)
            images_view2 = torch.unsqueeze(images[i][1].to(device),0)
            
            z_view1 = model.forward(images_view1) # NxD
            z_view1 = torch.squeeze(z_view1,0)
            
            z_view2 = model.forward(images_view2) # NxD
            z_view2 = torch.squeeze(z_view2,0)
            
            # normalize repr. along the batch dimension
            z_view1_norm = (z_view1 - z_view1.mean()) / z_view1.std() # NxD
            z_view2_norm = (z_view2 - z_view2.mean()) / z_view2.std() # NxD
            
            #D = z_view1.shape[1]
            
            # cross-correlation matrix
            
            n_samples = 64 
            idx =  random.choices(range((z_view1_norm[0,:,:].shape[0]*z_view1_norm[0,:,:].shape[1])), k=n_samples*n_samples)
            
            for i in range(z_view1_norm.shape[0]):
                z_view1_vec = z_view1_norm[i,:,:].reshape((z_view1_norm[i,:,:].shape[1]*z_view1_norm[i,:,:].shape[0]),1)[idx]
                z_view2_vec = z_view2_norm[i,:,:].reshape((z_view2_norm[i,:,:].shape[1]*z_view2_norm[i,:,:].shape[0]),1)[idx]
                cross_weakly = z_view1_vec @ z_view2_vec.mT # DxD
                c_diff = cross_weakly - torch.eye(n_samples*n_samples).cuda()
                weakly_correlated_loss += c_diff.sum()
        weakly_correlated_loss = ((weakly_correlated_loss/(n_samples*n_samples))/z_view1_norm.shape[0])/images.shape[0] # nomalizing the loss by dividing with batchsize, number of sampeled points, number of embeddings
        optimizer.zero_grad()
        weakly_correlated_loss.backward()
        optimizer.step()        
        
        metric_logger.update(loss=weakly_correlated_loss)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    scheduler.step()
    return weakly_correlated_loss 
