
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
    cum_loss = 0.0
    num_batches = 0

    for images in metric_logger.log_every(data_loader, print_freq, header):
        weakly_correlated_loss = 0.0
        for i in range(images.shape[0]):

            images_view1 = torch.unsqueeze(images[i][0].to(device),0)
            images_view2 = torch.unsqueeze(images[i][1].to(device),0)
            images_sobel = torch.unsqueeze(images[i][2],0).numpy()
            
            
            z_view1 = model.forward(images_view1) # NxD
            z_view1 = torch.squeeze(z_view1,0)
            
            z_view2 = model.forward(images_view2) # NxD
            z_view2 = torch.squeeze(z_view2,0)
            
            # normalize repr. along the batch dimension
            z_view1_norm = (z_view1 - z_view1.mean()) / z_view1.std() # NxD
            z_view2_norm = (z_view2 - z_view2.mean()) / z_view2.std() # NxD
            
            del z_view1, z_view2
            # Region of Intrest
            pos = np.array(np.where(images_sobel == np.max(images_sobel)))
            idx = int(np.round(pos.shape[1]/2))
            pos = [pos[2][idx],pos[3][idx]]
            
            sample_size = 64
            if pos[0] < sample_size/2:
                pos[0] = sample_size/2
            if pos[0] > 256 - sample_size/2:
                pos[0] = 256 - sample_size/2
                
            if pos[1] < sample_size/2:
                pos[1] = sample_size/2
            if pos[1] > 256 - sample_size/2:
                pos[1] = 256 - sample_size/2
            
            
            RoI_view1 = z_view1_norm[:,int(pos[0]-sample_size/2):int(pos[0]+sample_size/2),int(pos[1]-sample_size/2):int(pos[1]+sample_size/2) ]  
            RoI_view2 = z_view1_norm[:,int(pos[0]-sample_size/2):int(pos[0]+sample_size/2),int(pos[1]-sample_size/2):int(pos[1]+sample_size/2) ]  
            
            for j in range(z_view1_norm.shape[0]):
                z_view1_vec = RoI_view1[i,:,:].reshape((RoI_view1[i,:,:].shape[1]*RoI_view1[i,:,:].shape[0]),1)
                z_view2_vec = RoI_view2[i,:,:].reshape((RoI_view2[i,:,:].shape[1]*RoI_view2[i,:,:].shape[0]),1)
                cross_weakly = z_view1_vec @ z_view2_vec.mT # DxD
                cross_weakly = cross_weakly - torch.eye(sample_size*sample_size).cuda()
                weakly_correlated_loss += cross_weakly.sum()
             
                
        weakly_correlated_loss = ((weakly_correlated_loss/(sample_size*sample_size))/z_view1_norm.shape[0])/images.shape[0] # nomalizing the loss by dividing with batchsize, number of sampeled points, number of embeddings
        
        optimizer.zero_grad()
        weakly_correlated_loss.backward()
        optimizer.step()        
        
        cum_loss += weakly_correlated_loss.detach().cpu().numpy()
        num_batches += 1
        
        metric_logger.update(loss=weakly_correlated_loss)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    scheduler.step()
    return cum_loss/ num_batches
