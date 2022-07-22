
import torch
import random
from PIL import Image, ImageOps, ImageFilter
import utils
import numpy as np
import torchvision
import torchvision.transforms as T
import ipdb



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
            #prepare the images views
            images_view1 = torch.unsqueeze(images[i][0].to(device),0)
            images_view2 = torch.unsqueeze(images[i][1].to(device),0)
            images_sobel = torch.unsqueeze(images[i][2],0).numpy()
            
            # compute the embeddings
            z_view1 = model.forward(images_view1) # NxD
            z_view1 = torch.squeeze(z_view1,0)
            
            z_view2 = model.forward(images_view2) # NxD
            z_view2 = torch.squeeze(z_view2,0)
            
            # normalize embeddings along the batch dimension
            z_view1_norm = (z_view1 - z_view1.mean()) / z_view1.std() # NxD
            z_view2_norm = (z_view2 - z_view2.mean()) / z_view2.std() # NxD
            
            #delete non normalized embeddings for memmory reasons (on my laptop)
            del z_view1, z_view2

            # Compute the Region of Intrest
            sample_size = 64
            pos = [np.random.normal(256/2,50-sample_size/2,1)[0],np.random.normal(256/2,200-sample_size/2,1)[0]]            
            pos = np.round(pos).astype('uint8')
            
            # compute if valid RoI and correct it if not
            if pos[0] < sample_size/2:
                pos[0] = sample_size/2
            if pos[0] > 256 - sample_size/2:
                pos[0] = 256 - sample_size/2
                
            if pos[1] < sample_size/2:
                pos[1] = sample_size/2
            if pos[1] > 256 - sample_size/2:
                pos[1] = 256 - sample_size/2
            
            # Cutt out the RoI of the embeddings
            RoI_view1 = z_view1_norm[:,int(pos[0]-sample_size/2):int(pos[0]+sample_size/2),int(pos[1]-sample_size/2):int(pos[1]+sample_size/2) ]  
            RoI_view2 = z_view2_norm[:,int(pos[0]-sample_size/2):int(pos[0]+sample_size/2),int(pos[1]-sample_size/2):int(pos[1]+sample_size/2) ]  
            
            # #----------------------------------------------------------------------
            # test = images_view2.squeeze(0)[:,int(pos[0]-sample_size/2):int(pos[0]+sample_size/2),int(pos[1]-sample_size/2):int(pos[1]+sample_size/2) ]
            # test = test.cpu().numpy().transpose(1,2,0) * 255
            # import matplotlib.pyplot as plt
            # plt.imshow(test.astype('uint8'))
            # plt.show()
            # #---------------------------------------------------------------------- 

            # #Compute loss
            
            # # vectorize the embeddings
            # RoI_view1_vec = torch.reshape(torch.ravel(RoI_view1),(sample_size*sample_size,16))
            # RoI_view2_vec = torch.reshape(torch.ravel(RoI_view2),(sample_size*sample_size,16))
            
            # # Compute Corsscorrelation
            # cross_weakly = RoI_view1_vec @ RoI_view2_vec.mT # DxD    
            # norm_view1 = torch.linalg.norm(RoI_view1_vec,axis = 1) 
            # norm_view2 = torch.linalg.norm(RoI_view2_vec,axis = 1)
            # norm = torch.unsqueeze(norm_view1,1)@torch.unsqueeze(norm_view2,1).T
            
            # cross_weakly_norm = torch.div(cross_weakly,norm)
            
            
            #Compute Loss
            RoI_view1_norm = (RoI_view1 - RoI_view1.mean(0).unsqueeze(0))/RoI_view1.std(0).unsqueeze(0)
            RoI_view2_norm = (RoI_view2 - RoI_view2.mean(0).unsqueeze(0) )/RoI_view2.std(0).unsqueeze(0)
            
            sim = torch.nn.CosineSimilarity(dim=0, eps=1e-08)(RoI_view1_norm,RoI_view2_norm)
            sim = (sim + 1) / 2
            
            sim_diff = torch.abs(sim - 1)
             
            # # Substract the identity matrix and take the absolute value
            #cross_diff = torch.abs(cross_weakly_norm - torch.eye(sample_size*sample_size).cuda())
            # Sum up the loss over a batch
            weakly_correlated_loss += sim_diff.sum()
            
                
             
        # nomalizing the loss by dividing with batchsize, number of sampeled points, number of embeddings
        # weakly_correlated_loss = ((weakly_correlated_loss/(sample_size*sample_size))/z_view1_norm.shape[0])/images.shape[0] 
        
        # Optimizer
        optimizer.zero_grad()
        weakly_correlated_loss.backward()
        optimizer.step()        
        
        # cumultative loss over one training epoch (later devided by the number of batches)
        cum_loss += weakly_correlated_loss.detach().cpu().numpy()*10000
        num_batches += 1
        
        # log the progress
        metric_logger.update(loss=weakly_correlated_loss*10000)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # learnigreate update
    scheduler.step()
    return cum_loss/ num_batches
