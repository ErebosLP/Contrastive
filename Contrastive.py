import utils
import os
import torch

import random
import torchvision
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import torchvision.transforms as T
import segmentation_models_pytorch as smp
from PIL import Image, ImageOps, ImageFilter
from torch.utils.tensorboard import SummaryWriter


from engine import train_one_epoch
from City_imageloader import CityscapeDataset

def get_contrastive_model():
    #model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
    model = torchvision.models.resnet50(zero_init_residual=True)
    model = torch.nn.Sequential(*(list(model.children())[:-1]))

    return model

class GaussianBlur(object):
    def __init__(self, p,alpha):
        self.p = p
        self.alpha = alpha
        
    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * self.alpha + 0.1
            
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img
        
class Sobel:
    def __init__(self):
        pass
        
    def __call__(self,img):
        # Get x-gradient in "sx"

        sx = ndimage.sobel(img,axis=0,mode='constant')
        # Get y-gradient in "sy"
        sy = ndimage.sobel(img,axis=1,mode='constant')
        # Get square root of sum of squares
        sobel_out=np.hypot(sx,sy)
            
        return sobel_out

class RandomErasing:
    def __init__(self, p, area):
        self.p = p
        self.area = area

    def __call__(self, img):
        if np.random.random() < self.p:
            return img

        new_img = np.array(img)
        S_e = (np.random.random() * self.area + 0.1) * new_img.shape[0] * new_img.shape[1] # random area
        tot = 0
        while tot < S_e:
            y , x = np.random.randint(0, new_img.shape[0]-2) , np.random.randint(0, new_img.shape[1]-2)
            wy, wx = np.random.randint(1, new_img.shape[0] - y) , np.random.randint(1, new_img.shape[1] - x)

            if wy * wx > S_e*2:
                continue

            tot += wy * wx

            random_patch = np.random.rand(wy,wx,3)*255
            new_img[ y : y + wy , x : x + wx , : ] = random_patch

        return Image.fromarray(new_img)
    
class augmentation:
    def __init__(self):
            self.transform = T.Compose([
                T.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
                GaussianBlur(1,np.random.uniform(0.1,2)),
                T.RandomApply([Sobel()],p=1),
                T.ToTensor(),          
                ])
            

              
            self.transform_prime =  T.Compose([
                GaussianBlur(1,np.random.uniform(0.1,2)),
                T.ToTensor(),          
                ])
    
    
    def __call__(self, x, x_neg):
        y1 = self.transform(x)
        y1 = y1.to(torch.float32)
        y2 = self.transform_prime(x)
        y2 = y2.to(torch.float32)
        
        
        
        return torch.cat((y1.unsqueeze(0), y2.unsqueeze(0)),0)


def main():
    base_lr = 0.0001
    numEpochs = 5
    learningRate = base_lr

    # model name   
    model_name = 'model_Barlow_contrastive_numEpochs' + str(numEpochs)
    print('model name: ', model_name)
    
    # see if path exist otherwise make new directory
    out_dir = os.path.join('./results/Barlow/', model_name )
    print('out_dir: ', out_dir)
    
    if not os.path.exists(os.path.join(out_dir,'checkpoint')):
        os.makedirs(os.path.join(out_dir,'checkpoint'))
    initial_checkpoint = None
    
    ## ----------------------------------------------------------

    # writer = SummaryWriter(comment=comment_name)
    writer = SummaryWriter("./runs/" + model_name)
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    root = 'C:/Users/Jean-/sciebo/Documents/Masterprojekt/Dataset_test/' #'E:/Datasets/'
    #root = 'C:/Users/marie/sciebo/Master/Semester 2/Masterprojekt/Dataset_test/'
    dataset = CityscapeDataset(root,"train",augmentation())

    data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=True, num_workers=4)
    # import ipdb
    # ipdb.set_trace()
    model = smp.Unet(encoder_name='resnet50', encoder_weights=None, classes=16, activation='sigmoid')
    #model.load_state_dict(torch.load('C:/Users/Jean-/Desktop/Contrastive/max_valid_model.pth')['model_state_dict'])
    #model = get_contrastive_model()
    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=learningRate, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, numEpochs)
    
    start_epoch = 0
    ## ====================================================================================
    print('===========================================================================')
    print('start epoch: ', start_epoch)
# let's train it for X epochs
    num_epochs = numEpochs
    min_trainLoss = np.inf
    for epoch in range(start_epoch,num_epochs):
        
        # train for one epoch, printing every 10 iterations
        print('start train one epoch')
        losses_OE = train_one_epoch(model, optimizer, data_loader, device, epoch, 100, scheduler)
        writer.add_scalar('Loss_Barlow/train', losses_OE, epoch)

        # update the learning rate
        if epoch % 15 == 0:
            torch.save(model.state_dict(), out_dir + '/checkpoint/%08d_model.pth' % (epoch))
            torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'loss': loss
                }, out_dir + '/checkpoint/%08d_model.pth' % (epoch))

        print('losses_OE & min_trainLoss', losses_OE, '/', min_trainLoss)
        if min_trainLoss > losses_OE:
            min_trainLoss = losses_OE
            # torch.save(model.state_dict(), out_dir + '/checkpoint/max_valid_model.pth')
            torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'loss': loss
                }, out_dir + '/checkpoint/max_valid_model.pth')

    torch.save(model.state_dict(), './model/Barlow_model/'+ model_name + '.pth')
    






if __name__ == '__main__':
    main()