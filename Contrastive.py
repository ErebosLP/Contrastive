import os
import torch
import utils
import torchvision
import numpy as np
from engine import train_one_epoch
from City_imageloader import CityscapeDataset

from torch.utils.tensorboard import SummaryWriter

def get_contrastive_model():
    # load an instance segmentation model pre-trained pre-trained on COCO
    #model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
    model = torchvision.models.resnet50(zero_init_residual=True)
    return model

def augmentation():
    
    pass



def main():
    base_lr = 0.000001
    numEpochs = 1
    learningRate = base_lr

    # model name   
    model_name = 'model_Cityscapes_contrastive_numEpochs' + str(numEpochs) 
    print('model name: ', model_name)
    
    # see if path exist otherwise make new directory
    out_dir = os.path.join('./results/Cityscapes/', model_name )
    print('out_dir: ', out_dir)
    
    if not os.path.exists(os.path.join(out_dir,'checkpoint')):
        os.makedirs(os.path.join(out_dir,'checkpoint'))
    initial_checkpoint = None
    
    ## ----------------------------------------------------------

    # writer = SummaryWriter(comment=comment_name)
    writer = SummaryWriter("./runs/" + model_name)
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')    
    root = 'E:/Datasets/'
    dataset = CityscapeDataset(root,"train")
    data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=True, num_workers=4,collate_fn=utils.collate_fn)
    # import ipdb
    # ipdb.set_trace()
    
    model = get_contrastive_model()
    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=learningRate, weight_decay=0.0005)
    
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
        losses_OE = train_one_epoch(model, optimizer, data_loader, device, epoch, 10, learningRate)
        writer.add_scalar('Loss_Cityscapes/train', losses_OE, epoch)

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
























if __name__ == '__main__':
    main()