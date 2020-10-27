import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import torchvision
from torchvision import models, transforms
from utils.classifier_datasets import MyImageFolder
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import time
import os
import copy
import argparse
import albumentations as A
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm
from PIL import Image

def train_model(model, criterion, optimizer, scheduler, num_epochs,dataloaders,device,dataset_sizes):

    since = time.time()
    best_acc = 0.0
    start_epoch = 0

    if opt.resume != 'False':
        print('> loading resumed weights from',opt.resume)
        checkpoint = torch.load(opt.resume)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = 1e-4
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])

        
    print('> starting training...')
    for epoch in range(start_epoch,num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            if phase == 'train' and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }

            if phase=='train':
                torch.save(checkpoint,os.path.join(checkpoint_save_dir,'last.pt'))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(checkpoint, os.path.join(checkpoint_save_dir,'best.pt'))

                if opt.SaveBestInDrive !='NOT_SET':
                   torch.save(checkpoint, opt.SaveBestInDrive)
                   
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    return


def albumen_augmentations():

    transforms = A.Compose([
                        A.CLAHE(p=0.4),
                        A.GaussNoise(p=0.5),
                        A.GaussianBlur(blur_limit=(3,7), p=0.4),
                        A.MotionBlur(blur_limit=(3,7), p=0.4),
                        A.RandomBrightnessContrast(brightness_limit=0.6, contrast_limit=0.6, p=0.7),
                        A.RandomFog(p=0.3),
                        A.RGBShift(p=0.3), 
                        A.JpegCompression(quality_lower=50, p=0.4)
                      ])
    return  lambda  img:transforms(image=np.array(img))['image']


def get_class_weights(dataset_obj,print_stat=False):

    count_dict = {i:0 for c,i in dataset_obj.class_to_idx.items()}
    for _,y in dataset_obj:
        count_dict[y] += 1
    class_count = [i for i in count_dict.values()]
    class_weights = 1./torch.tensor(class_count, dtype=torch.float) 
    if print_stat:
        print("count_dict:",count_dict)
        print("class_count:",class_count)
        print("class_weights:",class_weights)
    return class_weights


def load_model(type='efficientnet', num_classes = 21):
 
    if type=='resnet34':
        model = models.resnet34(pretraind = True)
    elif type=='resnet18':
        model = models.resnet18(pretraind = True)
    elif type=='resnet50':
        model = models.resnet50(pretrain = True)       
    elif type=='resnet100':
        model = models.resnet100(pretraind = True)
    elif type=='efficientnet':
        model = EfficientNet.from_pretrained('efficientnet-b0')
    else:
        raise Exception("! Sorry, CNN_TYPE not recognized !")
        
    # change classifier
    if type == 'efficientnet':
        num_ftrs = model._fc.in_features
        model._fc = nn.Sequential(nn.Linear(num_ftrs, 128), nn.LeakyReLU(0.1), nn.Dropout(p=0.2), nn.Linear(128, num_classes))
        #model._fc = nn.Linear(num_ftrs, num_classes)
    else:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)

    return model
        
        
def train(opt):

    # Data augmentation and normalization 

    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomAffine(degrees=(-30, 30),translate=(0.25, 0.25),scale=(0.8, 1.5),resample=Image.BILINEAR),
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(p=0.5),
            albumen_augmentations(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    
    data_dir = opt.data_dir
    print('> loading data from',data_dir)
    image_datasets = {x: MyImageFolder(os.path.join(data_dir, x),data_transforms[x]) for x in ['train','val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print('> num images in train and val folder:',dataset_sizes)
    class_names = image_datasets['train'].classes
    print('> classes :', class_names)
    print('> class_to_idx :',image_datasets['train'].class_to_idx)

    class_weights = get_class_weights(image_datasets['train'], print_stat=True)
    target_list = torch.tensor(image_datasets['train'].targets)
    class_weights_all = class_weights[target_list]
    weighted_sampler = WeightedRandomSampler(weights=class_weights_all,num_samples=len(class_weights_all),replacement=True)

    train_loader = DataLoader(dataset=image_datasets['train'], shuffle=True, batch_size=opt.batch_size, sampler=weighted_sampler, num_worker = 4)
    val_loader = DataLoader(dataset=image_datasets['val'],shuffle=True, batch_size=opt.batch_size, num_worker=4)

    dataloaders =  {'train':train_loader, 'val':val_loader}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('> training on device:',device)

    # load model
    model_ft = None
    print('> loading {} model, number of classes is {}...'.format(opt.CNN_type,len(class_names)))
    model_ft = load_model(type = opt.CNN_type, num_classes = len(class_names))

    # compile model
    print('> compiling...')
    for param in model_ft.parameters():
        param.requires_grad = True
    
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001, amsgrad=True)    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.8)
    # exp_lr_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer_ft, T_0 = 10, T_mult = 2, eta_min = 1e-5)
    
    # train model
    train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,opt.epochs,dataloaders,device,dataset_sizes)

    
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='data directoris')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--resume', type=str, default='False', help='Resumed weight paths')
    parser.add_argument('--SaveBestInDrive', type=str, default='NOT_SET', help='saves the best model in given google drive path')
    parser.add_argument('--CNN_type', type=str, help='cnn type to be trained')
    opt = parser.parse_args()
    
    print('> opt:', opt)
    
    checkpoint_save_dir = 'classifier_checkpoints'
    if not os.path.exists(checkpoint_save_dir):
        os.mkdir(checkpoint_save_dir)
    print('> last and best model will be saved inside',checkpoint_save_dir)

    # Train
    train(opt)


