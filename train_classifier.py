import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import time
import os
import copy
import argparse
import albumentations as A
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm

def train_model(model, criterion, optimizer, scheduler, num_epochs,dataloaders,device,dataset_sizes):

    since = time.time()
    best_acc = 0.0
    start_epoch = 0
    
    if opt.resume != 'False':
        print('> loading resumed weights from',opt.resume)
        checkpoint = torch.load(opt.resume)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = 1e-4
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


def normalize():
    def f_normalize(img):
        img = np.array(img, dtype=np.float32)
        return img/255.0
    return lambda img: f_normalize(img)


def albumen_augmentations():
    transforms = A.Compose([
                        A.CLAHE(p=0.5),
                        A.GaussNoise(p=0.5),
                        A.GaussianBlur(blur_limit=(3,7), p=0.4),
                        A.MotionBlur(blur_limit=(3,7), p=0.4),
                        A.RandomBrightnessContrast(brightness_limit=0.6, contrast_limit=0.6, p=0.7),
                        A.RandomFog(p=0.3),
                        A.RGBShift(p=0.3), 
                        A.JpegCompression(quality_lower=50, p=0.5)
                      ])
    return  lambda  img:transforms(image=np.array(img))['image']


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
        model = EfficientNet.from_pretrained('efficientnet-b3')
    else:
        raise Exception("! Sorry, CNN_TYPE not recognized !")
        
    # change classifier
    if type == 'efficientnet':
        num_ftrs = model._fc.in_features
        model._fc = nn.Linear(num_ftrs, num_classes)
    else:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    return model
        
        
def train(opt):

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(-50, 50)),
            albumen_augmentations(),
            normalize(),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            normalize(),
            transforms.ToTensor(),
        ]),
    }
    
    data_dir = opt.data_dir
    print('> loading data from',data_dir)
    image_datasets = {x: ImageFolder(os.path.join(data_dir, x),data_transforms[x]) for x in ['train','val']}
    traffics = {'truck':0, 'pickup':1, 'car':2, 'suv':3, 'three wheelers (CNG)':4, 'bus':5, 'van':6, 'ambulance':7, 'rickshaw':8, 'minivan':9, 'motorbike':10, 'bicycle':11, 'army vehicle':12, 'human hauler':13, 'taxi':14, 'wheelbarrow':15, 'auto rickshaw':16, 'minibus':17, 'scooter':18, 'policecar':19, 'garbagevan':20}
    image_datasets['train'].class_to_idx = traffics
    image_datasets['val'].class_to_idx = traffics

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print('> num images in train and val folder:',dataset_sizes)
    class_names = image_datasets['train'].classes
    print('> classes :', class_names)

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=opt.batch_size,shuffle=True, num_workers=4) for x in ['train', 'val']}    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('> training on device:',device)

    # load model
    model_ft = None
    print('> loading {} model, number of classes is {}....'.format(opt.CNN_type,len(class_names)))
    model_ft = load_model(type = opt.CNN_type, num_classes = len(class_names))

    # compile model
    print('> compiling...')
    for param in model_ft.parameters():
        param.requires_grad = True
    
    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001, amsgrad=True)    # optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.6)
    
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


