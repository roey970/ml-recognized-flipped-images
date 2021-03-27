
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import PIL
from PIL import Image,ImageOps
from PIL import ExifTags
import matplotlib.pyplot as plt
import numpy as np
import time
import torch.nn.functional as F
from torch.utils import data
import torch.optim as optim
import random
import  os
import torchvision.models as models

BATCH_SIZE=2
NUM_WORKERES=0
CLASSES_NUM=4
FACTOR=0.65
PATIENCE=2
TRAIN_DATA_PATH='data/train_squared_orginized_224/'
TEST_DATA_PATH='data/test_orginized_squared_224/'
VALIDATE_DATA_PATH='data/test_orginized_squared_224/'#'/content/drive/My Drive/software for fun/python3/neural networks/roey net/validate_orginized_squared_224/'
NETWORK_PATH='./ROEYNET_RES18_GPU.pth'

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        r=random.randint(0,len(self.datasets)-1)
        #return tuple(d[r] for d in self.datasets)[i]
        #print("datasets={0}".format(self.datasets))
        #print("datasers[0]={0}".format(self.datasets[0]))
        #print("datasets[r][i]={0}".format(self.datasets[r][i]))
        #for i in self.datasets:
        #    print(i[0].shape)
        return self.datasets[r][i]

    def __len__(self):
        return min(len(d) for d in self.datasets)


def test(testloader, net):
    correct = 0
    total = 0
    total_unflipped=0
    correct_unflipped=0
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            #images, labels = data[0], data[1]
            outputs = net(images)
            _, predicted = torch.max(outputs.data.to(device), 1)#
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i in range(len(predicted)):
                if(labels[i]==0):
                    total_unflipped+=1
                    if(predicted[i]==0):
                        correct_unflipped+=1

    print("acc unflipped={0}".format(100*correct_unflipped/total_unflipped))
    accuracy = 100 * correct / total
    print("total={0} correct={1}".format(total,correct))
    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
    return accuracy

def show_test_against_taining(epocs,train_data,test_data):
    plt.plot(epocs, train_data, label="train accuracy", color='r', marker='o', markerfacecolor='k', linestyle='--',
             linewidth=3)
    plt.plot(epocs, test_data, label="test accuracy", color='g', marker='o', markerfacecolor='k', linestyle='--',
             linewidth=3)
    plt.title('test against train')
    plt.legend()
    plt.show()

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def train(trainloader,validate_loader, optimizer, net,scheduler):
    criterion = nn.CrossEntropyLoss()
    last = 0
    running_loss = 0.0
    train_loader_length=len(trainloader)
    #the num i am deviding with is
    when_to_chek=train_loader_length/4 #*BATCH_SIZE)

    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        #inputs, labels = data[0],data[1]#
        #print("lables={0}".format(labels))
        #for im in inputs:
        #    imshow(im)


        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        #print("outputs shape ={0}".format(outputs.shape))
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 300 == 299:  # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 300))
            last = running_loss / 300
            running_loss = 0.0

        if i%when_to_chek ==0 and i!=0:
            val_loss = validate(validate_loader, optimizer, net)
            scheduler.step(val_loss)
    return last


def validate(data_loader, optimizer, net):
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    for i, data in enumerate(data_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)
        #inputs, labels = data[0],data[1]#

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        # optimizer.step()

        # print statistics
        running_loss += loss.item()
        #if i % 2000 == 1999:  # print every 2000 mini-batches
        #    print('[%d, %5d] loss: %.3f' %
        ##          (epoch + 1, i + 1, running_loss / 2000))
        #    last = running_loss / 2000
        #    return last
    return running_loss/len(data_loader)

#get length of every dataset in list
#return list of indexes to go through
def make_square_fun(img):
    width, height = img.size
    #print(width,height)
    newsize = (112, 112)
    #img=transform_rotation270_fun(img)
    #img.show()
    if width>height:

        new_witdh=width-height
                                #left , top , right , button
        #img_cropped = img.crop((new_witdh/2,0,new_witdh/2,0))
        img_cropped = img.crop((new_witdh/2, 0,width-new_witdh/2, height))
        #img_cropped.show()
        out = img_cropped.resize(newsize)
        return out
    elif height>width:

        #img=transform_rotation270_fun(img)
        #img=img.resize((32,32))
        #img.show()
        new_height = height - width
        img_cropped = img.crop((0, new_height/2, width ,height-new_height/2))
        out = img_cropped.resize(newsize)
        return out
    else:
        return img.resize(newsize)


def resize_fun(img):
    newsize = (224, 224)
    out = img.resize(newsize)
    return out
def transform_rotation90_fun(img):
    out = img.transpose(PIL.Image.ROTATE_90)
    #torchvision.transforms.functional.affine(img, 90)
    return out

def transform_rotation180_fun(img):
    #torchvision.transforms.functional.affine(img, 180)
    out = img.transpose(PIL.Image.ROTATE_180)
    return out
def transform_rotation270_fun(img):
    out = img.transpose(PIL.Image.ROTATE_270)
    return out
def transform_mirror_fun(img):
    out=ImageOps.mirror(img)
    return out
def set_target_to_0_fun(target):
    return 0
def set_target_to_1_fun(target):
    return 1
def set_target_to_2_fun(target):
    return 2
def set_target_to_3_fun(target):
    return 3

# make_square also resize
#make_square = transforms.Lambda(make_square_fun)
resize = transforms.Lambda(resize_fun)
rotation90 = transforms.Lambda(transform_rotation90_fun)
rotation180 = transforms.Lambda(transform_rotation180_fun)
rotation270 = transforms.Lambda(transform_rotation270_fun)
mirror = transforms.Lambda(transform_mirror_fun)
set_target_to_0=transforms.Lambda(set_target_to_0_fun)
set_target_to_1=transforms.Lambda(set_target_to_1_fun)
set_target_to_2=transforms.Lambda(set_target_to_2_fun)
set_target_to_3=transforms.Lambda(set_target_to_3_fun)


transform_mom = transforms.Compose(
    [
        #make_square, #resize
        transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

transform_miror= transforms.Compose(
    [
        #make_square, #resize
        mirror,
        transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])


transform_rotation90_mom= transforms.Compose(
    [#make_square,
     rotation90,
    transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

transform_miror_rotation90= transforms.Compose(
    [
        #make_square, #resize
        mirror,
        rotation90,
        transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])


transform_rotation180_mom= transforms.Compose(
    [#make_square,
     rotation180,
    transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

transform_miror_rotation180= transforms.Compose(
    [
        #make_square, #resize
        mirror,
        rotation180,
        transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

transform_rotation270_mom= transforms.Compose(
    [#make_square,
     rotation270,
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

transform_miror_rotation270= transforms.Compose(
    [
        #make_square, #resize
        mirror,
        rotation270,
        transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])



os.chdir('C:/Users/User/Desktop/roey stuff/python3/nural networkds/')

data_path=TRAIN_DATA_PATH

##regular trainset
my_train_set0 = torchvision.datasets.ImageFolder(
     root=data_path,
     transform=transform_mom,target_transform=set_target_to_0)


my_train_set90 = torchvision.datasets.ImageFolder(
    root=data_path,
     transform=transform_rotation90_mom,target_transform=set_target_to_1)


my_train_set180 = torchvision.datasets.ImageFolder(
    root=data_path,
     transform=transform_rotation180_mom,target_transform=set_target_to_2)

my_train_set270 = torchvision.datasets.ImageFolder(
    root=data_path,
     transform=transform_rotation270_mom,target_transform=set_target_to_3)
#####
#mirror set
my_train_set_mirror0 = torchvision.datasets.ImageFolder(
     root=data_path,
     transform=transform_miror,target_transform=set_target_to_0)


my_train_set_mirror90 = torchvision.datasets.ImageFolder(
    root=data_path,
     transform=transform_miror_rotation90,target_transform=set_target_to_1)


my_train_set_mirror180 = torchvision.datasets.ImageFolder(
    root=data_path,
     transform=transform_miror_rotation180,target_transform=set_target_to_2)

my_train_set_mirror270 = torchvision.datasets.ImageFolder(
    root=data_path,
     transform=transform_miror_rotation270,target_transform=set_target_to_3)


train_loader = torch.utils.data.DataLoader(
             ConcatDataset( #all must be similar
                 my_train_set0,
                 my_train_set90,
                 my_train_set180,
                 my_train_set270,
                 my_train_set_mirror0,
                 my_train_set_mirror90,
                 my_train_set_mirror180,
                 my_train_set_mirror270
             ),
             batch_size=BATCH_SIZE, shuffle=True,
             num_workers=NUM_WORKERES, pin_memory=True)
#################################
#regular tests set
data_path=TEST_DATA_PATH
my_test_set0 = torchvision.datasets.ImageFolder(
     root=data_path,
     transform=transform_mom,target_transform=set_target_to_0)

my_test_set90 = torchvision.datasets.ImageFolder(
     root=data_path,
     transform=transform_rotation90_mom,target_transform=set_target_to_1)


my_test_set180 = torchvision.datasets.ImageFolder(
     root=data_path,
     transform=transform_rotation180_mom,target_transform=set_target_to_2)


my_test_set270 = torchvision.datasets.ImageFolder(
     root=data_path,
     transform=transform_rotation270_mom,target_transform=set_target_to_3)
#########
my_test_set_mirrir0 = torchvision.datasets.ImageFolder(
     root=data_path,
     transform=transform_miror,target_transform=set_target_to_0)

my_test_set_mirror90 = torchvision.datasets.ImageFolder(
     root=data_path,
     transform=transform_miror_rotation90,target_transform=set_target_to_1)


my_test_set_mirror180 = torchvision.datasets.ImageFolder(
     root=data_path,
     transform=transform_miror_rotation180,target_transform=set_target_to_2)


my_test_set_mirror270 = torchvision.datasets.ImageFolder(
     root=data_path,
     transform=transform_miror_rotation270,target_transform=set_target_to_3)

test_loader = torch.utils.data.DataLoader(
             ConcatDataset(
                 my_test_set0,
                 my_test_set90,
                 my_test_set180,
                 my_test_set270,
                 my_test_set_mirrir0,
                 my_test_set_mirror90,
                 my_test_set_mirror180,
                 my_test_set_mirror270



             ),
             batch_size=BATCH_SIZE, shuffle=True,
             num_workers=NUM_WORKERES, pin_memory=True)

############################

data_path=VALIDATE_DATA_PATH
my_validate_set0 = torchvision.datasets.ImageFolder(
     root=data_path,
     transform=transform_mom,target_transform=set_target_to_0)

my_validate_set90 = torchvision.datasets.ImageFolder(
     root=data_path,
     transform=transform_rotation90_mom,target_transform=set_target_to_1)


my_validate_set180 = torchvision.datasets.ImageFolder(
     root=data_path,
     transform=transform_rotation180_mom,target_transform=set_target_to_2)


my_validate_set270 = torchvision.datasets.ImageFolder(
     root=data_path,
     transform=transform_rotation270_mom,target_transform=set_target_to_3)

###########
my_validate_set_mirror0 = torchvision.datasets.ImageFolder(
     root=data_path,
     transform=transform_miror,target_transform=set_target_to_0)

my_validate_set_mirror90 = torchvision.datasets.ImageFolder(
     root=data_path,
     transform=transform_miror_rotation90,target_transform=set_target_to_1)


my_validate_set_mirror180 = torchvision.datasets.ImageFolder(
     root=data_path,
     transform=transform_miror_rotation180,target_transform=set_target_to_2)


my_validate_set_mirror270 = torchvision.datasets.ImageFolder(
     root=data_path,
     transform=transform_miror_rotation270,target_transform=set_target_to_3)


validate_loader = torch.utils.data.DataLoader(
             ConcatDataset(
                 my_validate_set0,
                 my_validate_set90,
                 my_validate_set180,
                 my_validate_set270,
                 my_validate_set_mirror0,
                 my_validate_set_mirror90,
                 my_validate_set_mirror180,
                 my_validate_set_mirror270


             ),
             batch_size=BATCH_SIZE, shuffle=True,
             num_workers=NUM_WORKERES, pin_memory=True)

#######################
classes = ('unflipped','90 flipped','180 flipped','270 flipped')

#net = Net()
#net=RoeyNet()
resnet18 = models.resnet18(pretrained=True)
net=resnet18
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net.to(device)
#net = torch.nn.parallel.DataParallel(net)
lr= 0.0005541146718671695
reg=0.00040379987997609583
optimizer = optim.Adam(net.parameters(), lr=lr,weight_decay=reg)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience = PATIENCE,verbose=True,factor=FACTOR)
epochs=[]
test_accuracy=[]
train_accuracy=[]

for epoch in range(50):  # loop over the dataset multiple times
    current_loss = train(train_loader,validate_loader,optimizer,net,scheduler)
    print(1)
    #val_loss = validate(validate_loader, optimizer, net)
    #scheduler.step(val_loss)
    current_test_accuracy = test(test_loader, net)
    current_train_accuracy = test(train_loader,net)
    test_accuracy.append(current_test_accuracy)
    train_accuracy.append(current_train_accuracy)
    epochs.append(epoch)
    if epoch % 5 == 0:
        show_test_against_taining(epochs, train_accuracy, test_accuracy)


show_test_against_taining(epochs,train_accuracy,test_accuracy)
torch.save(net, NETWORK_PATH)