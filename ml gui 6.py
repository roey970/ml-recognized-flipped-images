#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
author: roey bretschneider
"""
import threading
import wx
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import PIL
from PIL import Image, ImageOps
from PIL import ExifTags
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils import data
import torch
import torchvision.models as models
import os

# defines paths of images and net
TEST_DATA_PATH = 'C:/Users/User/Desktop/roey stuff/python3/nural networkds/data/test_orginized_squared_224/'
NETWORK_PATH = 'C:/Users/User/Desktop/roey stuff/python3/nural networkds/roeynet_res18.pth'  # ROEYNET_8_RES18_EVAL
NETWORK_PATH = 'C:/Users/User/Desktop/roey stuff/python3/nural networkds/ROEYNET_RES101_GPU_batch32_14000_4.pth'  # ROEYNET_RES101_GPU_batch32_14000_4       ROEYNET_RES18_GPU_140000_3
PATH = "empty"
BATCH_SIZE = 32 # define number of photos to go through each time
NUM_WORKERES = 0
BIAS = 4  # 6 define bias, bias is caused because most of the time the pictures are in the correct organization
# so the bet will decide to flip image only if it is sure


def not_rotated_image(image):
    """
    the image might be flipped in its raw form but with a flag indicating its true form
    the function check the existence of this flag and return the image in it's right form
    in respect of the flags
    :param image:
    :return:
    """
    # print(image)
    # im = Image.open(image)
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation': break
    except:
        return image
    try:
        exif = dict(image._getexif().items())

        if exif[orientation] == 3:
            image = image.rotate(180, expand=True)
            print(1)
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
            print(2)
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)
            print(3)
        return image
    except:
        # print("exept")
        return image


def make_square_fun(img):
    """
    get image and return it in square shape (224,224)
    it firstly make the image to a square in a way that wont hurt the proportions of the image
    doing so by corping part of the image
    :param img: get the image
    :return:
    """
    width, height = img.size
    newsize = (224, 224)
    if (width < 224 or height < 224):
        return img.resize(newsize).convert('RGB')
    if (width == 224 and height == 224):
        # print(1)
        return img.convert('RGB')
    # print(width,height)

    # img=transform_rotation270_fun(img)
    # img.show()
    if width > height:

        new_witdh = width - height
        # left , top , right , button
        # img_cropped = img.crop((new_witdh/2,0,new_witdh/2,0))
        img_cropped = img.crop((new_witdh / 2, 0, width - new_witdh / 2, height))
        # img_cropped.show()
        out = img_cropped.resize(newsize)
        return out.convert('RGB')
    elif height > width:

        # img=transform_rotation270_fun(img)
        # img=img.resize((32,32))
        # img.show()
        new_height = height - width
        img_cropped = img.crop((0, new_height / 2, width, height - new_height / 2))
        out = img_cropped.resize(newsize)
        return out.convert('RGB')
    else:
        return img.resize(newsize).convert('RGB')


class RoeyDataset(datasets.folder.ImageFolder):
    """
    create class of the data set
    """
    def __init__(self, root, transform=None, target_transform=None,
                 loader=datasets.folder.default_loader):
        super(RoeyDataset, self).__init__(root, transform, target_transform, loader)

    def __getitem__(self, index):
        path, target = self.samples[index]
        # sample = self.loader(path)
        sample = Image.open(path)
        # sample2=self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, path


########################################################################################################################
class GuiMl(wx.Frame):

    def __init__(self, parent, title):
        """
        initalize net related parameters, path of images parameter,is scaning parameters
        and save changes parameter
        :param parent:
        :param title: title of the gui
        """
        super(GuiMl, self).__init__(parent, title=title)
        self.net = torch.load(NETWORK_PATH, map_location=torch.device('cpu'))
        self.net.eval()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.path = "empty"
        self.good_pic_num = 0
        self.need_to_flip = 0
        # true if scanning else false
        self.flag = False
        self.save_changes = False

        self.InitUI()  # create things like buttons and stuff (everything inside the up in frontend)
        self.Centre()  # put app in center

    def InitUI(self):
        """
        gui build
        :return:
        """

        panel = wx.Panel(self)

        font = wx.SystemSettings.GetFont(wx.SYS_SYSTEM_FONT)

        font.SetPointSize(9)
        #######
        # create current dir box
        vbox = wx.BoxSizer(wx.VERTICAL)
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        self.st1 = wx.StaticText(panel, label='current dir')
        self.st1.SetFont(font)
        hbox1.Add(self.st1, flag=wx.RIGHT, border=8)
        self.tc = wx.TextCtrl(panel)
        self.tc.SetEditable(False)
        hbox1.Add(self.tc, proportion=1)
        vbox.Add(hbox1, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, border=10)
        vbox.Add((-1, 10))
        ######
        # create statistics box
        hbox2_point5 = wx.BoxSizer(wx.HORIZONTAL)
        self.st3 = wx.StaticText(panel, label='statistics')
        self.st3.SetFont(font)
        hbox2_point5.Add(self.st3, flag=wx.RIGHT, border=8)
        self.tc_statistics = wx.TextCtrl(panel)
        self.tc_statistics.SetEditable(False)
        hbox2_point5.Add(self.tc_statistics, proportion=1)
        vbox.Add(hbox2_point5, flag=wx.EXPAND | wx.LEFT | wx.TOP | wx.RIGHT, border=10)

        vbox.Add((-1, 10))
        ######
        # create logger box
        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        st2 = wx.StaticText(panel, label='logger:')
        st2.SetFont(font)
        hbox2.Add(st2)
        vbox.Add(hbox2, flag=wx.LEFT | wx.TOP, border=10)

        vbox.Add((-1, 10))
        ######
        # create the box that contains the name of the flipped imagess
        hbox3 = wx.BoxSizer(wx.HORIZONTAL)
        self.tc2 = wx.TextCtrl(panel, style=wx.TE_MULTILINE)
        self.tc2.SetEditable(False)
        hbox3.Add(self.tc2, proportion=1, flag=wx.EXPAND)
        vbox.Add(hbox3, proportion=1, flag=wx.LEFT | wx.RIGHT | wx.EXPAND,
                 border=10)

        vbox.Add((-1, 25))
        #####
        # contains the scan button,browse button and clear logger button
        hbox5 = wx.BoxSizer(wx.HORIZONTAL)
        btn1 = wx.Button(panel, label='scan', size=(70, 30))
        btn1.Bind(wx.EVT_BUTTON, self.scan)
        hbox5.Add(btn1)
        btn2 = wx.Button(panel, label='browse', size=(70, 30))
        btn2.Bind(wx.EVT_BUTTON, self.browse)
        self.dlg = wx.DirDialog(panel, "check", "c:/",
                                wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST, size=(70, 30))
        btn3 = wx.Button(panel, label='clear logger', size=(70, 30))
        btn3.Bind(wx.EVT_BUTTON, self.clear_logger)
        # hbox5.Add(self.dlg, flag=wx.LEFT | wx.BOTTOM, border=5)
        hbox5.Add(btn2, flag=wx.LEFT | wx.BOTTOM, border=5)
        hbox5.Add(btn3, flag=wx.LEFT | wx.BOTTOM, border=5)
        vbox.Add(hbox5, flag=wx.ALIGN_RIGHT | wx.RIGHT, border=10)

        # initialize timer
        self.timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.update, self.timer)
        self.timer.Start(1000)

        panel.SetSizer(vbox)

    def scan(self, event):
        """
        scan button event function, activated when scan is pressed
        :param event:
        :return:
        """
        if self.path != "empty" and self.flag == False:
            self.flag = True
            print(self.path)
            self.define_net_stuff()
            dataset = RoeyDataset(self.path, transform=self.transform)
            data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                                                      num_workers=NUM_WORKERES, pin_memory=True)
            thread = threading.Thread(target=self.test, args=(data_loader, self.net))
            thread.start()
            self.flag = False
            # test(data_loader,net)

        # print(self.dlg.GetPath())

    def browse(self, event):
        """
        browse path in the computer
        :param event:
        :return:
        """
        self.dlg.ShowModal()
        self.tc.SetLabelText(self.dlg.GetPath())
        # change path  to be readable
        self.path = self.dlg.GetPath().replace('\g'[0], '/') + "/"
        print(self.path)

    def clear_logger(self):
        self.tc2.Clear()

    def update(self, event):
        """
        update the statistics log
        :param event:
        :return:
        """
        massage = "good pics ={0} , need to flip={1}".format(self.good_pic_num, self.need_to_flip)
        self.tc_statistics.Clear()
        self.tc_statistics.WriteText(massage)

    ############################################################
    def orginize_fun(self, img):
        """
        get image and flip it according to flag it contains and get into a small square
        :param img:
        :return:
        """
        # img.show()
        img = not_rotated_image(img)
        img = make_square_fun(img)
        # img.show()
        return img

    def define_net_stuff(self):
        """
        define the transform of the image before it goes to the net
        :return:
        """
        orginize_pic = transforms.Lambda(self.orginize_fun)

        self.transform = transforms.Compose(
            [
                orginize_pic,
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    def test(self, testloader, net):
        """

        :param testloader:
        :param net:
        :return:
        """
        max_0_to_1 = 0
        max_0_to_2 = 0
        max_0_to_3 = 0
        counter = 0 # counts number of pictures that are right
        # num of pictures that the difference between most recognized and second most is not bigger then 2
        # between pictures that are not flipped and other variations of floppiness
        counter_clearly_recegnize = 0
        # num of pictures that the difference between most recognized and second most is bigger then 2
        # between pictures that are not flipped and other variations of floppiness
        not_counter_clearly_recegnize = 0
        sum = 0
        with torch.no_grad():
            for data in testloader:
                counter2 = 0
                images, labels, paths = data[0], data[1], data[2]
                # add bias to output of the net
                outputs = net(images)
                for out in outputs:
                    out[0] += BIAS
                print(BIAS)

                # predicted- contains the classification based on the outputs
                _, predicted = torch.max(outputs.data.to(self.device), 1)
                for i in outputs:
                    # check if it got clearly recognized
                    if i[0] > i[1] and i[0] > i[2] and i[0] > i[3]:
                        # print(i[0:4])
                        counter += 1
                        sum += i[0] - max(i[1], i[2], i[3])
                        if (2 > i[0] - max(i[1], i[2], i[3])):
                            not_counter_clearly_recegnize += 1
                        else:
                            counter_clearly_recegnize += 1
                    # used for investigating stuff
                    zero_to_1 = i[1] - i[0]
                    if zero_to_1 > max_0_to_1:
                        max_0_to_1 = zero_to_1

                    zero_to_2 = i[2] - i[0]
                    if zero_to_2 > max_0_to_2:
                        max_0_to_2 = zero_to_2

                    zero_to_3 = i[3] - i[0]
                    if zero_to_3 > max_0_to_3:
                        max_0_to_3 = zero_to_3

                # go through images and present the one it changes
                # and change it by the flag
                for i in range(len(predicted)):
                    filename, file_extension = os.path.splitext(paths[i])
                    if file_extension != '.jpeg' and file_extension != '.jpg' and file_extension != '.JPG' and file_extension != '.JPEG':
                        pass
                    elif (0 != predicted[i]):
                        print(outputs[i][0:4])
                        img = Image.open(paths[i])
                        img = not_rotated_image(img)

                        if (predicted[i] == 1):
                            img = img.rotate(270, expand=True)
                            if self.save_changes:
                                img.save(paths[i])
                        if predicted[i] == 2:
                            img = img.rotate(180, expand=True)
                            if self.save_changes:
                                img.save(paths[i])
                        if predicted[i] == 3:
                            img = img.rotate(90, expand=True)
                            if self.save_changes:
                                img.save(paths[i])
                        img.show()

                        massage = "{0}".format(paths[i])
                        print(massage)
                        self.tc2.WriteText(massage + '\n')
                        self.need_to_flip += 1
                    else:
                        self.good_pic_num += 1
                        # self.imshow(images[i])
        self.tc2.WriteText('end' + '\n')

    def imshow(self, img):
        """
        show image
        :param img:
        :return:
        """
        img = img / 2 + 0.5  # unnormalize
        npimg = img.numpy()
        img2 = Image.fromarray(np.transpose(npimg, (1, 2, 0)), 'RGB')
        img2.show()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()


def main():
    app = wx.App()
    ex = GuiMl(None, title='Fix Flipped Images')
    ex.Show()
    app.MainLoop()


if __name__ == '__main__':
    main()
