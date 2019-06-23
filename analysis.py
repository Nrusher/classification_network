#%%
import utils
import copy
import pandas as pd
import os
import shutil
import numpy as np
import progressbar as pbar
import sys
import time
import logging
import pathlib
import re
from PIL import Image
import matplotlib.pyplot as plt

def filter_data(l,a= 0.2):
    lc = l.copy()
    for i in range(1,len(l)):
        lc[i] = lc[i]*a + lc[i-1]*(1-a)
    return lc

#%%
#对比验证集损失函

fig,ax = plt.subplots()

data = utils.transform_log_to_pd_dataframe("./logs/traffic_LeNet_coslr_64x64_16_0.01.log")
ax.plot(data.epoch.values,filter_data(data.val_loss.values),label = 'LeNet')

data = utils.transform_log_to_pd_dataframe("./logs/traffic_Alexnet_coslr_64x64_16_0.005.log")
ax.plot(data.epoch.values,filter_data(data.val_loss.values),label = 'AlexNet')

data = utils.transform_log_to_pd_dataframe("./logs/traffic_ResNet18_coslr_64x64_16_0.01.log")
ax.plot(data.epoch.values,filter_data(data.val_loss.values),label = 'ResNet18')

data = utils.transform_log_to_pd_dataframe("./logs/traffic_VGG11_coslr_64x64_16_0.01.log")
ax.plot(data.epoch.values,filter_data(data.val_loss.values),label = 'VGG11')

data = utils.transform_log_to_pd_dataframe("./logs/traffic_DesenNet_coslr_64x64_16_0.01.log")
ax.plot(data.epoch.values,filter_data(data.val_loss.values),label = 'DesenNet')

data = utils.transform_log_to_pd_dataframe("./logs/traffic_EfficientNetb0_coslr_64x64_16_0.01.log")
ax.plot(data.epoch.values,filter_data(data.val_loss.values),label = 'EfficientNetb0')

ax.legend()
ax.set_xlabel("epoch")
ax.set_ylim(0, 8)
ax.set_ylabel("val_loss")
ax.set_title("epoch/val_loss")
plt.pause(0.1)

#%%
#对比验证集准确率

fig,ax = plt.subplots()

data = utils.transform_log_to_pd_dataframe("./logs/traffic_LeNet_coslr_64x64_16_0.01.log")
ax.plot(data.epoch.values,filter_data(data.val_acc.values),label = 'LeNet')
max_acc = data.val_acc.max()
print('LeNet:',max_acc)

data = utils.transform_log_to_pd_dataframe("./logs/traffic_Alexnet_coslr_64x64_16_0.005.log")
ax.plot(data.epoch.values,filter_data(data.val_acc.values),label = 'AlexNet')
max_acc = data.val_acc.max()
print('AlexNet:',max_acc)

data = utils.transform_log_to_pd_dataframe("./logs/traffic_ResNet18_coslr_64x64_16_0.01.log")
ax.plot(data.epoch.values,filter_data(data.val_acc.values),label = 'ResNet18')
max_acc = data.val_acc.max()
print('ResNet18:',max_acc)

data = utils.transform_log_to_pd_dataframe("./logs/traffic_VGG11_coslr_64x64_16_0.01.log")
ax.plot(data.epoch.values,filter_data(data.val_acc.values),label = 'VGG11')
max_acc = data.val_acc.max()
print('VGG11:',max_acc)

data = utils.transform_log_to_pd_dataframe("./logs/traffic_DesenNet_coslr_64x64_16_0.01.log")
ax.plot(data.epoch.values,filter_data(data.val_acc.values),label = 'DesenNet')
max_acc = data.val_acc.max()
print('DesenNet:',max_acc)

data = utils.transform_log_to_pd_dataframe("./logs/traffic_EfficientNetb0_coslr_64x64_16_0.01.log")
ax.plot(data.epoch.values,filter_data(data.val_acc.values),label = 'EfficientNetb0')
max_acc = data.val_acc.max()
print('EfficientNetb0:',max_acc)

ax.legend()
ax.set_xlabel("epoch")
ax.set_ylim(70, 100)
ax.set_ylabel("val_acc")
ax.set_title("epoch/val_acc")
plt.pause(0.1)

#%%
#学习率变化图

fig,ax = plt.subplots()

data = utils.transform_log_to_pd_dataframe("./logs/traffic_EfficientNetb0_coslr_64x64_16_0.01.log")
ax.plot(data.epoch.values,data.lr.values)

ax.legend()
ax.set_xlabel("epoch")
ax.set_ylabel("learn rate")
ax.set_title("epoch/lr")
plt.pause(0.1)

#%%
# batch size 大小影响
fig,ax = plt.subplots()

data = utils.transform_log_to_pd_dataframe("./logs/traffic_ResNet18_coslr_64x64_16_0.01.log")
ax.plot(data.epoch.values,data.val_loss.values,label = 'batch = 16')

data = utils.transform_log_to_pd_dataframe("./logs/traffic_ResNet18_coslr_64x64_64_0.01.log")
ax.plot(data.epoch.values,data.val_loss.values,label = 'batch = 64')
ax.legend()
ax.set_xlabel("epoch")
ax.set_ylim(0, 8)
ax.set_ylabel("val_loss")
ax.set_title("ResNet18 epoch/val_loss")
plt.pause(0.1)

#%%
# batch size 大小影响
fig,ax = plt.subplots()

data = utils.transform_log_to_pd_dataframe("./logs/traffic_ResNet18_coslr_64x64_16_0.01.log")
ax.plot(data.epoch.values,data.val_acc.values,label = 'batch = 16')

data = utils.transform_log_to_pd_dataframe("./logs/traffic_ResNet18_coslr_64x64_64_0.01.log")
ax.plot(data.epoch.values,data.val_acc.values,label = 'batch = 64')
ax.legend()
ax.set_xlabel("epoch")
ax.set_ylim(70, 100)
ax.set_ylabel("val_acc")
ax.set_title("ResNet18 epoch/val_acc")
plt.pause(0.1)

#%%
# 图像尺寸影响
fig,ax = plt.subplots()

data = utils.transform_log_to_pd_dataframe("./logs/traffic_LeNet_224x224_16_0.01.log")
ax.plot(data.epoch.values,filter_data(data.val_loss.values),label = '224x224')

data = utils.transform_log_to_pd_dataframe("./logs/traffic_LeNet_coslr_64x64_16_0.01.log")
ax.plot(data.epoch.values,filter_data(data.val_loss.values),label = '64x64')

ax.legend()
ax.set_xlabel("epoch")
ax.set_ylim(0, 8)
ax.set_ylabel("val_loss")
ax.set_title("LeNet epoch/val_loss")
plt.pause(0.1)

fig,ax = plt.subplots()

data = utils.transform_log_to_pd_dataframe("./logs/traffic_LeNet_224x224_16_0.01.log")
ax.plot(data.epoch.values,filter_data(data.val_acc.values),label = '224x224')

data = utils.transform_log_to_pd_dataframe("./logs/traffic_LeNet_coslr_64x64_16_0.01.log")
ax.plot(data.epoch.values,filter_data(data.val_acc.values),label = '64x64')

ax.legend()
ax.set_xlabel("epoch")
ax.set_ylim(70, 100)
ax.set_ylabel("val_acc")
ax.set_title("LeNet epoch/val_acc")
plt.pause(0.1)

#%%
plt.show()