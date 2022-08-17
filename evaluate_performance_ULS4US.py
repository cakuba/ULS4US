#  
#  Performance evaluation for ULS4US
#

# to filter some unnecessory warning messages
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import sys
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from models import unet
from utils import getData
from utils import printConfig
from configparser import ConfigParser

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
cfg_file = "./conf/training.conf"
data_file = "./conf/dataset.conf"

# parameters setting
cfg=ConfigParser()
cfg.read(cfg_file)
experiment_key = 'ULS4US'
try:
    paras=dict(cfg.items(experiment_key))
except:
    print("Configuration information error!")
    sys.exit()

# GPU config
os.environ["CUDA_VISIBLE_DEVICES"] = paras['gpu_id']

# training/validation dataset
width = int(paras['width'])
height = int(paras['width'])
img_size = (width, height)
crop_width = int(paras['crop_width'])
crop_height = int(paras['crop_height'])
preprocessing = paras['preprocessing']=='True'
seed = int(paras['seed'])

# model training parameters
batch_size = int(paras['batch_size'])
epochs = int(paras['epochs'])
lr = float(paras['lr'])
shuffle = paras['shuffle']=='True'
loss_mode = int(paras['loss_mode'])
history_saved = paras['history_saved']=='True'
augmented = paras['augmented']=='True'
nums_of_relocation = int(paras['nums_of_relocation'])
padding_ratio = float(paras['padding_ratio'])
if ("," in paras['beta']):
    beta1 = float(paras['beta'].split(",")[0])
    beta2 = float(paras['beta'].split(",")[1])
else:
    beta1 = float(paras['beta'])
    beta2 = float(paras['beta'])
printConfig.configInfo(paras)

# dataset information
dataset = paras['dataset']
cfg.read(data_file)
try:
    paras=dict(cfg.items(dataset))
except:
    print("dataset information error!")
    sys.exit()
data_dir = paras['test_data_dir']
dataName = paras['data_name']    
printConfig.configInfo(paras)

# obtain test dataset
test_images, test_labels = getData.get_data_orig(data_dir,size=(width, height),preprocessing=preprocessing)
test_images = np.expand_dims(test_images, axis=-1)
test_labels = np.expand_dims(test_labels, axis=-1)
nums_of_test_images = len(test_images)
print("[LOG] A total number of %d test images used for MIMO-UNet performance evaluation..." % (nums_of_test_images))

# MIMO-UNet     
model = unet.UNet_mimo(img1_width=width,img1_height=height,img2_width=crop_width,img2_height=crop_height,
                        loss_mode=loss_mode,beta1=beta1,beta2=beta2)
model.compile(optimizer=Adam(lr=lr),loss=[None]*len(model.outputs))

# load the trained weight file
weights_file = os.path.join('./weights/',experiment_key+'_model-weights-epoch-'+str(epochs).zfill(2)+'-lossmode-'+str(loss_mode)+'_'+dataName+'.hdf5')
model.load_weights(weights_file)

# 性能评估
from utils import metrics
thres = 0.5  # 对模型输出的结果进行二值化操作的阈值
metrics.evaluate_mimo(test_images,test_labels,model,crop_width=crop_width,
                      crop_height=crop_height,padding_ratio=padding_ratio,thres=thres)