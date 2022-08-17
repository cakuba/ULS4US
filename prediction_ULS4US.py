#  
#  Visualization of ULS4US prdictions
#

# to filter some unnecessory warning messages
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

from models import unet
from utils import getData
from utils import printConfig
from utils import metrics
from configparser import ConfigParser

cfg_file = "./conf/training.conf"
data_file = "./conf/dataset.conf"

# 获得配置参数
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

# 获得训练相关数据
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

# inference pipeline
img = test_images
target = test_labels
# random crop
img2, img2_target = getData.randomCrop(img,target,width=crop_width,height=crop_height)
# Stage I
stage1_img1_output, stage1_img2_output, _ = model.predict([img,target,img2,img2_target,img2_target])
# Stage II - relocate the lesion
stage2_img2, stage2_img2_target, stage1_output1_cropped = getData.roiLocator(img,target,stage1_img1_output,
                                                    width=crop_width,height=crop_height,paddingRatio=padding_ratio)
# Stage II - obtain the prediction
stage2_img1_output, stage2_img2_output, _ = model.predict([img,target,stage2_img2,stage2_img2_target,stage1_output1_cropped])

# update the prediction for the second branch
thres = 0.5
stage2_img2_output_ORIG = np.zeros(stage2_img1_output.shape,dtype=np.float32)
for bs in range(nums_of_test_images):
    output = np.squeeze(stage1_img1_output[bs])
    # Bounding Box
    upperLeft_x,lowerRight_x,upperLeft_y,lowerRight_y = getData.cropBoundingBox(output,thres,padding_ratio)
    tmp_height = lowerRight_y-upperLeft_y
    tmp_width = lowerRight_x-upperLeft_x
    tmp = cv2.resize(np.squeeze(stage2_img2_output[bs]),(tmp_width,tmp_height),cv2.INTER_AREA)
    stage2_img2_output_ORIG[bs,upperLeft_y:lowerRight_y,upperLeft_x:lowerRight_x] = np.expand_dims(tmp, axis=-1)

# obtain the finle prediction based on soft voting
stage2_img1_dice = metrics.dice(target,stage2_img1_output)
stage2_img2_dice = metrics.dice(target,stage2_img2_output_ORIG)
y_pred1 = ops.convert_to_tensor_v2_with_dispatch(stage2_img1_output)
y_pred2 = ops.convert_to_tensor_v2_with_dispatch(stage2_img2_output_ORIG)
y_pred1_weight, y_pred2_weight = metrics.softmax([stage2_img1_dice,stage2_img2_dice])
stage2_votingSD = tf.add(y_pred1*y_pred1_weight,y_pred2*y_pred2_weight) 

# visualization of ULS4US predictions
for bs in range(nums_of_test_images):
    plt.figure(figsize=(6,14))
    plt.subplot(3,1,1)
    plt.imshow(np.squeeze(img[bs], axis=-1),cmap='gray')

    plt.subplot(3,1,2)
    plt.imshow(np.squeeze(target[bs], axis=-1),cmap='gray')

    plt.subplot(3,1,3)
    _, tmp = cv2.threshold(np.squeeze(stage2_votingSD[bs], axis=-1), 0.5, 1, cv2.THRESH_BINARY)
    plt.imshow(tmp,cmap='gray')

    plt.savefig("./pred/%s_%s.png" %(experiment_key, str(bs).zfill(3)),dpi=300)
    plt.close()