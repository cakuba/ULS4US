from __future__ import print_function
import os
import cv2
import numpy as np

def get_data_orig(path, img_dir='img',mask_dir='mask', size=(256,256),normalized=True, preprocessing=False):
    """
    获得模型学习所需要的数据；
    其中图像格式(num_images, weight, height)
    标注格式(num_images, weight, height)，像素值为0/1
    注：训练数据目录结构如下
    training/
        img/
        mask/
    """

    files = os.listdir(os.path.join(path, img_dir))
    files.sort()
    images = np.zeros([len(files),size[0],size[1]])
    for i, file in enumerate(files):
        img = cv2.imread(os.path.join(path, img_dir, file),cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, size, cv2.INTER_AREA)
        if preprocessing:
            img = image_histEqu(img) 
        if normalized:
            images[i,:,:] = img/255
        else:
            images[i,:,:] = img

    files = os.listdir(os.path.join(path, mask_dir))
    files.sort()
    label = np.zeros([len(files),size[0],size[1]],dtype=np.uint8)
    for i, file in enumerate(files):
        img = cv2.imread(os.path.join(path, mask_dir, file),cv2.IMREAD_GRAYSCALE)
        # BE CAREFUL!
        img[img>0]=255
        img = cv2.resize(img, size, cv2.INTER_AREA)
        _, img_bin = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
        label[i,:,:] = img_bin

    return images, label

def image_histEqu(img, climit=4.0):
    """ 自适应的图像直方图均衡化+灰度拉伸处理
    
    输入：
        img - 图像矩阵（numpy格式）
    """
    
    # 自适应直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=climit, tileGridSize=(8,8))
    img_hist = clahe.apply(img)
    
    # 灰度拉伸
    img_min = np.min(img_hist)
    img_max = np.max(img_hist)
    
    return (img_hist-img_min)/(img_max-img_min)*255

def cropBoundingBox(mask,thres=127,paddingRatio=1.0):
    """
       给定轮廓标注，提供轮廓标注对应的外接矩形坐标
    """

    # 图像大小
    height = mask.shape[0]
    width = mask.shape[1]
    # 二值化
    _, mask_bin = cv2.threshold(mask, thres, 255, cv2.THRESH_BINARY)
    mask_bin = mask_bin.astype('uint8')
    # 降噪（删除ROI以外的噪点）
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    mask_open = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel)
    # 寻找轮廓
    contours, hierarchy = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 如果没有获得轮廓，返回输入图像的大小坐标
    if len(contours)==0:
        print("WARNING: cannot find the ROI!")
        return 0,width,0,height
    # 将轮廓按面积大小排序
    contours.sort(key=lambda c:cv2.contourArea(c), reverse=True)
    # 获得最大轮廓对应的外接矩形
    x,y,w,h = cv2.boundingRect(contours[0])
    # 缩放外接矩形
    center_x = x + w/2
    center_y = y + h/2
    nw = w*paddingRatio
    nh = h*paddingRatio
    
    # 左上角左边
    upperLeft_x = int(center_x-nw/2)
    if (upperLeft_x<0):
        upperLeft_x = 0

    upperLeft_y = int(center_y-nh/2)
    if (upperLeft_y<0):
        upperLeft_y = 0
    
    # 右下角左边
    lowerRight_x = int(upperLeft_x+nw)
    if (lowerRight_x>width):
        lowerRight_x = width
    
    lowerRight_y = int(upperLeft_y+nh)
    if (lowerRight_y>height):
        lowerRight_y = height
        
    # 返回截取后的图像
    return upperLeft_x,lowerRight_x,upperLeft_y,lowerRight_y

def roiLocator(imgs, masks, outputs, width, height, paddingRatio=1.1, thres=0.5, num=1):
    """
    To locate the ROI in the output image, usually based on the maximum contour area in the outputs;
       and then, crop the original image and mask based on the ROI

    Parameters:
    imgs    - the original image containing the ROIs; (batch_size, *, *, 1)
    masks   - the segmentaton mask for the image;     (batch_size, *, *, 1)
    outputs - the output image used for locating ROI;
    width   - the width of cropped image containing the ROIs
    height  - the height of cropped image containing the ROIs
    num     - the number of ROIs in the image; the default value is 1
    thres   - the threshold value to binarize the output image
    paddingRatio - the bounding box size relative to the maximum contour

    Returns:
    imgs_cropped - the original image cropped with the ROIs;    (batch_size, width, height, 1)
    masks_cropped - the segmentaton mask cropped with the ROIs; (batch_size, width, height, 1)
    outputs_cropped - cropped outputs with the ROIs;            (batch_size, width, height, 1)
    """

    batch_size, w, h, _ = imgs.shape
    imgs_cropped = np.zeros((batch_size,height,width))
    masks_cropped = np.zeros((batch_size,height,width))
    outputs_cropped = np.zeros((batch_size,height,width))
    for bs in range(batch_size):
        img = np.squeeze(imgs[bs])
        mask = np.squeeze(masks[bs])
        output = np.squeeze(outputs[bs])

        # 获得ROI边框坐标
        upperLeft_x,lowerRight_x,upperLeft_y,lowerRight_y = cropBoundingBox(output,thres,paddingRatio)
        # 获得ROI边框对应的cropped原始图像，并缩放到指定(height,width)
        imgs_cropped[bs] = cv2.resize(img[upperLeft_y:lowerRight_y,upperLeft_x:lowerRight_x],(height,width),cv2.INTER_AREA)
         # 获得ROI边框对应的cropped原始mask，并缩放到指定(height,width)
        tmp = cv2.resize(mask[upperLeft_y:lowerRight_y,upperLeft_x:lowerRight_x],(height,width),cv2.INTER_AREA)
        _, masks_cropped[bs] = cv2.threshold(tmp,thres,1,cv2.THRESH_BINARY)
        # 获得ROI边框对应的cropped outputs图像，并缩放到指定(height,width)
        outputs_cropped[bs] = cv2.resize(output[upperLeft_y:lowerRight_y,upperLeft_x:lowerRight_x],(height,width),cv2.INTER_AREA)


    imgs_cropped = np.expand_dims(imgs_cropped, axis=-1)
    masks_cropped = np.expand_dims(masks_cropped, axis=-1)
    return imgs_cropped, masks_cropped, outputs_cropped

def randomCrop(imgs, masks, width, height):
    """
    To randomly crop the images and the corresponding masks to the size of width and height
    """

    hw = int(width/2)
    hh = int(height/2)
    batch_size, w, h, _ = imgs.shape
    imgs_cropped = np.zeros((batch_size,height,width))
    masks_cropped = np.zeros((batch_size,height,width))
    for bs in range(batch_size):
        img = np.squeeze(imgs[bs])
        mask = np.squeeze(masks[bs])

        # 随机选取中心点
        center_x = np.random.choice(range(hw,w-hw))
        center_y = np.random.choice(range(hh,h-hh))
        upperLeft_x,lowerRight_x,upperLeft_y,lowerRight_y = center_x-hw, center_x+hw, center_y-hh, center_y+hh
        # 获得ROI边框对应的cropped原始图像，并缩放到指定(height,width)
        imgs_cropped[bs] = img[upperLeft_y:lowerRight_y,upperLeft_x:lowerRight_x]
         # 获得ROI边框对应的cropped原始mask，并缩放到指定(height,width)
        masks_cropped[bs] = mask[upperLeft_y:lowerRight_y,upperLeft_x:lowerRight_x]

    imgs_cropped = np.expand_dims(imgs_cropped, axis=-1)
    masks_cropped = np.expand_dims(masks_cropped, axis=-1)
    return imgs_cropped, masks_cropped    
