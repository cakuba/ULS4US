# 
# performance evaluation metrics
#

import cv2
import numpy as np
import tensorflow as tf
import tensorflow.keras.metrics as metrics
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops

from scipy.spatial.distance import directed_hausdorff
from . import getData


def dice(y_true, y_pred):
    """
    to compute the Dice Coef. between the human annoation and network prediction 
    
    Argumetns:
       y_true - human annotion in the format of [batch_size, width, height]
       y_pred - network prediction in the format of [batch_size, width, height] 
    
    """
    
    smooth = 1.0e-5
    y_pred = ops.convert_to_tensor_v2(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    intersection = tf.reduce_sum(tf.multiply(y_true, y_pred),axis=[1,2])
    y_true_sum = tf.reduce_sum(tf.multiply(y_true, y_true),axis=[1,2])
    y_pred_sum = tf.reduce_sum(tf.multiply(y_pred, y_pred),axis=[1,2])
    dc = (2.0*intersection+smooth)/(y_true_sum+y_pred_sum+smooth)
    
    return tf.reduce_mean(dc).numpy()

def iou(y_true, y_pred):
    """
       IOU (intersection over union) between the human annotation and network prediction
    """

    smooth = 1.0e-5
    y_pred = ops.convert_to_tensor_v2(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)

    intersection = tf.reduce_sum(tf.multiply(y_true, y_pred),axis=[1,2])
    y_true_sum = tf.reduce_sum(tf.multiply(y_true, y_true),axis=[1,2])
    y_pred_sum = tf.reduce_sum(tf.multiply(y_pred, y_pred),axis=[1,2])
    dc = (2.0*intersection+smooth)/(y_true_sum+y_pred_sum+smooth)
    im = tf.divide(dc, tf.subtract(2.0,dc))

    return tf.reduce_mean(im).numpy()

def mIOU(y_true, y_pred, thres=0.5):
    """
    compute mean IOU as tp / (tp + fp + fn)

    Note:
       TF API MeanIoU is buggy as
            m = metrics.MeanIoU(num_classes=num_of_classes)
            m.update_state(y_true,y_pred)
    """
    m = metrics.TruePositives(thresholds=thres)
    m.update_state(y_true, y_pred)
    tp = m.result().numpy()

    n = metrics.FalsePositives(thresholds=thres)
    n.update_state(y_true, y_pred)
    fp = n.result().numpy()

    p = metrics.FalseNegatives(thresholds=thres)
    p.update_state(y_true, y_pred)
    fn = p.result().numpy()   

    return tp/(tp+fp+fn)

def Hausdorff95(y_true, y_pred, thres=0.5):
    """
    Computes the Hausdorff distance from y_true to y_pred.

    Note:
        Hausdorff distance from y_true to y_pred is defined as the maximum
        of all distances from a point in y_true to the closest point in
        y_pred. It is an asymmetric metric.
    """

    # 对y_pred进行二值化
    y_pred = ops.convert_to_tensor_v2_with_dispatch(y_pred)
    threshold = math_ops.cast(thres, y_pred.dtype)
    y_pred = math_ops.cast(y_pred > thres, y_pred.dtype)

    # 计算Hausdorff距离
    hd = []
    for i in range(len(y_true)):
        yt = np.squeeze(y_true[i])
        yp = np.squeeze(y_pred[i])

        d1 = directed_hausdorff(yt, yp)[0]
        d2 = directed_hausdorff(yp, yt)[0]
        hd.append(max(d1,d2)*0.95)

    return np.mean(hd)

def acc(y_true, y_pred, thres=0.5):
    """
    Compute the binary accuracy of y_pred to y_true given a threshold value

    Note:
        acc = (TP+TN)/(TP+FP+TN+FN)
    """

    m = metrics.BinaryAccuracy(threshold=thres)
    m.update_state(y_true,y_pred)

    return m.result().numpy()

def precision(y_true, y_pred, thres=0.5):
    """
    Compute the precision of y_pred to y_true

    Note:
        acc = TP/(TP+FP)
    """

    m = metrics.Precision(thresholds=thres)
    m.update_state(y_true,y_pred)

    return m.result().numpy()

def recall(y_true, y_pred, thres=0.5):
    """
    Computer the recall of y_pred to y_true

    Note:
        acc = TP/(TP+FN)
    """

    m = metrics.Recall(thresholds=thres)
    m.update_state(y_true,y_pred)

    return m.result().numpy()

def sensitivity(y_true, y_pred, thres=0.5):
    """
    Compute the precision of y_pred to y_true, same as RECALL

    Note:
        acc = TP/(TP+FN)
    """

    return recall(y_true,y_pred,thres=thres)

def specificity(y_true, y_pred, thres=0.5):
    """
    Computer the recall of y_pred to y_true

    Note:
        acc = TN/(TN+FP)
    """

    m = metrics.TrueNegatives(thresholds=thres)
    m.update_state(y_true, y_pred)
    tn = m.result().numpy()

    n = metrics.FalsePositives(thresholds=thres)
    n.update_state(y_true, y_pred)
    fp = n.result().numpy()

    return tn/(tn+fp)

def F1(y_true, y_pred, thres=0.5):
    """
    Computer F1 score of y_pred to y_true
    """

    p = precision(y_true, y_pred, thres=thres)
    r = recall(y_true, y_pred, thres=thres)

    return 2*p*r/(p+r)

def auc(y_true, y_pred, num_of_thresholds=200):
    """
    Compute ROC between y_pred and y_true
    """

    m = metrics.AUC(num_thresholds=num_of_thresholds)
    mauc = []
    for i in range(len(y_true)):
        yt = np.squeeze(y_true[i]).ravel()
        yp = np.squeeze(y_pred[i]).ravel()
        m.update_state(yt,yp)
        a = m.result().numpy()
        mauc.append(a)

        m.reset_states()

    return np.mean(mauc)

def evaluate(y_true,y_pred,thres=0.5):
    """
    Evaluate all performance metrics with default parameters

    Metrics:
         Dice/Hausdorff95/Accuracy/AUC/Precision/Recall/F1/Sensitivity/Specificity
    """

    dc = dice(y_true,y_pred)
    hausdorff95 = Hausdorff95(y_true,y_pred,thres=thres)
    mauc = auc(y_true,y_pred)
    macc = acc(y_true,y_pred,thres=thres)
    mprecision = precision(y_true,y_pred,thres=thres)
    mrecall = recall(y_true,y_pred,thres=thres)
    msensitivity = sensitivity(y_true,y_pred,thres=thres)
    mspecificity = specificity(y_true,y_pred,thres=thres)
    mF1 = F1(y_true,y_pred,thres=thres)
    mIoU = mIOU(y_true,y_pred,thres=thres)

    print("=============================")
    print("Accuracy    : %6.3f" % macc)
    print("Dice Coef   : %6.3f" % dc)
    print("Hausdorff95 : %6.3f" % hausdorff95)
    print("AUC ROC     : %6.3f" % mauc)
    print("Precision   : %6.3f" % mprecision)
    print("Recall      : %6.3f" % mrecall)
    print("Sensitivity : %6.3f" % msensitivity)
    print("Specificity : %6.3f" % mspecificity)
    print("F1          : %6.3f" % mF1)
    print("mIoU        : %6.3f" % mIoU)
    print("=============================\n")
   
def softmax(x):
    """
        softmax计算实现权重归一化
    """
    e_x = np.exp(x - np.max(x)) # 防止exp()数值溢出
    return e_x / e_x.sum(axis=0)

def evaluate_mimo(img, target, model, crop_width=256, crop_height=256, padding_ratio=1.1, thres=0.5):
    """
    Performance evaluation of Unet_mimo models based on the combination of outputs from two branches
    """
    # 随机crop训练图像和对应的标注
    img2, img2_target = getData.randomCrop(img,target,width=crop_width,height=crop_height)

    # Stage I
    stage1_img1_output, stage1_img2_output, _ = model.predict([img,target,img2,img2_target,img2_target])

    # Stage II - 基于stage I输出更新第二分支输入
    stage2_img2, stage2_img2_target, stage1_output1_cropped = getData.roiLocator(img,target,stage1_img1_output,
                                                        width=crop_width,height=crop_height,paddingRatio=padding_ratio)

    # Stage II - 更新模型预测结果
    stage2_img1_output, stage2_img2_output, _ = model.predict([img,target,stage2_img2,stage2_img2_target,stage1_output1_cropped])

    # 第一分支的原始性能
    stage2_img1_acc = acc(target,stage2_img1_output)
    stage2_img1_dice = dice(target,stage2_img1_output)
    stage2_img1_hd95 = Hausdorff95(target,stage2_img1_output)
    stage2_img1_mIOU = mIOU(target,stage2_img1_output)

    # 第二分支的原始性能（将cropped的结果放回到原始位置）
    stage2_img2_output_ORIG = np.zeros(stage2_img1_output.shape,dtype=np.float32)
    batch_size = stage1_img1_output.shape[0]
    for bs in range(batch_size):
        output = np.squeeze(stage1_img1_output[bs])
        # 获得ROI边框坐标
        upperLeft_x,lowerRight_x,upperLeft_y,lowerRight_y = getData.cropBoundingBox(output,thres,padding_ratio)
        height = lowerRight_y-upperLeft_y
        width = lowerRight_x-upperLeft_x
        tmp = cv2.resize(np.squeeze(stage2_img2_output[bs]),(width,height),cv2.INTER_AREA)
        stage2_img2_output_ORIG[bs,upperLeft_y:lowerRight_y,upperLeft_x:lowerRight_x] = np.expand_dims(tmp, axis=-1)

    stage2_img2_acc = acc(target,stage2_img2_output_ORIG)
    stage2_img2_dice = dice(target,stage2_img2_output_ORIG)
    stage2_img2_hd95 = Hausdorff95(target,stage2_img2_output_ORIG)
    stage2_img2_mIOU = mIOU(target,stage2_img2_output_ORIG)

    # 二值化
    y_pred1 = ops.convert_to_tensor_v2_with_dispatch(stage2_img1_output)
    y_pred1 = math_ops.cast(y_pred1 > thres, y_pred1.dtype)
    y_pred2 = ops.convert_to_tensor_v2_with_dispatch(stage2_img2_output_ORIG)
    y_pred2 = math_ops.cast(y_pred2 > thres, y_pred2.dtype)

    # 第一/第二分支AND操作后的性能
    stage2_imgAND = tf.multiply(y_pred1,y_pred2)
    stage2_imgAND_acc = acc(target,stage2_imgAND)
    stage2_imgAND_dice = dice(target,stage2_imgAND)
    stage2_imgAND_hd95 = Hausdorff95(target,stage2_imgAND)
    stage2_imgAND_mIOU = mIOU(target,stage2_imgAND)

    # 第一/第二分支OR操作后的性能
    stage2_imgOR = tf.add(y_pred1,y_pred2)
    stage2_imgOR = math_ops.cast(stage2_imgOR >= 1, y_pred2.dtype)
    stage2_imgOR_acc = acc(target,stage2_imgOR)
    stage2_imgOR_dice = dice(target,stage2_imgOR)
    stage2_imgOR_hd95 = Hausdorff95(target,stage2_imgOR)
    stage2_imgOR_mIOU = mIOU(target,stage2_imgOR)

    # 第一/第二分支soft voting (ACC)后的性能
    y_pred1 = ops.convert_to_tensor_v2_with_dispatch(stage2_img1_output)
    y_pred2 = ops.convert_to_tensor_v2_with_dispatch(stage2_img2_output_ORIG)
    y_pred1_weight, y_pred2_weight = softmax([stage2_img1_acc,stage2_img2_acc])
    stage2_votingS = tf.add(y_pred1*y_pred1_weight,y_pred2*y_pred2_weight)
    stage2_votingS_acc = acc(target,stage2_votingS)
    stage2_votingS_dice = dice(target,stage2_votingS)
    stage2_votingS_hd95 = Hausdorff95(target,stage2_votingS)
    stage2_votingS_mIOU = mIOU(target,stage2_votingS)

    # 第一/第二分支soft voting (Dice)后的性能
    y_pred1 = ops.convert_to_tensor_v2_with_dispatch(stage2_img1_output)
    y_pred2 = ops.convert_to_tensor_v2_with_dispatch(stage2_img2_output_ORIG)
    y_pred1_weight, y_pred2_weight = softmax([stage2_img1_dice,stage2_img2_dice])
    stage2_votingSD = tf.add(y_pred1*y_pred1_weight,y_pred2*y_pred2_weight)
    stage2_votingSD_acc = acc(target,stage2_votingSD)
    stage2_votingSD_dice = dice(target,stage2_votingSD)
    stage2_votingSD_hd95 = Hausdorff95(target,stage2_votingSD)
    stage2_votingSD_mIOU = mIOU(target,stage2_votingSD)

    print("                        -- ACC --  Dice  -- HD95  -- mIOU ")
    print(" 1ST BRANCH ONLY          {:6.3f}  {:6.3f} {:6.3f} {:6.3f} ".format(stage2_img1_acc,stage2_img1_dice,stage2_img1_hd95,stage2_img1_mIOU))
    print(" 2ND BRANCH ONLY          {:6.3f}  {:6.3f} {:6.3f} {:6.3f} ".format(stage2_img2_acc,stage2_img2_dice,stage2_img2_hd95,stage2_img2_mIOU))
    print(" LOGICAL AND              {:6.3f}  {:6.3f} {:6.3f} {:6.3f} ".format(stage2_imgAND_acc,stage2_imgAND_dice,stage2_imgAND_hd95,stage2_imgAND_mIOU))
    print(" LOGICAL OR               {:6.3f}  {:6.3f} {:6.3f} {:6.3f} ".format(stage2_imgOR_acc,stage2_imgOR_dice,stage2_imgOR_hd95,stage2_imgOR_mIOU))
    print(" Soft Voting(ACC)         {:6.3f}  {:6.3f} {:6.3f} {:6.3f} ".format(stage2_votingS_acc,stage2_votingS_dice,stage2_votingS_hd95,stage2_votingS_mIOU))
    print(" Soft Voting(Dice)        {:6.3f}  {:6.3f} {:6.3f} {:6.3f} ".format(stage2_votingSD_acc,stage2_votingSD_dice,stage2_votingSD_hd95,stage2_votingSD_mIOU))
    print("")