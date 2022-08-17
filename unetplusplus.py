#
#  UNet++ for Lesion segmentation in Ultrasound Images
#

# to filter some unnecessory warning messages
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import sys
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from models import unetplusplus as models
from utils import getData
from utils import printConfig
from configparser import ConfigParser

cfg_file = "./conf/training.conf"
data_file = "./conf/dataset.conf"

if __name__ == "__main__":
    # 获得配置参数
    experiment_key = sys.argv[0].split(".")[0]
    cfg=ConfigParser()
    cfg.read(cfg_file)
    try:
        paras=dict(cfg.items(experiment_key))
    except:
        print("Configuration information error!")
        sys.exit()

    # GPU config
    os.environ["CUDA_VISIBLE_DEVICES"] = paras['gpu_id']

    # whether to train from scratch
    resumed_training = paras['resumed_training']=='True'

    # training/validation dataset
    width = int(paras['width'])
    height = int(paras['width'])
    img_size = (width, height)
    preprocessing = paras['preprocessing']=='True'
    seed = int(paras['seed'])

    # model training parameters
    batch_size = int(paras['batch_size'])
    epochs = int(paras['epochs'])
    lr = float(paras['lr'])
    shuffle = paras['shuffle']=='True'
    history_saved = paras['history_saved']=='True'
    augmented = paras['augmented']=='True'
    printConfig.configInfo(paras)

    # 获得训练相关数据
    dataset = paras['dataset']
    cfg.read(data_file)
    try:
        paras=dict(cfg.items(dataset))
    except:
        print("dataset %s error!" % dataset)
        sys.exit()
    data_dir = paras['training_data_dir']
    dataName = paras['data_name']      
    printConfig.configInfo(paras)

    # 获得训练和验证数据集
    images, labels = getData.get_data_orig(data_dir,size=(width, height),preprocessing=preprocessing)    
    train_images, valid_images, train_labels, valid_labels = train_test_split(images, labels, test_size=0.18, random_state=seed)
    # 填充最后一个维度到3个通道
    train_images = np.expand_dims(train_images, axis=-1).repeat(3,axis=-1)
    train_labels = np.expand_dims(train_labels, axis=-1)
    valid_images = np.expand_dims(valid_images, axis=-1).repeat(3,axis=-1)
    valid_labels = np.expand_dims(valid_labels, axis=-1)
    nums_of_train_images = len(train_images)
    nums_of_valid_images = len(valid_images)
    print(train_images.shape,valid_images.shape)    
    print("[TRAINING LOG] A total number of %d training and %d valid images fed into the network training..." % (nums_of_train_images,nums_of_valid_images))

    # 模型训练权重和日志文件
    weights_file = os.path.join('./weights/',experiment_key+'_model-weights-epoch-'+str(epochs).zfill(2)+'_'+dataName+'.hdf5')
    last_weights_file = os.path.join('./weights/',experiment_key+'_model-weights-epoch-'+str(epochs).zfill(2)+'_'+dataName+'_FINAL.hdf5')
    history_file = os.path.join('./history', experiment_key+'_history-epoch-'+str(epochs)+'_'+dataName+'.dat')

    # 训练数据增强
    if augmented:
        data_gen_args = dict(rotation_range=10,
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            zoom_range=0.2,
                            fill_mode="wrap")
        image_datagen = ImageDataGenerator(**data_gen_args)
        mask_datagen = ImageDataGenerator(**data_gen_args)
    else:     
        image_datagen = ImageDataGenerator()
        mask_datagen = ImageDataGenerator()
    image_generator = image_datagen.flow(train_images,seed=seed,batch_size=batch_size,shuffle=shuffle)
    mask_generator = mask_datagen.flow(train_labels,seed=seed,batch_size=batch_size,shuffle=shuffle)
    train_generator = zip(image_generator, mask_generator)

    # fcn16模型初始化
    model = models.Xnet()
    model.compile(optimizer=Adam(lr=lr),loss=tf.keras.losses.BinaryCrossentropy(),metrics=['accuracy'])
    checkpoint = ModelCheckpoint(weights_file,monitor='val_loss',verbose=1, mode='min', save_best_only=True)

    if resumed_training:
        model.load_weights(last_weights_file)
        #model.load_weights(weights_file)
        print("[TRAINING LOG] The model is trained from a previous checkpoint... \n")

    # network training
    steps_per_epoch = int(nums_of_train_images/batch_size)+1
    # network training
    h = model.fit(train_generator,
                        epochs=epochs,
                        steps_per_epoch=steps_per_epoch,
                        validation_data=(valid_images, valid_labels),
                        validation_batch_size=batch_size,
                        callbacks=[checkpoint],
                        verbose=1)

    print("[TRAINING LOG] ====================================================")
    print("[TRAINING LOG] --- END OF TRAINING EPOCH %d----" % (epochs))
    print("[TRAINING LOG] ====================================================") 

    # 保留历史文件
    if history_saved:
        history = h.history
        if resumed_training:
            with open(history_file,'rb') as f:
                h1 = pickle.load(f)
                for i,j in h1.items():
                    history[i].extend(j)

        with open(history_file,'wb') as f:
            pickle.dump(history, f)
        print("[TRAINING LOG] Network training history successfully saved!")

        # 保存最后一次训练的权重文件
        weights_file_final = os.path.join('./weights/',experiment_key+'_model-weights-epoch-'+str(epochs).zfill(2)+'_'+dataName+'_FINAL.hdf5')
        model.save_weights(weights_file_final)
        print("[TRAINING LOG] Network weight file successfully saved!")        