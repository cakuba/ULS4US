#
#  Univeral Segmentation Framework for Lesions in Ultrasound Images
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
from models import unet
from utils import getData
from utils import printConfig
from configparser import ConfigParser

cfg_file = "./conf/training.conf"
data_file = "./conf/dataset.conf"

if __name__ == "__main__":
    # parameter settings
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

    # model training parameters
    width = int(paras['width'])
    height = int(paras['width'])
    img_size = (width, height)
    crop_width = int(paras['crop_width'])
    crop_height = int(paras['crop_height'])
    preprocessing = paras['preprocessing']=='True'
    seed = int(paras['seed'])
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

    # nums_of_relocation settings
    if (nums_of_relocation<2):
        print("[WARNING] The hyperparameter NUM_OF_RELOCATION should be larger than 1!")

    # training/validation dataset
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

    # load training/validation data
    images, labels = getData.get_data_orig(data_dir,size=(width, height),preprocessing=preprocessing)
    train_images, valid_images, train_labels, valid_labels = train_test_split(images, labels, test_size=0.18, random_state=seed)
    train_images = np.expand_dims(train_images, axis=-1)
    train_labels = np.expand_dims(train_labels, axis=-1)
    valid_images = np.expand_dims(valid_images, axis=-1)
    valid_labels = np.expand_dims(valid_labels, axis=-1)
    nums_of_train_images = len(train_images)
    nums_of_valid_images = len(valid_images)
    print(train_images.shape,train_labels.shape,valid_images.shape,valid_labels.shape)    
    print("[TRAINING LOG] A total number of %d training and %d valid images fed into the network training..." % (nums_of_train_images,nums_of_valid_images))

    # model weight file / training history file
    weights_file = os.path.join('./weights/',experiment_key+'_model-weights-epoch-'+str(epochs).zfill(2)+'-lossmode-'+str(loss_mode)+'_'+dataName+'.hdf5')
    last_weights_file = os.path.join('./weights/',experiment_key+'_model-weights-epoch-'+str(epochs).zfill(2)+'-lossmode-'+str(loss_mode)+'_'+dataName+'_FINAL.hdf5')
    history_file = os.path.join('./history', experiment_key+'_history-epoch-'+str(epochs)+'-lossmode-'+str(loss_mode)+'_'+dataName+'.dat')

    # data augmentation
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

    # Train and validate the model
    model = unet.UNet_mimo(img1_width=width,img1_height=height,img2_width=crop_width,img2_height=crop_height,
                                      loss_mode=loss_mode,beta1=beta1,beta2=beta2)
    model.compile(optimizer=Adam(lr=lr),loss=[None]*len(model.outputs))
    checkpoint = ModelCheckpoint(weights_file,monitor='val_loss',verbose=1, mode='min', save_best_only=True)

    if resumed_training:
        model.load_weights(last_weights_file)
        print("[TRAINING LOG] The model is trained from a previous checkpoint... \n")

    # start training
    history = []
    valid_history = []
    metrics_names = model.metrics_names
    metrics_names.insert(0,'loss')
    steps_per_epoch = int(nums_of_train_images/batch_size)+1
    valid_img1 = valid_images
    valid_img1_target = valid_labels
    valid_loss = 1.e+5
    for e in range(epochs):
        step = 0
        #for step in range(steps_per_epoch):
        for data in train_generator:
            # input images and annotation for the first branch
            img1 = data[0]
            img1_target = data[1]
            # random crop
            img2, img2_target = getData.randomCrop(img1,img1_target,width=crop_width,height=crop_height)
            output1_cropped = img2_target

            # training for the batch data
            for num in range(nums_of_relocation):
                print("--- Epoch %d step %d inner loop %d----" % (e, step, num))
                h = model.fit([img1,img1_target,img2,img2_target,output1_cropped],verbose=2)
                history.append(h.history)

                # inference
                img1_output, _, _ = model.predict([img1,img1_target,img2,img2_target,output1_cropped])
                # update the input image and annotation for the second branch
                img2, img2_target, output1_cropped = getData.roiLocator(img1,img1_target,img1_output,
                                                             width=crop_width,height=crop_height,paddingRatio=padding_ratio)            

            print("=========================================")
            print("======== end of inner loops =============")

            step = step+1
            if (step>steps_per_epoch):
                break

        # performance evaluation in the validation set
        valid_img2, valid_img2_target = getData.randomCrop(valid_img1,valid_img1_target,width=crop_width,height=crop_height)
        valid_output1_cropped = valid_img2_target
        valid_img1_output, _, _ = model.predict([valid_img1,valid_img1_target,valid_img2,valid_img2_target,valid_output1_cropped])
        valid_img2, valid_img2_target, valid_output1_cropped = getData.roiLocator(valid_img1,valid_img1_target,valid_img1_output,
                                                                width=crop_width,height=crop_height,paddingRatio=padding_ratio)        
        valid_eval = model.evaluate([valid_img1,valid_img1_target,valid_img2,valid_img2_target,valid_output1_cropped])
        valid_history.append(dict(zip(metrics_names, valid_eval)))
        if (valid_eval[0] < valid_loss):
            print("[TRAINING LOG] Epoch {:0>5d}: val_loss improved from {:.5f} to {:.5f}, saving model to {}".format(e,valid_loss,valid_eval[0],weights_file))
            valid_loss = valid_eval[0]
            model.save_weights(weights_file)
        else:
            print("[TRAINING LOG] Epoch {:0>5d}: val_loss did not improve from {:.5f}".format(e,valid_loss))


        print("[TRAINING LOG] ====================================================")
        print("[TRAINING LOG] --- END OF TRAINING EPOCH %d----" % (e))
        print("[TRAINING LOG] ====================================================") 

    if history_saved:
        # the weight file saved for the resumed training
        weights_file_final = os.path.join('./weights/',experiment_key+'_model-weights-epoch-'+str(epochs).zfill(2)+'-lossmode-'+str(loss_mode)+'_'+dataName+'_FINAL.hdf5')
        model.save_weights(weights_file_final)
        print("[TRAINING LOG] Network weight file successfully saved!")

        if resumed_training:
            with open(history_file,'rb') as f:
                h1, h2 = pickle.load(f)
                history.extend(h1)
                valid_history.extend(h2)    
        
        with open(history_file,'wb') as f:
            pickle.dump([history,valid_history], f)
        print("[TRAINING LOG] Network training history successfully saved!")