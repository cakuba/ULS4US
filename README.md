[license]: https://github.com/cakuba/ULS4US/blob/main/LICENSE
[weight_file]: https://pan.baidu.com/s/1DsRNngvP3uup_6a4knDr8g

# ULS4US
an End-to-End Universal Lesion Segmentation Framework for 2D Ultrasound Images

## Diagram of ULS4US framework
ULS4US is composed of a multiple-in multiple-out (MIMO) UNet integrating multiscale features from both full and cropped partial images, a two-stage lesion-aware learning algorithm recursively locating and segmenting the lesions in a reinforced way, and an lesion-adaptive loss function for the MIMO-UNet consisting of two weighted and one self-supervised loss components designed for intra- and inter-branches of network outputs, respectively.

The diagram and workflows of ULS4US framework is shown as below
![image](https://user-images.githubusercontent.com/1317555/185050939-9177eb9b-7f5b-44df-be2f-8aad8e00aa5d.png)

In summary, we break the task of ULS for US images into two stages: (Stage 1) to detect the presence of the lesion and roughly delineate its outline in the original image (i.e., to treat the lesion segmentation as a small-scale object segmentation problem), and then, (Stage 2) to crop the original image to contain the lesion only and stay focus on the cropped partial image in order to obtain the accurate lesion boundary (i.e., to treat the lesion segmentation as normal or even large size object segmentation problem)

## MIMO-UNet architecture in ULS4US
We re-design the conventional UNet architecture to implement a new multiscale feature fusion network as MIMO-UNet, shown as below
![image](https://user-images.githubusercontent.com/1317555/185051697-d68417f9-083c-4635-9d1e-e6fa0019c66c.png)

Main modifications to UNet include
 - an additional input branch (IB), along with an additional output branch (OB), is added. The input image size of this additional IB is 1/4 of the original IB; i.e., the number of pixels in both horizontal and vertical directions is 1/2 of the original image.
 - two input images for dual IBs are fed separately into two encoders, which extract the features from the input image and convert them into a 32x32 feature map with 512 dimensions
 - besides the up-sampled feature maps, the decoder performs a concatenate operation on feature maps of the same size from both encoders via skip-connections.
 - a customized layer to compute the network loss is appended as the third output branch

## Demonstration of segmentation results
The performance of ULS4US is assessed in a unified dataset consisting of two public and three private US image datasets which involve over 2300 images and three specific types of organs, and comparative experiments on the individual and unified datasets suggest that ULS4US is likely scalable with more data. The trained network weight can be downloaded from [[Baidu NetDisk](https://pan.baidu.com/s/1DsRNngvP3uup_6a4knDr8g) with the access code 'ccj1'].

![image](https://user-images.githubusercontent.com/1317555/185053461-b84e1340-7bf1-4ab9-9ca9-2b45780d076d.png)

## How to use the ULS4US?

0. fully tested with Ubuntu 18.04 LTS, Python 3.6.9 and Keras 2.4.0 with Tensorflow 2.4.1 as the backend in a server equipped with Nvidia GTX 3090 GPUs

1. clone the repo to local directory 
```Bash
   git clone https://github.com/cakuba/ULS4US.git
```
2.  prepare the training and test dataset
- go to the directory `data` and create a new sub-directory as you like; e.g., `breastUS`
- enter `breastUS` directory and arrange the structure for the training and test set as 

           breastUS
                 |
                 -- training
                           | 
                           -- img
                                |
                                -- 000.png
                                -- 001.png
                           -- mask
                                |
                                -- 000.png
                                -- 001.png                                  
                 |         
                 -- test
                       |
                       -- img
                            |
                            -- 000.png
                            -- 001.png
                       -- mask
                            |
                            -- 000.png
                            -- 001.png                          
                        
- go to the directory `conf` and add the `breastUS` data information into the file `dataset.conf` as

           [breastUS]
           training_data_dir = ./data/breastUS/training
           test_data_dir = ./data/breastUS/test
           data_name = breast
  
- in the same directory, update the file `training.conf` and replace the value of the key 'dataset' as `breast` under the section **[ULS4US]** (NOTE: this corresponds to the same value of the key 'data_name' in the file `dataset.conf` 

4. start training
```Bash
   python ULS4US.py
```

Most of the training hyperparameters can be changed in the file `training.conf` under the section **[ULS4US]**

5. performance evaluation of ULS4US
```Bash
   python evaluate_performance_ULS4US.py
```  
you should observe some outputs as below

![image](https://user-images.githubusercontent.com/1317555/185055464-ca3f2675-52d6-4071-a082-83f2b0575125.png)

6. predictions of ULS4US
```Bash
   python prediction_ULS4US.py
```
you should find the predictions for the sample test data saved as PNG files in the sub-directory 'pred'. 

And congratulations, you have just used ULS4US for your own data! Also, we have prepared some sample test data in the directory `data/all_mixed` and provided the predictions of ULS4US for these test data in the directory `pred`. Please feel free to let us know if you have any questions.

## Who are we?

ULS4US is proposed and maintained by researchers from <a href="https://www.wit.edu.cn/" target="_blank">WIT</a>.

## License

See [LICENSE][license] for ULS4US
