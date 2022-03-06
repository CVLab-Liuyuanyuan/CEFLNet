# CFELNet.pytorch

## The code of Clip-aware Expressive Feature Learning for Video-based Facial Expression Recognition.

## Requirement
torch==1.7.1

tensorboardX==2.4.1

pytorch-warmup==0.0.4

torchvision==0.8.2

numpy==1.19.5

Pillow==8.3.2


## Options
* ``` --lr ```: initial learning rate
* ``` --epochs ```: number of total epochs to run
* ``` --momentum ```: momentum
* ``` --weight-decay ```: weight decay (default: 1e-4)
* ``` --train_video_root ```: path to train videos
* ``` --train_list_root ```: path to train videos list
* ``` --test_video_root ```: path to test videos
* ``` --test_list_root ```: path to test videos list
* ``` --batch_size ```: input batch size
* etc.

## pre-trained model and Validated model
You can get the pre-trained model and validated model from https://pan.baidu.com/s/1xYCvyVOTfxZAz1b0C76iuQ (Extraction Code：6q72) 

To facilitate validation, we provide the test model on the AFEW dataset：https://pan.baidu.com/s/1Cm9pTiBs5-fivDZP8mLBDA (Extraction Code：8lq5) 


## Dataset
You can get AFEW dataset at https://sites.google.com/site/emotiwchallenge/

