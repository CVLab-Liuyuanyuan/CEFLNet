from __future__ import print_function
import torch
import torch.utils.data
import torchvision.transforms as transforms
from data import NewDataset

seed = 3456
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

cate2label = {'MMI':{0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise',
                     'Angry': 0,'Disgust': 1,'Fear': 2,'Happy': 3,'Sad': 4,'Surprise': 5},
                'BU3D':{0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise',
                     'Angry': 0,'Disgust': 1,'Fear': 2,'Happy': 3,'Sad': 4,'Surprise': 5},
              'AFEW':{0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise',
                  'Angry': 0,'Disgust': 1,'Fear': 2,'Happy': 3,'Neutral': 4,'Sad': 5,'Surprise': 6},
              'DFEW':{0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise',
                        'Angry': 0,'Disgust': 1,'Fear': 2,'Happy': 3,'Neutral': 4,'Sad': 5,'Surprise': 6}
              }

cate2label = cate2label['AFEW']


def LoadNewDataset(root_train, list_train, batchsize, root_eval, list_eval):

    train_dataset = NewDataset.NewDataset(
        video_root=root_train,
        video_list=list_train,
        rectify_label=cate2label,
        isTrain= True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )


    val_dataset = NewDataset.NewDataset(
        video_root=root_eval,
        video_list=list_eval,
        rectify_label=cate2label,
        isTrain = False,
        transform=transforms.Compose([transforms.ToTensor()]),
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batchsize, shuffle=True,num_workers=4,
         pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batchsize, shuffle=False,num_workers=4,
         pin_memory=True)

    return train_loader, val_loader


def LoadParameter(_structure, _parameterDir):
    checkpoint = torch.load(_parameterDir)
    pretrained_state_dict = checkpoint['state_dict']
    model_state_dict = _structure.state_dict()

    for key in pretrained_state_dict:
        if ((key == 'module.fc.weight') | (key == 'module.fc.bias') | (key == 'module.feature.weight') | (key == 'module.feature.bias')):

            pass
        else:
            model_state_dict[key.replace('module.', '')] = pretrained_state_dict[key]

    _structure.load_state_dict(model_state_dict)
    model = torch.nn.DataParallel(_structure).cuda()

    return model
