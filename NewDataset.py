import os
import random
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np

seed = 3456
torch.manual_seed(seed)            # 为CPU设置随机种子
torch.cuda.manual_seed(seed)       # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)   # 为所有GPU设置随机种子
random.seed(seed)
np.random.seed(seed)

class NewDataset(data.Dataset):
    def __init__(self,video_root,video_list,isTrain,rectify_label,transform=None):
        super(NewDataset, self).__init__()
        ### param area####################
        self.first_clips_number = 30  
        self.search_clips_number = 75 
        self.choose_num = 5 
        self.each_clips=15 
        self.clips = 7 
        self.next_jump = 10 
        self.isTrain = isTrain
        self.test_first = 15

        #####data path ###################
        self.video_root = video_root
        self.video_list = video_list
        self.rectify_label = rectify_label
        self.transform = transform
        #############################

        self.video_label = self.read_data(self.video_root,self.video_list,self.rectify_label)


    def load_per(self,path):
        return np.load(path)

    def read_data(self,video_root,video_list,rectify_label):
        video_label_list = []
        # 读取文件，获取所有视频数据
        with open(video_list,'r') as imf:

            for id, line in enumerate(imf):
                video_label = line.strip().split()
                video_name = video_label[0]
                label = rectify_label[video_label[1]]

                video_label_list.append((os.path.join(video_root,video_name),label))

        return video_label_list

    def __getitem__(self, index):

        data_path,label = self.video_label[index]

        frame_path_list = sorted(os.listdir(data_path))

        if self.isTrain:
            first_loc = random.randint(0, self.first_clips_number - 1)
        else:
            first_loc = self.test_first
        # 从first_loc开始往后选each_clips+search_clips_number张图片，each_clips为第一个clip的张数，后面的search_clips_number为后面的clip所需要的图片数
        sub_frames_list = frame_path_list[first_loc:first_loc + self.search_clips_number]

        data_clips = []

        # clip 0~clips-1
        cur_loc = 0
        for i in range(0, self.clips):
            high_range = cur_loc + self.each_clips
            low_range = cur_loc
            frames_tmp = sub_frames_list[low_range:high_range]
            data_clips.append(self.get_image(frames_tmp, data_path))

            cur_loc += self.next_jump

        data_clips = self.order_clip(data_clips,[0,1,2,3,4,5,6])

        return {'data':data_clips,'label':label,'path':data_path}

    def order_clip(self,data_clips,order):
        clip_0 = torch.stack(data_clips[order[0]], dim=3)
        clip_1 = torch.stack(data_clips[order[1]], dim=3)
        clip_2 = torch.stack(data_clips[order[2]], dim=3)
        clip_3 = torch.stack(data_clips[order[3]], dim=3)
        clip_4 = torch.stack(data_clips[order[4]], dim=3)
        clip_5 = torch.stack(data_clips[order[5]], dim=3)
        clip_6 = torch.stack(data_clips[order[6]], dim=3)

        data_clips_order = torch.stack([clip_0, clip_1, clip_2, clip_3, clip_4, clip_5, clip_6], dim=4)
        return data_clips_order

    # 读取每个clip的图片数据
    def get_image(self,frames,data_path):
        # 随机采样 choose_num 个图片id
        if self.isTrain:
            indexs = self.sample(self.choose_num,0,self.each_clips-1)
        else:
            indexs = [x for x in range(0,self.each_clips,self.each_clips // self.choose_num)]
        # 读取图片
        result_list = []
        for loc in indexs:
            img_path = os.path.join(data_path,frames[loc])
            img = Image.open(img_path).convert("RGB")
            img = img.resize((224,224))
            if self.transform is not None:
                img = self.transform(img)
            result_list.append(img)

        return result_list

    def sample(self,num,min_index,max_index):
        s = set()
        while len(s) < num:
            tmp = random.randint(min_index,max_index)
            s.add(tmp)
        return list(s)

    def __len__(self):
        return len(self.video_label)

