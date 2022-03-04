import torch.nn as nn
import math
import torch
import numpy as np

from torch.backends import cudnn
cudnn.benchmark = False            # if benchmark=True, deterministic will be False
cudnn.deterministic = True

seed = 3456
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def norm_angle(angle):
    norm_angle = sigmoid(10 * (abs(angle) / 0.7853975 - 1))
    return norm_angle


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Temporal_attn(nn.Module):
    def __init__(self,clips,d_model):
        super(Temporal_attn, self).__init__()
        self.attn = nn.Sequential(*[
            nn.Conv1d(d_model,1,1,bias=False),
            nn.Softmax(dim=2)
        ])
        self._init_params()

    def _init_params(self):
        for subnet in [self.attn]:
            for m in subnet.modules():
                self._init_module(m)

    def _init_module(self,m):
        if isinstance(m,nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m,nn.Conv1d):
            nn.init.kaiming_normal_(m.weight,mode='fan_out')
        elif isinstance(m,nn.Linear):
            m.bias.data.fill_(1.0)

    def forward(self, input):
        h = self.attn(input)
        output = torch.squeeze(h,dim=1)
  
        return output


class ResNet_AT(nn.Module):
    def __init__(self, block, layers,clips=7,d_model=512,nhead=4,dropout=0.1, end2end=True,key_clips=1):
        self.inplanes = 64
        self.end2end = end2end
        self.key_clips = key_clips
        super(ResNet_AT, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
    
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        self.attn = nn.MultiheadAttention(512, nhead, dropout=dropout)
        self.norm1 = nn.LayerNorm(512)
        self.dropout2 = nn.Dropout(0.1)
        self.cos = nn.CosineSimilarity(dim=1)
        self.beta = nn.Sequential(nn.Linear(1024, 1),
                                  nn.Sigmoid())
    
        self.maxpooling = nn.AdaptiveMaxPool1d(1)
        self.d_model = d_model
        self.clips = clips
        self.fc = nn.Linear(1024, 512)

        self.temp_attn = Temporal_attn(clips,d_model)
        self.maxpool2 = nn.AdaptiveMaxPool1d(1)

        self.classifier = nn.Sequential(*[nn.Linear(512, 256), nn.ReLU(True),
                                          nn.Linear(256, 128), nn.ReLU(True),
                                         nn.Dropout(0.4),
                                         nn.Sequential(nn.Linear(128, 7))])
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x='', img_num_per_clip=5):

        video_feature = []
        video_beta = []
        for i in range(self.clips):
            vs = []
            ff = x[:, :, :, :, :, i]  
            for j in range(img_num_per_clip):
                f = ff[:, :, :, :, j]  
                f = self.conv1(f)
                f = self.bn1(f)
                f = self.relu(f)
                f = self.maxpool(f)

                f = self.layer1(f)
                f = self.layer2(f)
                f = self.layer3(f)
                f = self.layer4(f)
                f = self.avgpool(f)

                f = f.squeeze(3).squeeze(2)  # f[1, 512, 1, 1] ---> f[1, 512]
                vs.append(f)

            vs_stack = torch.stack(vs, dim=2)
            vs_stack = vs_stack.permute(2,0,1)
            output,weight = self.attn(vs_stack,vs_stack,vs_stack)
            output = vs_stack + self.dropout2(output)
            output = self.norm1(output).permute(1,2,0)
            global_feature = self.maxpooling(output)
            global_feature = global_feature.squeeze(2)
            weights = []
            for i in range(len(vs)):
                vs[i] = torch.cat([vs[i], global_feature], dim=1)
                weights.append(self.beta(self.dropout(vs[i])))
            cascadeVs_stack = torch.stack(vs, dim=2)
            weights_stack = torch.stack(weights, dim=2)
            output = cascadeVs_stack.mul(weights_stack).sum(2).div((weights_stack).sum(2))
            output = self.fc(output)
            video_feature.append(output)
            video_beta.append(weights_stack)
        video_feature = torch.stack(video_feature,dim=1)
        clip_att = self.temp_attn(video_feature.permute(0,2,1))
        max_index = torch.argmax(clip_att, dim=1)
        att_video_feature = (video_feature * clip_att.unsqueeze(dim=2)).permute(0,2,1)
        att_video_feature = self.maxpool2(att_video_feature).squeeze(dim=2)
        clf_output = self.classifier(att_video_feature)

        return clf_output, max_index


def resnet18_AT(**kwargs):
    model = ResNet_AT(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


