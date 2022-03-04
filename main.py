from __future__ import print_function
import argparse
import time
import torch
import os
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from data import load_materials
import util
from models import Model
from tensorboardX import SummaryWriter
import pytorch_warmup as warmup
import random

from torch.backends import cudnn
cudnn.benchmark = False            # if benchmark=True, deterministic will be False
cudnn.deterministic = True

seed = 3456
torch.manual_seed(seed)            
torch.cuda.manual_seed(seed)       
torch.cuda.manual_seed_all(seed)  
random.seed(seed)


parser = argparse.ArgumentParser(description='PyTorch EFSLNet')
parser.add_argument('--train_video_root', required=True, help='path to train videos')
parser.add_argument('--train_list_root', required=True, help='path to train videos list')
parser.add_argument('--test_video_root', required=True, help='path to test videos')
parser.add_argument('--test_list_root', required=True, help='path to test videos list')
parser.add_argument('--batch_size',type=int, default=8,help='input batch size')
parser.add_argument('--parameterDir',type=str, default='./pretrain_para/Resnet18_pytorch.pth.tar')
parser.add_argument('--gpu_ids', type=str, default='0',help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--eval_model_path', type=str)
parser.add_argument('--warm_up', default=10, type=int)
parser.add_argument('--epochs', default=120, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=1e-04, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum　(default: 0.9)')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=20, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('-e', '--evaluate', default=False, dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')

best_prec = 0
args = parser.parse_args()
writer = SummaryWriter(comment=args.name)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids


def main():
    global args, best_prec
    util.save_param(args)

    ''' Load data '''
    arg_rootTrain = args.train_video_root
    arg_listTrain = args.train_list_root

    arg_rooteval = args.test_video_root
    arg_listeval = args.test_list_root

    batch_size = args.batch_size
    
    # TODO 需要修改
    train_loader, val_loader = load_materials.LoadNewDataset(arg_rootTrain, arg_listTrain, batch_size, arg_rooteval, arg_listeval)

    ''' Load model '''
    _structure = Model.resnet18_AT()
    _parameterDir = args.parameterDir

    if args.evaluate:
        _parameterDir = args.eval_model_path
        model = torch.nn.DataParallel(_structure).cuda()
        model.load_state_dict(torch.load(_parameterDir)['state_dict'])
    else:
        model = load_materials.LoadParameter(_structure, _parameterDir)

    ''' Loss & Optimizer '''
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=(0.9, 0.999),
                                 weight_decay=args.weight_decay)

    ''' Train & Eval '''
    if args.evaluate:
        validate(val_loader, model)
        return
    print('args.lr: ', args.lr)
    print('batch_size: ', batch_size)

    lr_schduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=160)
    warmup_scheduler = warmup.LinearWarmup(optimizer, warmup_period=args.warm_up)
    warmup_scheduler.last_step = -1
    for epoch in range(0, args.epochs):
        lr_schduler.step(epoch)
        warmup_scheduler.dampen()
        print(epoch, optimizer.param_groups[0]['lr'])
        train(train_loader, model, criterion, optimizer, epoch)
        prec1 = validate(val_loader, model)

        writer.add_scalar('test_acc', prec1, epoch)

        is_best = prec1 >= best_prec
        if is_best:
            print('better model!')
            best_prec = max(prec1, best_prec)
            util.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'prec1': prec1,
            }, args.name)
        else:
            print('Model too bad & not save')


def train(train_loader, model, criterion, optimizer, epoch):

    batch_time = util.AverageMeter()
    running_loss, count, correct_count = 0., 0, 0
    model.train()
    end = time.time()

    for i, data in enumerate(train_loader):
        target_first = data['label']
        input_var = torch.autograd.Variable(data['data'])
        target = target_first.cuda(non_blocking=True)
        target_var = torch.autograd.Variable(target)
        pred_score, _ = model(input_var)

        # compute gradient and do SGD step
        loss = criterion(pred_score, target_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        correct_count += (torch.max(pred_score, dim=1)[1] == target_var).sum()
      
        count += input_var.size(0)

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t Loss {loss:.4f}\t Cls_Acc {acc:.4f}\t'
                  .format(epoch, i, len(train_loader), loss=running_loss/count, acc=int(correct_count)/count))

        batch_time.update(time.time() - end)
        end = time.time()
    print(' Train_Acc {train_acc:.4f}\t  Train_Loss {train_Loss:.4f}\t  '.
          format(train_acc=int(correct_count)/count, train_Loss=running_loss/count))

    writer.add_scalar('loss', running_loss / count, epoch)
    writer.add_scalar('cls_acc', int(correct_count) / count, epoch)
  

def validate(val_loader, model):
    model.eval()
    test_correct_count, test_count,= 0, 0
    max_loc = []
    with torch.no_grad():
        for i,data in enumerate(val_loader):
            input_var = torch.autograd.Variable(data['data'])
            target_first = data['label']
            target = target_first.cuda(non_blocking=True)
            target_var = torch.autograd.Variable(target)
            pred_score,loc = model(input_var)
            test_correct_count += (torch.max(pred_score, dim=1)[1] == target_var).sum()
            test_count += input_var.size(0)
        test_acc = int(test_correct_count)/test_count
        print(' Test_Acc: {test_Video:.4f} '.format(test_Video=test_acc))
        #print(' max location: ', max_loc)

        return test_acc


if __name__ == '__main__':
    main()
    writer.close()

