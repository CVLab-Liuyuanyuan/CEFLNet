import os
import torch
import time


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)  # first position is score; second position is pred.
    pred = pred.t()  # .t() is T of matrix (256 * 1) -> (1 * 256)
    correct = pred.eq(target.view(1, -1).expand_as(pred))  # target.view(1,2,2,-1): (256,) -> (1, 2, 2, 64)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))

    return res


def adjust_learning_rate(optimizer, epoch, learning_rate, end_epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch in [20,30,50,54,55,60,65,70,75,80,85,90,100]:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.2

        learning_rate = learning_rate* 0.2
        print('Adjust_learning_rate ' + str(epoch))
        print('New_LearningRate: {}'.format(learning_rate))
    return learning_rate


def save_checkpoint(state,name=None):

    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    if not name is None:
        path = os.path.join('./checkpoints',name)
        if not os.path.exists(path):
            os.mkdir(path)
    else:
        path = os.path.join('./checkpoints')
        if not os.path.exists(path):
            os.mkdir(path)

    epoch = state['epoch']
    save_dir = os.path.join(path,str(epoch)+'_'+str(round(float(state['prec1']), 4)))
    torch.save(state, save_dir)
    print(save_dir)


def save_param(args):
    name = args.name
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    if not name is None:
        path = os.path.join('./checkpoints',name)
        if not os.path.exists(path):
            os.mkdir(path)
    else:
        path = os.path.join('./checkpoints')
        if not os.path.exists(path):
            os.mkdir(path)

    params = str(args)
    str_time = get_time()
    file_name = os.path.join(path,str_time + '_param.txt')

    f = open(file_name,'w')
    f.write(params)
    f.close()

def get_time():
    return str(time.strftime("%Y_%m_%d_%H_%M_%S",time.localtime()))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

