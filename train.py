import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

import time
from tensorboardX import SummaryWriter
from datasets import find_dataset_def
from models import *
from utils import *
import sys
import datetime
import ast
from datasets.data_io import *

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='PyTorch Codebase for AA-RMVSNet')
parser.add_argument('--mode', default='train', help='train, val or test')

parser.add_argument('--inverse_depth', help='True or False flag, input should be either "True" or "False".',
    type=ast.literal_eval, default=False)
parser.add_argument('--origin_size', help='True or False flag, input should be either "True" or "False".',
    type=ast.literal_eval, default=False)
parser.add_argument('--save_depth', help='True or False flag, input should be either "True" or "False".',
    type=ast.literal_eval, default=False)

parser.add_argument('--max_h', type=int, default=512, help='Maximum image height when training')
parser.add_argument('--max_w', type=int, default=640, help='Maximum image width when training.')

parser.add_argument('--light_idx', type=int, default=3, help='select while in test')
parser.add_argument('--view_num', type=int, default=3, help='training view num setting')

parser.add_argument('--image_scale', type=float, default=0.25, help='pred depth map scale')

parser.add_argument('--dataset', default='dtu_yao', help='select dataset')
parser.add_argument('--trainpath', help='train datapath')
parser.add_argument('--testpath', help='test datapath')
parser.add_argument('--trainlist', help='train list')
parser.add_argument('--vallist', help='val list')
parser.add_argument('--testlist', help='test list')

parser.add_argument('--epochs', type=int, default=6, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')

parser.add_argument('--batch_size', type=int, default=12, help='train batch size')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
parser.add_argument('--interval_scale', type=float, default=1.06, help='the number of depth values')

parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
parser.add_argument('--logdir', default='./checkpoints/debug', help='the directory to save checkpoints/logs')
parser.add_argument('--save_dir', default=None, help='the directory to save checkpoints/logs')
parser.add_argument('--resume', action='store_true', help='continue to train the model')

parser.add_argument('--summary_freq', type=int, default=20, help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')


# parse arguments and check
args = parser.parse_args()
if args.resume:
    assert args.mode == "train"
    assert args.loadckpt is None
if args.testpath is None:
    args.testpath = args.trainpath

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# create logger
if not os.path.isdir(args.logdir):
    os.mkdir(args.logdir)

current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
print("current time", current_time_str)

print("creating new summary file")
logger = SummaryWriter(args.logdir)

print("argv:", sys.argv[1:])
print_args(args)

SAVE_DEPTH = args.save_depth
if SAVE_DEPTH:
    if args.save_dir is None:
        sub_dir, ckpt_name = os.path.split(args.loadckpt)
        index = ckpt_name[6:-5]
        save_dir = os.path.join(sub_dir, index)
    else:
        save_dir = args.save_dir
    print(os.path.exists(save_dir), ' exists', save_dir)
    if not os.path.exists(save_dir):
        print('save dir', save_dir)
        os.makedirs(save_dir)


MVSDataset = find_dataset_def(args.dataset)
train_dataset = MVSDataset(args.trainpath, args.trainlist, "train", args.view_num, args.numdepth, args.interval_scale, args.inverse_depth, args.origin_size, -1, args.image_scale) # Training with False, Test with inverse_depth
#val_dataset = MVSDataset(args.trainpath, args.vallist, "val", 5, args.numdepth, args.interval_scale, args.inverse_depth, args.origin_size, args.light_idx, args.image_scale) #view_num = 5, light_idx = 3
test_dataset = MVSDataset(args.testpath, args.testlist, "test", 5, args.numdepth, args.interval_scale, args.inverse_depth, args.origin_size, args.light_idx, args.image_scale) # use 3
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=12, drop_last=True)
#ValImgLoader = DataLoader(val_dataset, args.batch_size, shuffle=False, num_workers=4, drop_last=False)
TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4, drop_last=False)
# Use test set (with gt depths) for validation


print('model: AA-RMVSNet')
model = AARMVSNet(image_scale=args.image_scale, max_h=args.max_h, max_w=args.max_w)
model = model.cuda()
model = nn.parallel.DataParallel(model)

print('loss: Cross Entropy')
model_loss = mvsnet_cls_loss

print('optimizer: Adam \n')
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# load parameters
start_epoch = 0
if (args.mode == "train" and args.resume):
    saved_models = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.logdir, saved_models[-1])
    print("resuming from:", loadckpt)
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'])
    optimizer.load_state_dict(state_dict['optimizer'])
    print(optimizer)

    start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)
    model.load_state_dict(state_dict['model'])
print("start at epoch {}".format(start_epoch))

# main function
def train():
    print('run train()')
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=2e-06)
    ## get intermediate learning rate
    for _ in range(start_epoch):
        lr_scheduler.step()
    for epoch_idx in range(start_epoch, args.epochs):
        
        print('Epoch {}/{}:'.format(epoch_idx, args.epochs))

        lr_scheduler.step()
        global_step = len(TrainImgLoader) * epoch_idx
        print('Start Training')
        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = train_sample(sample, detailed_summary=do_summary)
            
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
            
            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
                logger.add_scalar('train/lr', lr, global_step)
                save_images(logger, 'train', image_outputs, global_step)
            del scalar_outputs, image_outputs
            print(
                'Epoch {}/{}, Iter {}/{}, LR {}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs, batch_idx,
                                                                                     len(TrainImgLoader), lr, loss,
                                                                                     time.time() - start_time))

        # checkpoint
        if (epoch_idx + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()},
                "{}/model_{:0>6}.ckpt".format(args.logdir, epoch_idx))

    
        avg_test_scalars = DictAverageMeter()
        for batch_idx, sample in enumerate(TestImgLoader):
            start_time = time.time()
            global_step = len(TestImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = test_sample(sample, detailed_summary=do_summary)
            if do_summary:
                save_scalars(logger, 'test', scalar_outputs, global_step)
                save_images(logger, 'test', image_outputs, global_step)
            avg_test_scalars.update(scalar_outputs)
            #del scalar_outputs, image_outputs
            del image_outputs
            
            print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}, ame = {:3f}, thres2mm = {:3f}, thres4mm = {:3f}, thres8mm = {:3f}'.format(
                                epoch_idx, args.epochs, batch_idx,
                                len(TestImgLoader), loss,
                                time.time() - start_time,
                                scalar_outputs["abs_depth_error"], scalar_outputs["thres2mm_error"], 
                                scalar_outputs["thres4mm_error"], scalar_outputs["thres8mm_error"]))
        save_scalars(logger, 'fulltest', avg_test_scalars.mean(), global_step)
        print("avg_test_scalars:", avg_test_scalars.mean())


def train_sample(sample, detailed_summary=False):
    model.train()
    optimizer.zero_grad()

    sample_cuda = tocuda(sample)
    depth_gt = sample_cuda["depth"]
    mask = sample_cuda["mask"]
    depth_interval = sample_cuda["depth_interval"]
    depth_value = sample_cuda["depth_values"]
    outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])

    prob_volume = outputs['prob_volume']
    loss, depth_est = model_loss(prob_volume, depth_gt, mask, depth_value)

    loss.backward()
    optimizer.step()
    scalar_outputs = {"loss": loss}
    image_outputs = {"depth_est": depth_est * mask, "depth_gt": sample["depth"],
                     "ref_img": sample["imgs"][:, 0],
                     "mask": sample["mask"]}
    if detailed_summary:
        image_outputs["errormap"] = (depth_est - depth_gt).abs() * mask
        scalar_outputs["abs_depth_error"] = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5)
        scalar_outputs["thres2mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 2)
        scalar_outputs["thres4mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 4)
        scalar_outputs["thres8mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 8)

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


@make_nograd_func
def test_sample(sample, detailed_summary=True):
    model.eval()
    sample_cuda = tocuda(sample)
    depth_gt = sample_cuda["depth"]
    mask = sample_cuda["mask"]
    depth_interval = sample_cuda["depth_interval"]
    depth_value = sample_cuda["depth_values"]
    outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])

    prob_volume = outputs['prob_volume']
    loss, depth_est, photometric_confidence = model_loss(prob_volume, depth_gt, mask, depth_value, return_prob_map=True)

    scalar_outputs = {"loss": loss}
    image_outputs = {"depth_est": depth_est * mask,
                     "photometric_confidence": photometric_confidence * mask, 
                     "depth_gt": sample["depth"],
                     "ref_img": sample["imgs"][:, 0],
                     "mask": sample["mask"]}

    if detailed_summary:
        image_outputs["errormap"] = (depth_est - depth_gt).abs() * mask
        
    scalar_outputs["abs_depth_error"] = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5)
    scalar_outputs["thres2mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 2)
    scalar_outputs["thres4mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 4)
    scalar_outputs["thres8mm_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, 8)

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs



if __name__ == '__main__':
    train()