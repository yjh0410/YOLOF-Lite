from __future__ import division

import os
import argparse
import time
import random

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data.voc import VOCDetection
from data.coco import COCODataset
from config.yolof_config import yolof_config
from data.transforms import TrainTransforms, ValTransforms, BaseTransforms

from utils import distributed_utils
from utils.criterion import build_criterion
from utils.com_flops_params import FLOPs_and_Params
from utils.misc import CollateFunc, ModelEMA, get_total_grad_norm
from utils.solver.optimizer import build_optimizer
from utils.solver.warmup_schedule import build_warmup

from evaluator.coco_evaluator import COCOAPIEvaluator
from evaluator.voc_evaluator import VOCAPIEvaluator

from models.yolof import build_model


def parse_args():
    parser = argparse.ArgumentParser(description='YOLOf Detection')
    # basic
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='use cuda.')
    parser.add_argument('--batch_size', default=16, type=int, 
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--start_epoch', type=int, default=0,
                        help='start epoch to train')
    parser.add_argument('--max_epoch', type=int, default=12,
                        help='max epoch to train')
    parser.add_argument('--lr_epoch', nargs='+', default=[8, 10], type=int,
                        help='lr epoch to decay')
    parser.add_argument('--num_workers', default=4, type=int, 
                        help='Number of workers used in dataloading')
    parser.add_argument('--num_gpu', default=1, type=int, 
                        help='Number of GPUs to train')
    parser.add_argument('--eval_epoch', type=int,
                            default=2, help='interval between evaluations')
    parser.add_argument('--tfboard', action='store_true', default=False,
                        help='use tensorboard')
    parser.add_argument('--save_folder', default='weights/', type=str, 
                        help='path to save weight')

    # input image size               
    parser.add_argument('--img_size', type=int, default=640,
                        help='The size of input image')

    # optimizer
    parser.add_argument('-opt', '--optimizer', default='sgd', type=str, 
                        help='optimizer: sgd, adamw')
    parser.add_argument('--grad_clip_norm', type=float, default=-1.,
                        help='grad clip.')

    # visualize
    parser.add_argument('--vis_data', action='store_true', default=False,
                        help='visualize input data.')
    parser.add_argument('--vis_targets', action='store_true', default=False,
                        help='visualize the targets.')

    # model
    parser.add_argument('-v', '--version', default='yolof50', choices=['yolof18', 'yolof50', 'yolof50-DC5', \
                                                                       'yolof101', 'yolof101-DC5', 'yolof53', 'yolof53-DC5', \
                                                                       'yoloft-DC5', 'yolofs-DC5', 'yolofb-DC5', 'yolofl-DC5', 'yolofx-DC5'],
                        help='build yolof')
    parser.add_argument('--conf_thresh', default=0.05, type=float,
                        help='NMS threshold')
    parser.add_argument('--nms_thresh', default=0.6, type=float,
                        help='NMS threshold')
    parser.add_argument('--topk', default=1000, type=int,
                        help='NMS threshold')
    parser.add_argument('-p', '--coco_pretrained', default=None, type=str,
                        help='coco pretrained weight')

    # dataset
    parser.add_argument('--root', default='/mnt/share/ssd2/dataset',
                        help='data root')
    parser.add_argument('-d', '--dataset', default='coco',
                        help='coco, voc, widerface, crowdhuman')
    
    # Matcher
    parser.add_argument('-ls', '--label_assignment', default='static', type=str,
                        help='label assignment method')

    # Loss
    parser.add_argument('--loss_cls_weight', default=1.0, type=float,
                        help='weight of cls loss')
    parser.add_argument('--loss_reg_weight', default=1.0, type=float,
                        help='weight of reg loss')

    # train trick
    parser.add_argument('--ema', action='store_true', default=False,
                        help='EMA')
    parser.add_argument('-ms', '--multi_scale', action='store_true', default=False,
                        help='Multi scale augmentation')
    parser.add_argument('--mosaic', action='store_true', default=False,
                        help='Mosaic augmentation')
    parser.add_argument('--no_warmup', action='store_true', default=False,
                        help='do not use warmup')
    parser.add_argument('--wp_iter', type=int, default=500,
                        help='The upper bound of warm-up')

    # DDP train
    parser.add_argument('-dist', '--distributed', action='store_true', default=False,
                        help='distributed training')
    parser.add_argument('--dist_url', default='env://', 
                        help='url used to set up distributed training')
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--sybn', action='store_true', default=False, 
                        help='use sybn.')

    return parser.parse_args()


def train():
    args = parse_args()
    print("Setting Arguments.. : ", args)
    print("----------------------------------------------------------")

    # dist
    if args.distributed:
        distributed_utils.init_distributed_mode(args)
        print("git:\n  {}\n".format(distributed_utils.get_sha()))

    # path to save model
    path_to_save = os.path.join(args.save_folder, args.dataset, args.version)
    os.makedirs(path_to_save, exist_ok=True)

    # cuda
    if args.cuda:
        print('use cuda')
        cudnn.benchmark = True
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # YOLOF Config
    cfg = yolof_config[args.version]
    print('Model Configuration: \n', cfg)

    # multi scale trick
    train_size = args.img_size
    val_size = args.img_size
    scale_jitter = None
    if args.multi_scale:
        print('Multi scale training ...')
        scale_jitter = cfg['scale_jitter']

    # dataset and evaluator
    dataset, evaluator, num_classes = build_dataset(cfg, args, device)

    # dataloader
    dataloader = build_dataloader(args, dataset, CollateFunc())

    # criterion
    criterion = build_criterion(args=args, device=device, cfg=cfg, num_classes=num_classes)
    
    # build model
    net = build_model(args=args, 
                      cfg=cfg,
                      device=device, 
                      num_classes=num_classes, 
                      trainable=True,
                      coco_pretrained=args.coco_pretrained)
    model = net
    model = model.to(device).train()

    model_without_ddp = model
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    # compute FLOPs and Params
    if distributed_utils.is_main_process:
        model_without_ddp.trainable = False
        model_without_ddp.eval()
        FLOPs_and_Params(model=model_without_ddp, 
                         img_size=args.img_size, 
                         device=device)
        model_without_ddp.trainable = True
        model_without_ddp.train()
        
    # EMA
    ema = ModelEMA(model_without_ddp) if args.ema else None

    # optimizer
    optimizer = build_optimizer(model=model_without_ddp,
                                base_lr=args.lr,
                                name=args.optimizer,
                                momentum=cfg['momentum'],
                                weight_decay=cfg['weight_decay'])
    
    # lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_epoch)

    # warmup scheduler
    warmup_scheduler = build_warmup(name=cfg['warmup'],
                                    base_lr=args.lr,
                                    wp_iter=args.wp_iter,
                                    warmup_factor=cfg['warmup_factor'])

    # training configuration
    max_epoch = args.max_epoch
    batch_size = args.batch_size
    epoch_size = len(dataset) // (batch_size * args.num_gpu)
    best_map = -1.
    warmup = not args.no_warmup

    t0 = time.time()
    # start training loop
    for epoch in range(args.start_epoch, max_epoch):
        if args.distributed:
            dataloader.sampler.set_epoch(epoch)            

        # train one epoch
        for iter_i, (images, targets) in enumerate(dataloader):
            ni = iter_i + epoch * epoch_size
            # warmup
            if ni < args.wp_iter and warmup:
                warmup_scheduler.warmup(ni, optimizer)

            elif ni == args.wp_iter and warmup:
                # warmup is over
                print('Warmup is over')
                warmup = False
                warmup_scheduler.set_lr(optimizer, args.lr, args.lr)

            # multi-scale trick
            if iter_i % 10 == 0 and iter_i > 0 and args.multi_scale:
                # randomly choose a new size
                train_size = int(random.choice(scale_jitter) * args.img_size)
                model_without_ddp.reset_anchors(train_size)
            if args.multi_scale:
                # interpolate
                images = F.interpolate(input=images, 
                                       size=train_size, 
                                       mode='bilinear', 
                                       align_corners=False)

            # to device
            images = images.to(device)

            # inference
            cls_pred, box_pred = model_without_ddp(images)

            # compute loss
            cls_loss, reg_loss, total_loss = criterion(img_size = train_size,
                                                       pred_cls = cls_pred,
                                                       pred_box = box_pred,
                                                       targets = targets,
                                                       anchor_boxes = model_without_ddp.anchor_boxes,
                                                       images = images,
                                                       vis_labels=args.vis_targets)

            # visualize targets
            if args.vis_targets:
                continue
            
            loss_dict = dict(
                cls_loss=cls_loss,
                reg_loss=reg_loss,
                total_loss=total_loss
            )
            loss_dict_reduced = distributed_utils.reduce_dict(loss_dict)

            # check loss
            if torch.isnan(total_loss):
                print('loss is NAN !!')
                continue

            # Backward and Optimize
            total_loss.backward()
            if args.grad_clip_norm > 0. and epoch < args.lr_epoch[0]:
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            else:
                total_norm = get_total_grad_norm(model.parameters())
            optimizer.step()
            optimizer.zero_grad()

            if args.ema:
                ema.update(model_without_ddp)

            # display
            if distributed_utils.is_main_process() and iter_i % 10 == 0:
                t1 = time.time()
                cur_lr = [param_group['lr']  for param_group in optimizer.param_groups]
                print('[Epoch %d/%d][Iter %d/%d][lr: %.6f][Loss: cls %.2f || reg %.2f || gnorm: %.2f || size %d || time: %.2f]'
                        % (epoch+1, 
                           max_epoch, 
                           iter_i, 
                           epoch_size, 
                           cur_lr[0],
                           loss_dict_reduced['cls_loss'].item(), 
                           loss_dict_reduced['reg_loss'].item(), 
                           total_norm,
                           train_size, 
                           t1-t0),
                        flush=True)

                t0 = time.time()

        lr_scheduler.step()
        
        # evaluation
        if (epoch + 1) % args.eval_epoch == 0 or (epoch + 1) == max_epoch:
            # check evaluator
            if distributed_utils.is_main_process():
                if evaluator is None:
                    print('No evaluator ... save model and go on training.')
                    print('Saving state, epoch: {}'.format(epoch + 1))
                    weight_name = '{}_epoch_{}.pth'.format(args.version, epoch + 1)
                    torch.save(model_without_ddp.state_dict(), os.path.join(path_to_save, weight_name)) 
                else:
                    print('eval ...')
                    model_eval = ema.ema if args.ema else model_without_ddp

                    # set eval mode
                    model_eval.trainable = False
                    model_eval.reset_anchors(val_size)
                    model_eval.eval()

                    # evaluate
                    evaluator.evaluate(model_eval)

                    cur_map = evaluator.map
                    if cur_map > best_map:
                        # update best-map
                        best_map = cur_map
                        # save model
                        print('Saving state, epoch:', epoch + 1)
                        weight_name = '{}_epoch_{}_{:.2f}.pth'.format(args.version, epoch + 1, best_map*100)
                        torch.save(model_eval.state_dict(), os.path.join(path_to_save, weight_name)) 

                    # set train mode.
                    model_eval.trainable = True
                    model_eval.reset_anchors(train_size)
                    model_eval.train()
        
            if args.distributed:
                # wait for all processes to synchronize
                dist.barrier()

        # close mosaic augmentation
        if args.mosaic and max_epoch - epoch == 5:
            print('close Mosaic Augmentation ...')
            dataloader.dataset.mosaic = False


def build_dataset(cfg, args, device):
    # transform
    train_transform = TrainTransforms(img_size=args.img_size, 
                                      pixel_mean=cfg['pixel_mean'],
                                      pixel_std=cfg['pixel_std'],
                                      format=cfg['format'])
    val_transform = ValTransforms(img_size=args.img_size,
                                  pixel_mean=cfg['pixel_mean'],
                                  pixel_std=cfg['pixel_std'],
                                  format=cfg['format'])
    color_augment = BaseTransforms(img_size=args.img_size,
                                   pixel_mean=cfg['pixel_mean'],
                                   pixel_std=cfg['pixel_std'],
                                   format=cfg['format'])
    
    if args.dataset == 'voc':
        data_dir = os.path.join(args.root, 'VOCdevkit')
        num_classes = 20
        # dataset
        dataset = VOCDetection(img_size=args.img_size,
                               data_dir=data_dir, 
                               transform=train_transform,
                               color_augment=color_augment,
                               mosaic=args.mosaic)
        # evaluator
        evaluator = VOCAPIEvaluator(data_dir=data_dir,
                                    device=device,
                                    transform=val_transform)

    elif args.dataset == 'coco':
        data_dir = os.path.join(args.root, 'COCO')
        num_classes = 80
        # dataset
        dataset = COCODataset(img_size=args.img_size,
                              data_dir=data_dir,
                              image_set='train2017',
                              transform=train_transform,
                              color_augment=color_augment,
                              mosaic=args.mosaic)
        # evaluator
        evaluator = COCOAPIEvaluator(data_dir=data_dir,
                                     device=device,
                                     transform=val_transform)
    
    else:
        print('unknow dataset !! Only support voc and coco !!')
        exit(0)

    # dataset
    print('==============================')
    print('Training model on:', args.dataset)
    print('The dataset size:', len(dataset))

    return dataset, evaluator, num_classes


def build_dataloader(args, dataset, collate_fn=None):
    # distributed
    if args.distributed:
        # dataloader
        dataloader = torch.utils.data.DataLoader(
                        dataset=dataset, 
                        batch_size=args.batch_size, 
                        collate_fn=collate_fn,
                        num_workers=args.num_workers,
                        pin_memory=True,
                        sampler=torch.utils.data.distributed.DistributedSampler(dataset)
                        )

    else:
        # dataloader
        dataloader = torch.utils.data.DataLoader(
                        dataset=dataset, 
                        shuffle=True,
                        batch_size=args.batch_size, 
                        collate_fn=collate_fn,
                        num_workers=args.num_workers,
                        pin_memory=True,
                        drop_last=True
                        )
    return dataloader
    

if __name__ == '__main__':
    train()
