import torch
from torch.utils.data import DataLoader

import argparse
import numpy as np
import os

import netstructure
import dataloader
from dataloader import transforms
from utils import utils
import model

# 使用image net的均值和标注差 进行数据的初始化
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

parser = argparse.ArgumentParser()

#设置模式， 是train还是Test还是Train 
parser.add_argument('--mode', default='test', type=str,
                    help='Validation mode on small subset or test mode on full test data')

# Training data
parser.add_argument('--data_dir', default='data/SceneFlow', type=str, help='Training dataset')
parser.add_argument('--dataset_name', default='SceneFlow', type=str, help='Dataset name')

parser.add_argument('--batch_size', default=64, type=int, help='Batch size for training')
parser.add_argument('--val_batch_size', default=64, type=int, help='Batch size for validation')
parser.add_argument('--num_workers', default=8, type=int, help='Number of workers for data loading')
parser.add_argument('--img_height', default=288, type=int, help='Image height for training')
parser.add_argument('--img_width', default=512, type=int, help='Image width for training')

# For KITTI, using 384x1248 for validation
parser.add_argument('--val_img_height', default=576, type=int, help='Image height for validation')
parser.add_argument('--val_img_width', default=960, type=int, help='Image width for validation')

####################################      Model 的基础参数    #############################################
parser.add_argument('--seed', default=326, type=int, help='Random seed for reproducibility')
parser.add_argument('--checkpoint_dir', default=None, type=str, required=True,
                    help='Directory to save model checkpoints and logs')
parser.add_argument('--learning_rate', default=1e-3, type=float, help='Learning rate')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay for optimizer')
parser.add_argument('--max_disp', default=192, type=int, help='Max disparity')
parser.add_argument('--max_epoch', default=64, type=int, help='Maximum epoch number for training')
parser.add_argument('--resume', action='store_true', help='Resume training from latest checkpoint')

####################################      AANet的 控制参数    #############################################
parser.add_argument('--feature_type', default='aanet', type=str, help='Type of feature extractor') # 使用aanet
parser.add_argument('--no_feature_mdconv', action='store_true', help='Whether to use mdconv for feature extraction') # 默认使用md conv
parser.add_argument('--feature_pyramid', action='store_true', help='Use pyramid feature') # 默认不使用Feature Pyramid
parser.add_argument('--feature_pyramid_network', action='store_true', help='Use FPN') # 默认不使用FPN
parser.add_argument('--feature_similarity', default='correlation', type=str,
                    help='Similarity measure for matching cost')         #求Cost Volume默认的使用correlation
# refinement 使用2次慢慢向上卷积
parser.add_argument('--num_downsample', default=2, type=int, help='Number of downsample layer for feature extraction') 
# 使用Adaptive 进行聚合
parser.add_argument('--aggregation_type', default='adaptive', type=str, help='Type of cost aggregation')
# 尺寸为3 用于ISA和CSA
parser.add_argument('--num_scales', default=3, type=int, help='Number of stages when using parallel aggregation')
# AAMoudules 的个数，这里使用6个AAMoudules
parser.add_argument('--num_fusions', default=6, type=int, help='Number of multi-scale fusions when using parallel'
                                                               'aggragetion')
# num_stages_blocks： Deform Conv 的 Residual Block的个数
parser.add_argument('--num_stage_blocks', default=1, type=int, help='Number of deform blocks for ISA')

# AAMoudule的个数里面有num_deform_blocks个使用卷积的Residual Blocks
parser.add_argument('--num_deform_blocks', default=3, type=int, help='Number of DeformBlocks for aggregation')
# 是否使用中间监督，也就是D/12,/D/6也进行监督计算，贡献LOSS
parser.add_argument('--no_intermediate_supervision', action='store_true',
                    help='Whether to add intermediate supervision')

# 是否进行分组卷积，以及分组卷积的个数
parser.add_argument('--deformable_groups', default=2, type=int, help='Number of deformable groups')
parser.add_argument('--mdconv_dilation', default=2, type=int, help='Dilation rate for deformable conv')
# Refinement , StereodrNet的种类
parser.add_argument('--refinement_type', default='stereodrnet', help='Type of refinement module')

# 是否还用pre-trained Model
parser.add_argument('--pretrained_aanet', default=None, type=str, help='Pretrained network')
# 是否需要冻结BN层
parser.add_argument('--freeze_bn', action='store_true', help='Switch BN to eval mode to fix running statistics')


###################################  Learning Rate ###############################################
parser.add_argument('--lr_decay_gamma', default=0.5, type=float, help='Decay gamma') # Decay Rate
parser.add_argument('--lr_scheduler_type', default='MultiStepLR', help='Type of learning rate scheduler') # 使用MutilStepLR更新Learning Weights
parser.add_argument('--milestones', default=None, type=str, help='Milestones for MultiStepLR') # 遇到mileStone的epoch的时候，更新LR

# Loss
parser.add_argument('--highest_loss_only', action='store_true', help='Only use loss on highest scale for finetuning') # 使用最高分辨率作为输出
parser.add_argument('--load_pseudo_gt', action='store_true', help='Load pseudo gt for supervision')  # 是否用Pseudo GT  for KITTI dataset

# Log
parser.add_argument('--print_freq', default=100, type=int, help='Print frequency to screen (iterations)')
parser.add_argument('--summary_freq', default=100, type=int, help='Summary frequency to tensorboard (iterations)')
parser.add_argument('--no_build_summary', action='store_true', help='Dont save sammary when training to save space')
parser.add_argument('--save_ckpt_freq', default=10, type=int, help='Save checkpoint frequency (epochs)')

parser.add_argument('--evaluate_only', action='store_true', help='Evaluate pretrained models')
parser.add_argument('--no_validate', action='store_true', help='No validation')
parser.add_argument('--strict', action='store_true', help='Strict mode when loading checkpoints')
parser.add_argument('--val_metric', default='epe', help='Validation metric to select best model')

args = parser.parse_args()
logger = utils.get_logger()

# 是否存在可以保留的路径
utils.check_path(args.checkpoint_dir)
# Save the args into a json file, 保存一个控制参数
utils.save_args(args)

# Command  Test
# 保存 此时的输入？ -? -?-?-?-?-?-?-?-?
filename = 'command_test.txt' if args.mode == 'test' else 'command_train.txt'
utils.save_command(args.checkpoint_dir, filename)


def main():
    # For reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Torch beckmark is aviable
    torch.backends.cudnn.benchmark = True
    
    # 确定device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train loader
    train_transform_list = [transforms.RandomCrop(args.img_height, args.img_width),
                            transforms.RandomColor(),  # Random Color
                            transforms.RandomVerticalFlip(),    
                            transforms.ToTensor(),
                            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                            ]
    train_transform = transforms.Compose(train_transform_list)   # train transfrom
   
    # 建立输入的数据集
    train_data = dataloader.StereoDataset(data_dir=args.data_dir,
                                          dataset_name=args.dataset_name,
                                          mode='train' if args.mode != 'train_all' else 'train_all',   # KITTI train-all  or train
                                          load_pseudo_gt=args.load_pseudo_gt,
                                          transform=train_transform)
    # LOG  ： train loader 的信息
    logger.info('=> {} training samples found in the training set'.format(len(train_data)))

    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # Validation loader：  这里不进行数据增强
    val_transform_list = [transforms.RandomCrop(args.val_img_height, args.val_img_width, validate=True),
                          transforms.ToTensor(),
                          transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
                         ]
    val_transform = transforms.Compose(val_transform_list)
    val_data = dataloader.StereoDataset(data_dir=args.data_dir,
                                        dataset_name=args.dataset_name,
                                        mode=args.mode,   
                                        transform=val_transform)

    val_loader = DataLoader(dataset=val_data, batch_size=args.val_batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, drop_last=False)  #pin_memory 表示内存锁页，速度会快很多

    # Network
    aanet = nets.AANet(args.max_disp,
                       num_downsample=args.num_downsample,
                       feature_type=args.feature_type,
                       no_feature_mdconv=args.no_feature_mdconv,
                       feature_pyramid=args.feature_pyramid,
                       feature_pyramid_network=args.feature_pyramid_network,
                       feature_similarity=args.feature_similarity,
                       aggregation_type=args.aggregation_type,
                       num_scales=args.num_scales,
                       num_fusions=args.num_fusions,
                       num_stage_blocks=args.num_stage_blocks,
                       num_deform_blocks=args.num_deform_blocks,
                       no_intermediate_supervision=args.no_intermediate_supervision,
                       refinement_type=args.refinement_type,
                       mdconv_dilation=args.mdconv_dilation,
                       deformable_groups=args.deformable_groups).to(device)
    
    # 打印网络的目前的信息
    logger.info('%s' % aanet)

    if args.pretrained_aanet is not None:
        logger.info('=> Loading pretrained AANet: %s' % args.pretrained_aanet)
        # Enable training from a partially pretrained model
        utils.load_pretrained_net(aanet, args.pretrained_aanet, no_strict=(not args.strict))
     
     # 这里使用多个GPU设备进行训练
    if torch.cuda.device_count() > 1:
        logger.info('=> Use %d GPUs' % torch.cuda.device_count())
        aanet = torch.nn.DataParallel(aanet)   # 数据并行化处理 ：DataParallel 

    # Save parameters
    num_params = utils.count_parameters(aanet)  # 返回多少个训练的参数
    logger.info('=> Number of trainable parameters: %d' % num_params)
    save_name = '%d_parameters' % num_params
    open(os.path.join(args.checkpoint_dir, save_name), 'a').close()

    # Optimizer
    # Learning rate for offset learning is set 0.1 times those of existing layers
    # Specific_Params : Defrom Conv Ones
    specific_params = list(filter(utils.filter_specific_params,
                                  aanet.named_parameters()))
    # Base Params:  Parameters except the Defrom convs ones
    base_params = list(filter(utils.filter_base_params,
                              aanet.named_parameters()))

    specific_params = [kv[1] for kv in specific_params]  # kv is a tuple (key, value)
    base_params = [kv[1] for kv in base_params]

    specific_lr = args.learning_rate * 0.1    # defrom conv is the learning rate *1, other use normal learning rate
    params_group = [
        {'params': base_params, 'lr': args.learning_rate},
        {'params': specific_params, 'lr': specific_lr},
    ]
    
    # 确定Optimizer
    optimizer = torch.optim.Adam(params_group, weight_decay=args.weight_decay)

    # Resume training
    if args.resume:
        # AANet
        start_epoch, start_iter, best_epe, best_epoch = utils.resume_latest_ckpt(
            args.checkpoint_dir, aanet, 'aanet')

        # Optimizer
        utils.resume_latest_ckpt(args.checkpoint_dir, optimizer, 'optimizer')
    else:
        start_epoch = 0
        start_iter = 0
        best_epe = None
        best_epoch = None

    # LR scheduler
    if args.lr_scheduler_type is not None:
        last_epoch = start_epoch if args.resume else start_epoch - 1
        if args.lr_scheduler_type == 'MultiStepLR':
            milestones = [int(step) for step in args.milestones.split(',')]
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                                milestones=milestones,
                                                                gamma=args.lr_decay_gamma,
                                                                last_epoch=last_epoch)
        else:
            raise NotImplementedError

    train_model = model.Model(args, logger, optimizer, aanet, device, start_iter, start_epoch,
                              best_epe=best_epe, best_epoch=best_epoch)

    logger.info('=> Start training...')

    # 如果只是 evaluation ， 令batch_size =1
    if args.evaluate_only:
        assert args.val_batch_size == 1
        train_model.validate(val_loader)  
    else:
        for _ in range(start_epoch, args.max_epoch):
            if not args.evaluate_only:
                train_model.train(train_loader)
            if not args.no_validate:
                train_model.validate(val_loader)
            if args.lr_scheduler_type is not None:
                lr_scheduler.step() # Behind the lr_schedular

        logger.info('=> End training\n\n')


if __name__ == '__main__':
    main()
