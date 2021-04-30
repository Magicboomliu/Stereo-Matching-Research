import torch
import time
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import os

from utils import utils
from utils.visualization import disp_error_img, save_images
from metric import d1_metric, thres_metric


class Model(object):
    def __init__(self, args, logger, optimizer, aanet, device, start_iter=0, start_epoch=0,
                 best_epe=None, best_epoch=None):
        self.args = args    # args就是网络的参数
        self.logger = logger  # Logger就是Logger
        self.optimizer = optimizer # 使用什么优化器
        self.aanet = aanet  # 输入网络模型
        self.device = device # 在什么地方进行训练，在什么device上面
        self.num_iter = start_iter  # Iteration 的数目
        self.epoch = start_epoch # 开始的epoach数目

        self.best_epe = 999. if best_epe is None else best_epe  # 最好的Epe是多少？ 根据模型测试出来的
        self.best_epoch = -1 if best_epoch is None else best_epoch # 在那一个epoch上达到最好的这个EPE?
         
         # 是否需要保存模型到 checkpoint dir
        if not args.evaluate_only:
            self.train_writer = SummaryWriter(self.args.checkpoint_dir)

    def train(self, train_loader):
        args = self.args   # 需要的一些参数
        logger = self.logger # Log Something 

        steps_per_epoch = len(train_loader)   # Batch SIze
        device = self.device # GPU or CPU

        self.aanet.train()  # 告诉编译器，当前的model是 “Train”模式， 要更新所有的Learningable Parameters
         
         # Test 时候不需要BatchNorm层，需要固定BatchNorm的参数，使用eval
        if args.freeze_bn:
            def set_bn_eval(m):
                classname = m.__class__.__name__
                if classname.find('BatchNorm') != -1:
                    m.eval()

            self.aanet.apply(set_bn_eval)

        # Learning rate summary
        base_lr = self.optimizer.param_groups[0]['lr']              # Base Learning rate
        offset_lr = self.optimizer.param_groups[1]['lr']            # Learning rate offset
        self.train_writer.add_scalar('base_lr', base_lr, self.epoch + 1)   # base_lr 是 y轴的数值，self.epoch是x的变化量
        self.train_writer.add_scalar('offset_lr', offset_lr, self.epoch + 1)  # base_lr 是 y轴的数值，self.epoch是x的变化量

        last_print_time = time.time()  # From the training Time

        for i, sample in enumerate(train_loader):
            left = sample['left'].to(device)  # [B, 3, H, W]
            right = sample['right'].to(device)
            gt_disp = sample['disp'].to(device)  # [B, H, W]

            mask = (gt_disp > 0) & (gt_disp < args.max_disp)   # Selected the Mask where the disp is valid
            
            # Whether use the pseudo GT
            if args.load_pseudo_gt:
                pseudo_gt_disp = sample['pseudo_disp'].to(device)
                pseudo_mask = (pseudo_gt_disp > 0) & (pseudo_gt_disp < args.max_disp) & (~mask)  # inverse mask
            
            # if mask is None: Discard this image
            if not mask.any():
                continue
            
            # 注意是在这里拿到了 pred_disp_pyramid的 预测的结果
            pred_disp_pyramid = self.aanet(left, right)  # list of H/12, H/6, H/3, H/2, H

            # 如何训练的问题
            if args.highest_loss_only:
                pred_disp_pyramid = [pred_disp_pyramid[-1]]  # only the last highest resolution output

            disp_loss = 0
            pseudo_disp_loss = 0
            pyramid_loss = []
            pseudo_pyramid_loss = []

            # Loss weights For each layer
            if len(pred_disp_pyramid) == 5: # [H/12, H/6, H/3, H/2, H]
                pyramid_weight = [1 / 3, 2 / 3, 1.0, 1.0, 1.0]  # AANet and AANet+
            elif len(pred_disp_pyramid) == 4:
                pyramid_weight = [1 / 3, 2 / 3, 1.0, 1.0]
            elif len(pred_disp_pyramid) == 3:
                pyramid_weight = [1.0, 1.0, 1.0]  # 1 scale only
            elif len(pred_disp_pyramid) == 1:
                pyramid_weight = [1.0]  # highest loss only
            else:
                raise NotImplementedError

            assert len(pyramid_weight) == len(pred_disp_pyramid)
            ######这里开始每个每个尺寸的叠加LOSS####################

            for k in range(len(pred_disp_pyramid)):
                pred_disp = pred_disp_pyramid[k]  # Get the disparity
                weight = pyramid_weight[k]                # Get the weight
                 
                 ## 这里已经做过尺度的处理了
                if pred_disp.size(-1) != gt_disp.size(-1): 
                    # 首先扩大预测 和 disp_gt比一比
                    pred_disp = pred_disp.unsqueeze(1)  # [B, 1, H, W]
                    pred_disp = F.interpolate(pred_disp, size=(gt_disp.size(-2), gt_disp.size(-1)),
                                              mode='bilinear', align_corners=False) * (gt_disp.size(-1) / pred_disp.size(-1))
                    pred_disp = pred_disp.squeeze(1)  # [B, H, W]

                curr_loss = F.smooth_l1_loss(pred_disp[mask], gt_disp[mask],
                                             reduction='mean')  # 当前尺度下的Loss

                disp_loss += weight * curr_loss  # weight_loss_sum ： 记录为disp-loss
                pyramid_loss.append(curr_loss)  # 记录一下当前的各个尺度的误差

                # Pseudo gt loss
                if args.load_pseudo_gt:
                    pseudo_curr_loss = F.smooth_l1_loss(pred_disp[pseudo_mask], pseudo_gt_disp[pseudo_mask],
                                                        reduction='mean')
                    pseudo_disp_loss += weight * pseudo_curr_loss 

                    pseudo_pyramid_loss.append(pseudo_curr_loss)

            total_loss = disp_loss + pseudo_disp_loss                         # 求2个误差的和

            self.optimizer.zero_grad()   # Gradient 清空
            total_loss.backward()           # 计算loss
            self.optimizer.step()                # 更新操作

            self.num_iter += 1     # 每次过一个Train loader, Iteration nums +1
            
            # 这里干的事情是 ： LOG 当前训练的进度
            if self.num_iter % args.print_freq == 0:                    
                this_cycle = time.time() - last_print_time           
                last_print_time += this_cycle                                   

                logger.info('Epoch: [%3d/%3d] [%5d/%5d] time: %4.2fs disp_loss: %.3f' %
                            (self.epoch + 1, args.max_epoch, i + 1, steps_per_epoch, this_cycle,
                             disp_loss.item()))              # 输出当前的epoch, 当前的step, 当前的loss
            
            # 这里干的事情是 ： Summary 当前训练的东西（写在文件里面）
            if self.num_iter % args.summary_freq == 0:
                img_summary = dict()   # 这里首先保存一个 image summary 的字典，用于后期可以用来看效果。
                img_summary['left'] = left
                img_summary['right'] = right
                img_summary['gt_disp'] = gt_disp

                if args.load_pseudo_gt:
                    img_summary['pseudo_gt_disp'] = pseudo_gt_disp

                # Save pyramid disparity prediction
                for s in range(len(pred_disp_pyramid)):
                    # Scale from low to high, reverse
                    save_name = 'pred_disp' + str(len(pred_disp_pyramid) - s - 1)    # 当前的尺寸
                    save_value = pred_disp_pyramid[s] # 当前的disparity MAP
                    img_summary[save_name] = save_value # 记录为一个Image Summary中的其中一项， 方便后期查看
               
                # 拿到一个全尺度 的 Disparity 的预测结果
                pred_disp = pred_disp_pyramid[-1]  
                
                # 接下来的部分 就是记录 FULL resolution 下面的 DISP EROR
                if pred_disp.size(-1) != gt_disp.size(-1):
                    pred_disp = pred_disp.unsqueeze(1)  # [B, 1, H, W]
                    pred_disp = F.interpolate(pred_disp, size=(gt_disp.size(-2), gt_disp.size(-1)),
                                              mode='bilinear', align_corners=False) * (gt_disp.size(-1) / pred_disp.size(-1))
                    pred_disp = pred_disp.squeeze(1)  # [B, H, W]

                #   Get the disp Error  for watching
                img_summary['disp_error'] = disp_error_img(pred_disp, gt_disp)

                # Save the images: the left, the right, the gt-disparity ,the peseudo disparity , the scale disparity , the totola loss
                save_images(self.train_writer, 'train', img_summary, self.num_iter)

                # 统计当前的 F1 loss 和 epe
                epe = F.l1_loss(gt_disp[mask], pred_disp[mask], reduction='mean')

                # 保存tensorborad  一个iter: 也就是一个 Batch Size 之后， 模型的 EPE , Disp Loss 和 total loss
                self.train_writer.add_scalar('train/epe', epe.item(), self.num_iter)
                self.train_writer.add_scalar('train/disp_loss', disp_loss.item(), self.num_iter)
                self.train_writer.add_scalar('train/total_loss', total_loss.item(), self.num_iter)

                # Save loss of different scale, 保存显示每个Scale的Loss
                for s in range(len(pyramid_loss)):
                    save_name = 'train/loss' + str(len(pyramid_loss) - s - 1)
                    save_value = pyramid_loss[s]
                    self.train_writer.add_scalar(save_name, save_value, self.num_iter)
                
                # 记录当前的D1 Value
                d1 = d1_metric(pred_disp, gt_disp, mask)
                self.train_writer.add_scalar('train/d1', d1.item(), self.num_iter)
                # 这里的记录所有的数据中， e大于3的means
                thres1 = thres_metric(pred_disp, gt_disp, mask, 1.0)
                thres2 = thres_metric(pred_disp, gt_disp, mask, 2.0)
                thres3 = thres_metric(pred_disp, gt_disp, mask, 3.0)
                self.train_writer.add_scalar('train/thres1', thres1.item(), self.num_iter)
                self.train_writer.add_scalar('train/thres2', thres2.item(), self.num_iter)
                self.train_writer.add_scalar('train/thres3', thres3.item(), self.num_iter)

        self.epoch += 1   # Train loader 结束之后，再来一个epoch的意思

        # Always save the latest model for resuming training， 每一个epoch ， Save一次data
        if args.no_validate:
            utils.save_checkpoint(args.checkpoint_dir, self.optimizer, self.aanet,
                                  epoch=self.epoch, num_iter=self.num_iter,
                                  epe=-1, best_epe=self.best_epe,
                                  best_epoch=self.best_epoch,
                                  filename='aanet_latest.pth')

            # Save checkpoint of specific epoch
            if self.epoch % args.save_ckpt_freq == 0:
                model_dir = os.path.join(args.checkpoint_dir, 'models')
                utils.check_path(model_dir)
                utils.save_checkpoint(model_dir, self.optimizer, self.aanet,
                                      epoch=self.epoch, num_iter=self.num_iter,
                                      epe=-1, best_epe=self.best_epe,
                                      best_epoch=self.best_epoch,
                                      save_optimizer=False)

    # This is Used for validation
    def validate(self, val_loader):
        args = self.args
        logger = self.logger
        logger.info('=> Start validation...')  # 开始验证
        
        # 如果验证模型而不是测试， 使用pretrained  Model
        if args.evaluate_only is True:
            # 如果存在PreTained 的模型， 直接用
            if args.pretrained_aanet is not None:
                pretrained_aanet = args.pretrained_aanet
            # 如果不存在，去load训练出来的BEST AANET的Module
            else:
                model_name = 'aanet_best.pth'
                pretrained_aanet = os.path.join(args.checkpoint_dir, model_name)   # Get pretained Model Name
                if not os.path.exists(pretrained_aanet):  # KITTI without validation
                    pretrained_aanet = pretrained_aanet.replace(model_name, 'aanet_latest.pth')  # 如果不存在，就去获取最新的模型
            # 读取要获取的模型的 地址
            logger.info('=> loading pretrained aanet: %s' % pretrained_aanet)
            # 读取相关的模型
            utils.load_pretrained_net(self.aanet, pretrained_aanet, no_strict=True)
        
        # 这里指定模式为eval ,也就是不更新相关参数， 冻结Dropout 和BN层
        self.aanet.eval()
       
        #  Validation Set 的数目
        num_samples = len(val_loader)
        logger.info('=> %d samples found in the validation set' % num_samples)
        
        # 一些 validation的 Mertic Parameters
        val_epe = 0  # EPE
        val_d1 = 0     # D1 
        val_thres1 = 0  # D1- threshold 1
        val_thres2 = 0  # D1 - threshold 2
        val_thres3 = 0  # D1 - threshold 3

        val_count = 0    ### What is this used for ? 
        
        # 把 valiadation 的测试的结果最后写在一个 txt文件里面，方便操作
        val_file = os.path.join(args.checkpoint_dir, 'val_results.txt')
        
        #  So  what is these used for ?
        num_imgs = 0
        valid_samples = 0

        for i, sample in enumerate(val_loader):
            if i % 100 == 0:
                logger.info('=> Validating %d/%d' % (i, num_samples))  # 已经验证了100倍的XXXX

            left = sample['left'].to(self.device)  # [B, 3, H, W]
            right = sample['right'].to(self.device)
            gt_disp = sample['disp'].to(self.device)  # [B, H, W]
            mask = (gt_disp > 0) & (gt_disp < args.max_disp)

            if not mask.any():
                continue

            valid_samples += 1     # 已经验证的Samples的个数

            num_imgs += gt_disp.size(0)
            
            # torch.no_grad() 是一个上下文管理器，被该语句 wrap 起来的部分将不会track 梯度。
            with torch.no_grad():
                pred_disp = self.aanet(left, right)[-1]  # [B, H, W]   # 这里就是 取的Full Resolution

            if pred_disp.size(-1) < gt_disp.size(-1):
                pred_disp = pred_disp.unsqueeze(1)  # [B, 1, H, W]
                pred_disp = F.interpolate(pred_disp, (gt_disp.size(-2), gt_disp.size(-1)),
                                          mode='bilinear', align_corners=False) * (gt_disp.size(-1) / pred_disp.size(-1))
                pred_disp = pred_disp.squeeze(1)  # [B, H, W]

            epe = F.l1_loss(gt_disp[mask], pred_disp[mask], reduction='mean')
            d1 = d1_metric(pred_disp, gt_disp, mask)
            thres1 = thres_metric(pred_disp, gt_disp, mask, 1.0)
            thres2 = thres_metric(pred_disp, gt_disp, mask, 2.0)
            thres3 = thres_metric(pred_disp, gt_disp, mask, 3.0)

            val_epe += epe.item()   # Epe 求和
            val_d1 += d1.item()        # D1 求和
            val_thres1 += thres1.item()  # 分别XXX 求和
            val_thres2 += thres2.item()
            val_thres3 += thres3.item()
            #####################This Part is used for Visualization#####################################
            # Save 3 images for visualization
            if not args.evaluate_only:   #  这里的i指代的是第几个 sample， 分别在1/4,  1/2 ， 和3/4的这几个位置进行Visualization
                if i in [num_samples // 4, num_samples // 2, num_samples // 4 * 3]:
                    img_summary = dict()
                    img_summary['disp_error'] = disp_error_img(pred_disp, gt_disp)
                    img_summary['left'] = left
                    img_summary['right'] = right
                    img_summary['gt_disp'] = gt_disp
                    img_summary['pred_disp'] = pred_disp
                    save_images(self.train_writer, 'val' + str(val_count), img_summary, self.epoch)
                    val_count += 1    # 所以这东西的上限就是 4

        logger.info('=> Validation done!')
         

         # 求平均 的 metric mean, 每一个epoch都会求一次
        mean_epe = val_epe / valid_samples
        mean_d1 = val_d1 / valid_samples
        mean_thres1 = val_thres1 / valid_samples
        mean_thres2 = val_thres2 / valid_samples
        mean_thres3 = val_thres3 / valid_samples

        # Save validation results
        with open(val_file, 'a') as f:
            f.write('epoch: %03d\t' % self.epoch)
            f.write('epe: %.3f\t' % mean_epe)
            f.write('d1: %.4f\t' % mean_d1)
            f.write('thres1: %.4f\t' % mean_thres1)
            f.write('thres2: %.4f\t' % mean_thres2)
            f.write('thres3: %.4f\n' % mean_thres3)

        logger.info('=> Mean validation epe of epoch %d: %.3f' % (self.epoch, mean_epe))
       
       # Train + val 为了就是看到val的结果
        if not args.evaluate_only:
            self.train_writer.add_scalar('val/epe', mean_epe, self.epoch)
            self.train_writer.add_scalar('val/d1', mean_d1, self.epoch)
            self.train_writer.add_scalar('val/thres1', mean_thres1, self.epoch)
            self.train_writer.add_scalar('val/thres2', mean_thres2, self.epoch)
            self.train_writer.add_scalar('val/thres3', mean_thres3, self.epoch)
        # not args.evaluate_only 的意思就是： Trian + val 同时进行，为的就是能够找到最合适epoch，来保存最好的模型
        if not args.evaluate_only:

            if args.val_metric == 'd1':
                # 更新最好的d1和对应的epoch
                if mean_d1 < self.best_epe:
                    # Actually best_epe here is d1
                    self.best_epe = mean_d1
                    self.best_epoch = self.epoch    #记录Branch主要用来保存
                    # Save the best_epe对应的那个epoch的模型
                    utils.save_checkpoint(args.checkpoint_dir, self.optimizer, self.aanet,
                                          epoch=self.epoch, num_iter=self.num_iter,
                                          epe=mean_d1, best_epe=self.best_epe,
                                          best_epoch=self.best_epoch,
                                          filename='aanet_best.pth')
            
            # 使用mean epe 来进行评估
            elif args.val_metric == 'epe':
                if mean_epe < self.best_epe:
                    self.best_epe = mean_epe
                    self.best_epoch = self.epoch
                   
                   # Save the mean对应的那个epoch的模型
                    utils.save_checkpoint(args.checkpoint_dir, self.optimizer, self.aanet,
                                          epoch=self.epoch, num_iter=self.num_iter,
                                          epe=mean_epe, best_epe=self.best_epe,
                                          best_epoch=self.best_epoch,
                                          filename='aanet_best.pth')
            else:
                raise NotImplementedError
         
         ############如果当前的epoch已经是指定的最大的epoch的#########################
         # 证明已经遍历完成了，可以找到最好的了 
        if self.epoch == args.max_epoch:
            ####### 在文件中保存最好插入最好的结果
            # Save best validation results
            with open(val_file, 'a') as f:
                f.write('\nbest epoch: %03d \t best %s: %.3f\n\n' % (self.best_epoch,
                                                                     args.val_metric,
                                                                     self.best_epe))

            logger.info('=> best epoch: %03d \t best %s: %.3f\n' % (self.best_epoch,
                                                                    args.val_metric,
                                                                    self.best_epe))
        ####### Save最好的模型
        # Always save the latest model for resuming training
        if not args.evaluate_only:
            utils.save_checkpoint(args.checkpoint_dir, self.optimizer, self.aanet,
                                  epoch=self.epoch, num_iter=self.num_iter,
                                  epe=mean_epe, best_epe=self.best_epe,
                                  best_epoch=self.best_epoch,
                                  filename='aanet_latest.pth')

            # Save checkpoint of specific epochs
            if self.epoch % args.save_ckpt_freq == 0:
                model_dir = os.path.join(args.checkpoint_dir, 'models')
                utils.check_path(model_dir)
                utils.save_checkpoint(model_dir, self.optimizer, self.aanet,
                                      epoch=self.epoch, num_iter=self.num_iter,
                                      epe=mean_epe, best_epe=self.best_epe,
                                      best_epoch=self.best_epoch,
                                      save_optimizer=False)
