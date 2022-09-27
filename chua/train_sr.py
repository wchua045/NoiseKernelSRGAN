'''
Code is based on RealSR: https://github.com/Tencent/Real-SR
'''


import os
import math
import argparse
import random
import logging
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import socket
from contextlib import closing

from data.data_sampler import DistIterSampler
import options.options as option
from utils import util
from data import create_dataloader, create_dataset  # creates LR-HR pair
from models import create_model


def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def init_dist(backend='nccl', rank=0):
    ''' initialization for distributed training: https://stackoverflow.com/questions/66498045/how-to-solve-dist-init-process-group-from-hanging-or-deadlocks'''
    # if mp.get_start_method(allow_none=True) is None:
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn')
    # rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    MASTER_ADDR = '127.0.0.1'
    MASTER_PORT = find_free_port()
    # set up the master's ip address so this child process can coordinate
    os.environ['MASTER_ADDR'] = MASTER_ADDR
    print(f"{MASTER_ADDR}")
    os.environ['MASTER_PORT'] = MASTER_PORT
    print(f"{MASTER_PORT}")
    dist.init_process_group(
        backend=backend, world_size=num_gpus, rank=rank)
    #dist.init_process_group(backend=backend, init_method="tcp://127.0.0.1:23571", world_size=num_gpus, rank=rank)


def main():
    # options
    parser = argparse.ArgumentParser()
    parser.add_argument('-opt', type=str, help='Path to option YMAL file.')
    parser.add_argument('--launcher', choices=['none', 'pytorch'], default='pytorch',
                        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    # distributed training settings
    if args.launcher == 'none':  # disabled distributed training
        opt['dist'] = False
        rank = -1
        print('Disabled distributed training.')
    else:
        opt['dist'] = True
        init_dist()
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        #world_size = 1
        #rank = 0

    # loading resume state if exists
    if opt['path'].get('resume_state', None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(opt['path']['resume_state'],
                                  map_location=lambda storage, loc: storage.cuda(device_id))
        option.check_resume(opt, resume_state['iter'])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0)
        if resume_state is None:
            util.mkdir_and_rename(
                opt['path']['experiments_root'])  # rename experiment folder if exists
            util.mkdirs((path for key, path in opt['path'].items() if not key == 'experiments_root'
                         and 'pretrain_model' not in key and 'resume' not in key))

        # config loggers. Before it, the log will not work
        util.setup_logger('base', opt['path']['log'], 'train_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        util.setup_logger('val', opt['path']['log'], 'val_' + opt['name'], level=logging.INFO,
                          screen=True, tofile=True)
        logger = logging.getLogger('base')
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt['use_tb_logger'] and 'debug' not in opt['name']:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    'You are using PyTorch {}. Tensorboard will use [tensorboardX]'.format(version))
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir='tb_logger/' + opt['name'])
    else:
        util.setup_logger('base', opt['path']['log'],
                          'train', level=logging.INFO, screen=True)
        logger = logging.getLogger('base')

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    # random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    if rank <= 0:
        logger.info('Random seed: {}'.format(seed))
        logger.info('RANK AND WORLD SIZE: {:d}, iter: {:d}'.format(
            rank, world_size))
    util.set_random_seed(seed)

    # torch.backends.cudnn.benckmark = True
    # torch.backends.cudnn.deterministic = True

    # create train and val dataloader
    dataset_ratio = 1  # or 200 in RealSR -- enlarge the size of each epoch
    train_loader = None
    val_loader = None
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            # print('\n\n\n\n\n\n\n\n', dataset_opt)
            train_set = create_dataset(dataset_opt)  # 1484
            train_size = int(
                math.ceil(len(train_set) / dataset_opt['batch_size']))  # 53
            total_iters = int(opt['train']['niter'])  # 3000
            # division by zero
            total_epochs = int(math.ceil(total_iters / train_size))
            # if opt['dist']:
            if True:
                train_sampler = DistIterSampler(
                    train_set, world_size, rank, dataset_ratio)
                total_epochs = int(
                    math.ceil(total_iters / (train_size * dataset_ratio)))
            else:
                train_sampler = None
            train_loader = create_dataloader(
                train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info('Number of train images: {:,d}, iters: {:,d}'.format(
                    len(train_set), train_size))
                logger.info('Total epochs needed: {:d} for iters {:,d}'.format(
                    total_epochs, total_iters))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info('Number of val images in [{:s}]: {:d}'.format(
                    dataset_opt['name'], len(val_set)))
        else:
            raise NotImplementedError(
                'Phase [{:s}] is not recognized.'.format(phase))
    assert train_loader is not None

    # create model
    model = create_model(opt)

    # resume training
    if resume_state:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            resume_state['epoch'], resume_state['iter']))

        start_epoch = resume_state['epoch']
        current_step = resume_state['iter']
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0

    # training
    logger.info('Start training from epoch: {:d}, iter: {:d}'.format(
        start_epoch, current_step))
    for epoch in range(start_epoch, total_epochs + 1):
        # if opt['dist']:
        if True:
            train_sampler.set_epoch(epoch)
        for ii, train_data in enumerate(train_loader):
            current_step += 1
            if current_step > total_iters:
                break
            # update learning rate
            if ii > 0:
                model.update_learning_rate(
                    current_step, warmup_iter=opt['train']['warmup_iter'])

            # training
            model.feed_data(train_data)
            model.optimize_parameters(current_step)

            if rank <= 0:
                logs = model.get_current_log()
                message = "ep={}, iter={}/{}, lr={:6f}".format(
                    epoch, current_step, total_iters, model.get_current_learning_rate())
                for k, v in logs.items():
                    message += ', {:s}={:.5f}'.format(k, v)
                    # tensorboard logger
                    if opt['use_tb_logger'] and 'debug' not in opt['name']:
                        tb_logger.add_scalar(k, v, current_step)
                print(message + "\r", end="   ")
            # log
            # if current_step % opt['logger']['print_freq'] == 0:
            #     logs = model.get_current_log()
            #     message = '<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> '.format(
            #         epoch, current_step, model.get_current_learning_rate())
            #     for k, v in logs.items():
            #         message += '{:s}: {:.4e} '.format(k, v)
            #         # tensorboard logger
            #         if opt['use_tb_logger'] and 'debug' not in opt['name']:
            #             if rank <= 0:
            #                 tb_logger.add_scalar(k, v, current_step)
            #     if rank <= 0:
            #         logger.info(message)
            logger.info('Epoch: {:d}, iter: {:d}'.format(
                epoch, current_step))

            # validation
            if current_step % opt['train']['val_freq'] == 0 and rank <= 0 and val_loader is not None:
                avg_psnr = val_pix_err_f = val_pix_err_nf = val_mean_color_err = 0.0
                idx = 0
                print("\n\tvalidation: ep={}, curr_step={}".format(
                    epoch, current_step))
                n_test_pick = len(val_set.test_samples)
                for ti in range(n_test_pick):
                    test_sample = val_set.get_test_sample(ti)
                    img_name = "TEST_" + \
                        os.path.splitext(os.path.basename(
                            test_sample['LQ_path'][0]))[0]
                    img_dir = os.path.join(opt['path']['val_images'], img_name)
                    util.mkdir(img_dir)
                    model.feed_data(test_sample, need_GT=False)
                    model.test()
                    visuals = model.get_current_visuals(need_GT=False)
                    sr_img = util.tensor2img(visuals['SR'])  # uint8
                    save_img_path = os.path.join(
                        img_dir, '{:s}_{:d}.png'.format(img_name, current_step))
                    util.save_img(sr_img, save_img_path)

                for val_data in val_loader:
                    if idx >= opt['datasets']['val']['num_val']:
                        break
                    idx += 1
                    img_name = os.path.splitext(
                        os.path.basename(val_data['LQ_path'][0]))[0]
                    img_dir = os.path.join(opt['path']['val_images'], img_name)
                    # img_dir = opt['path']['val_images']
                    util.mkdir(img_dir)

                    model.feed_data(val_data, need_GT=True)
                    model.test()

                    visuals = model.get_current_visuals(need_GT=True)
                    sr_img = util.tensor2img(visuals['SR'])  # uint8
                    gt_img = util.tensor2img(visuals['GT'])  # uint8

                    # Save SR images for reference
                    save_img_path = os.path.join(img_dir,
                                                 '{:s}_{:d}.png'.format(img_name, current_step))
                    util.save_img(sr_img, save_img_path)

                    # # calculate PSNR
                    '''Peak signal-to-noise ratio (PSNR) is an engineering term for the ratio between the maximum possible power of a signal and the power of corrupting noise that affects the fidelity of its representation. Because many signals have a very wide dynamic range, PSNR is usually expressed as a logarithmic quantity using the decibel scale.

                    PSNR is commonly used to quantify reconstruction quality for images and video subject to lossy compression.'''

                    crop_size = opt['scale']
                    gt_img = gt_img / 255.
                    sr_img = sr_img / 255.
                    cropped_sr_img = sr_img[crop_size:-
                                            crop_size, crop_size:-crop_size, :]
                    cropped_gt_img = gt_img[crop_size:-
                                            crop_size, crop_size:-crop_size, :]
                    avg_psnr += util.calculate_psnr(
                        cropped_sr_img * 255, cropped_gt_img * 255)
                avg_psnr = avg_psnr / idx
                val_pix_err_f /= idx
                val_pix_err_nf /= idx
                val_mean_color_err /= idx

                # log
                logger.info('# Validation # PSNR: {:.3f}'.format(avg_psnr))
                logger_val = logging.getLogger('val')  # validation logger
                logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.3f}'.format(
                    epoch, current_step, avg_psnr))
                # tensorboard logger
                if opt['use_tb_logger'] and 'debug' not in opt['name']:
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)
                    tb_logger.add_scalar(
                        'val_pix_err_f', val_pix_err_f, current_step)
                    tb_logger.add_scalar(
                        'val_pix_err_nf', val_pix_err_nf, current_step)
                    tb_logger.add_scalar(
                        'val_mean_color_err', val_mean_color_err, current_step)

            # save models and training states
            if current_step % opt['logger']['save_checkpoint_freq'] == 0:
                if rank <= 0:
                    logger.info('Saving models and training states.')
                    model.save(current_step)
                    model.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info('Saving the final model.')
        model.save('latest')  # wait where is the model saved....
        logger.info('End of training.')


if __name__ == '__main__':
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    #num_gpus = torch.cuda.device_count()
    #torch.multiprocessing.spawn(main, nprocs=num_gpus, args=(num_gpus, ))
    main()
