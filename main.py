

from __future__ import print_function
import argparse
import torch.nn as nn
import torch.nn.parallel
from model import hyper_model
import torch.utils.data
import time
from PIL import ImageFile
from data_generator import *

import sys
sys.dont_write_bytecode = True

# ============================ Data & Networks =====================================
from tqdm import tqdm
# ==================================================================================


ImageFile.LOAD_TRUNCATED_IMAGES = True
os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES']='0'
seed = 1024
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def pretrain(encoder, max_epochs):
    fc_layer = nn.Linear(441*64, 64)

import torchvision
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('./runs/pics/img2class')
import scipy as sp
import scipy.stats
import scipy
def mean_confidence_interval(data, confidence=0.95):
    a = [1.0*np.array(data[i]) for i in range(len(data))]
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, h

def test(opt):


    # define loss function (criterion) and optimizer
    # model.method.eval()
    # optionally resume from a checkpoint
    model = hyper_model.Trainer(opt)
    model_dir = opt.outf
    ckpt = os.path.join(model_dir, 'model_best.pth.tar')
    if os.path.isfile(ckpt):
        print("=> loading checkpoint '{}'".format(ckpt))
        checkpoint = torch.load(ckpt)
        epoch_index = checkpoint['epoch_index']
        best_prec1 = checkpoint['best_prec1']
        keys = checkpoint['state_dict'].keys()
        checkpoint_tmp = {'state_dict': {}}
        for k in keys:
            k_tmp = k.replace('.module', '')
            checkpoint_tmp['state_dict'][k_tmp] = checkpoint['state_dict'][k]
        model.method.load_state_dict(checkpoint_tmp['state_dict'])
        # model.g_optimizer.load_state_dict(checkpoint_tmp['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})".format(ckpt, checkpoint['epoch_index']))
    else:
        print("=> no checkpoint found at '{}', use pretrained.".format(model_dir))
        if opt.metric == 'DWT':
            root_ = '/home/hsq/Projects/dwt-domain-adaptation/saved_model/conv_weights'
            if opt.data_name == 'visda':
                model_path = os.path.join(root_, 'visda_%s_to_%s_conv64f_best.pth.tar' % (opt.source_domain, opt.target_domain))
            elif opt.data_name == 'mini-Imagenet':
                model_path = os.path.join(root_, 'mini-imagenet_ycm_%s_to_%s_conv64f_best.pth.tar' % (opt.source_domain, opt.target_domain))
            elif opt.data_name == 'officehome':
                model_path = os.path.join(root_, 'officehome_%s_to_%s_conv64f_best.pth.tar' % (opt.source_domain, opt.target_domain))
            else:
                raise ValueError('Error dataset name.')
            model_dict = model.method.gen.state_dict()
            ckpt = torch.load(model_path, map_location='cuda:0')
            pretrained_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            if len(pretrained_dict) == 0:
                raise ValueError('Error ckpt.')
            model_dict.update(pretrained_dict)
            model.method.gen.load_state_dict(model_dict)
            print('load pretrained model from %s successly.'%root_)
        elif opt.metric in ('MCD', 'MCDR2D2'):
            root_ = '/home/hsq/Projects/IMSE/MCD_pretrained'
            if opt.data_name == 'visda':
                model_path = os.path.join(root_, 'MCD_%s_to_%s_model_conv64f.pt' % (
                opt.source_domain, opt.target_domain))
            elif opt.data_name == 'mini-Imagenet':
                model_path = os.path.join(root_, 'mini-imagenet_ycm_MCD_%s_to_%s_model_conv64f.pth.tar' % (
                opt.source_domain, opt.target_domain))
            elif opt.data_name == 'officehome':
                model_path = os.path.join(root_, 'officehome_MCD_%s_to_%s_model_conv64f.pth.tar' % (
                opt.source_domain, opt.target_domain))
            else:
                raise ValueError('Error dataset name.')
            model_dict = model.method.gen.state_dict()
            ckpt = torch.load(model_path, map_location='cuda:0')
            pretrained_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            if len(pretrained_dict) == 0:
                raise ValueError('Error ckpt.')
            model_dict.update(pretrained_dict)
            model.method.gen.load_state_dict(model_dict)
            print('load pretrained model from %s successly.' % root_)
        elif opt.metric == 'DANN':
            root_ = '/home/hsq/Projects/IMSE/DANN_pretrained'
            if opt.data_name == 'visda':
                model_path = os.path.join(root_, 'DANN_%s_to_%s_model_conv64f.pth' % (
                opt.source_domain, opt.target_domain))
            elif opt.data_name == 'mini-Imagenet':
                model_path = os.path.join(root_, 'mini-imagenet_ycm_DANN_conv64f_%s_to_%s.pth' % (
                opt.source_domain, opt.target_domain))
            elif opt.data_name == 'officehome':
                model_path = os.path.join(root_, 'officehome_DANN_conv64f_%s_to_%s.pth' % (
                opt.source_domain, opt.target_domain))
            else:
                raise ValueError('Error dataset name.')
            model_dict = model.method.gen.state_dict()
            ckpt = torch.load(model_path, map_location='cuda:0')
            pretrained_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            if len(pretrained_dict) == 0:
                raise ValueError('Error ckpt.')
            model_dict.update(pretrained_dict)
            model.method.gen.load_state_dict(model_dict)
            print('load pretrained model from %s successly.' % root_)
        elif opt.metric == 'ADDA':
            root_ = '/home/hsq/Projects/IMSE/ADDA_pretrained'
            if opt.data_name == 'visda':
                model_path = os.path.join(root_, 'visda_ADDA_conv64f_%s_to_%s_epoch_19.pth' % (
                opt.source_domain, opt.target_domain))
            elif opt.data_name == 'mini-Imagenet':
                model_path = os.path.join(root_, 'mini-imagenet_ycm_ADDA_conv64f_%s_to_%s_epoch_19.pth' % (
                opt.source_domain, opt.target_domain))
            elif opt.data_name == 'officehome':
                model_path = os.path.join(root_, 'officehome_ADDA_conv64f_%s_to_%s_epoch_19.pth' % (
                opt.source_domain, opt.target_domain))
            else:
                raise ValueError('Error dataset name.')
            model_dict = model.method.gen.state_dict()
            ckpt = torch.load(model_path, map_location='cuda:0')
            pretrained_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            if len(pretrained_dict) == 0:
                raise ValueError('Error ckpt.')
            model_dict.update(pretrained_dict)
            model.method.gen.load_state_dict(model_dict)
            print('load pretrained model from %s successly.' % root_)

        elif opt.rotate:
            root_ = '/home/hsq/Projects/rotate_pretrained'
            if opt.data_name == 'visda':
                model_path = os.path.join(root_, 'visda_%s_to_%s_rotate_resnet_best.pth.tar' % (
                opt.source_domain, opt.target_domain))
            elif opt.data_name == 'mini-Imagenet':
                model_path = os.path.join(root_, 'mini-imagenet_ycm_%s_to_%s_conv64f_best.pth.tar' % (
                opt.source_domain, opt.target_domain))
            elif opt.data_name == 'officehome':
                model_path = os.path.join(root_, 'officehome_%s_to_%s_conv64f_best.pth.tar' % (
                opt.source_domain, opt.target_domain))
            else:
                raise ValueError('Error dataset name.')
            model_dict = model.method.gen.state_dict()
            ckpt = torch.load(model_path, map_location='cuda:0')
            pretrained_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            a_dict = {}
            for k in pretrained_dict:
                new_k = k.replace('module.', '')
                a_dict[new_k] = pretrained_dict[k]
                # pretrained_dict.pop(k)
            pretrained_dict = a_dict
            # print(pretrained_dict.keys())
            # exit()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            if len(pretrained_dict) == 0:
                raise ValueError('Error ckpt.')
            model_dict.update(pretrained_dict)
            model.method.gen.load_state_dict(model_dict)
            print('load pretrained model from %s successly.' % root_)
        elif 'Baseline' in opt.metric:
            root_ = '/home/hsq/Projects/rotate_pretrained'
            if opt.data_name == 'visda':
                model_path = os.path.join(root_, 'visda_%s_to_%s_normal_resnet_best.pth.tar' % (
                    opt.source_domain, opt.target_domain))
            elif opt.data_name == 'mini-Imagenet':
                model_path = os.path.join(root_, 'mini-imagenet_ycm_%s_to_%s_conv64f_best.pth.tar' % (
                    opt.source_domain, opt.target_domain))
            elif opt.data_name == 'officehome':
                model_path = os.path.join(root_, 'officehome_%s_to_%s_conv64f_best.pth.tar' % (
                    opt.source_domain, opt.target_domain))
            else:
                raise ValueError('Error dataset name.')
            model_dict = model.method.gen.state_dict()
            ckpt = torch.load(model_path, map_location='cuda:0')
            pretrained_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            a_dict = {}
            for k in pretrained_dict:
                new_k = k.replace('module.', '')
                a_dict[new_k] = pretrained_dict[k]
                # pretrained_dict.pop(k)
            pretrained_dict = a_dict
            # print(pretrained_dict.keys())
            # exit()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            if len(pretrained_dict) == 0:
                raise ValueError('Error ckpt.')
            model_dict.update(pretrained_dict)
            model.method.gen.load_state_dict(model_dict)
            print('load pretrained model from %s successly.' % root_)

        elif opt.backbone == 'ResNet12':
            # ckpt = torch.load('pretrained_model/mini_resnet_model_3pool_epoch161.pth.tar')

            def load_model(model, dir):
                model_dict = model.state_dict()
                print('loading model from :', dir)
                pretrained_dict = torch.load(dir)['params']
                if 'encoder' in list(pretrained_dict.keys())[0]:  # load from a parallel meta-trained model
                    if 'module' in list(pretrained_dict.keys())[0]:
                        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items()}
                    else:
                        pretrained_dict = {k: v for k, v in pretrained_dict.items()}
                else:
                    pretrained_dict = { k: v for k, v in
                                       pretrained_dict.items()}  # load from a pretrained model
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                # print(pretrained_dict)
                model_dict.update(pretrained_dict)  # update the param in encoder, remain others still
                model.load_state_dict(model_dict)
            if opt.data_name == 'officehome':
                model_dict = model.method.gen.state_dict()
                ckpt = torch.load('./pretrained_model/officehome_resnet12_model_3pool_epoch90.pth.tar')
                pretrained_dict = ckpt['state_dict']
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                model.method.gen.load_state_dict(model_dict)
            if opt.data_name == 'visda':
                model_dict = model.method.gen.state_dict()
                if opt.combine_uda_fsl:
                    # e_dir = '/home/hsq/Projects/MCD_DA/classification/pretrained/MCD_%s_to_%s_resnet.pth.tar'%(opt.source_domain, opt.target_domain, )
                    e_dir = 'ADDA_pretrained/%s2%s-ADDA-target-encoder-final.pt'%(opt.source_domain, opt.target_domain, )
                    # e_dir = '/home/hsq/Projects/FSUDA_lds_spatial/ADDA_pretrained/%s2%s-ADDA-target-encoder-final.pt'% (opt.source_domain, opt.target_domain, )
                    ckpt = torch.load(e_dir)
                    n_ckpt = {}
                    for k in ckpt.keys():
                        n_k = k.replace('module.', '')
                        n_ckpt[n_k] = ckpt[k]
                    ckpt = {'state_dict': n_ckpt}
                    print('load from %s' % (e_dir))
                else:
                    ckpt = torch.load('./pretrained_model/visda_%s_resnet12_model_3pool.pth.tar' % opt.source_domain)
                pretrained_dict = ckpt['state_dict']
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                model.method.gen.load_state_dict(model_dict)
                if pretrained_dict.__len__() != 0:
                    print('load pretrained model success.')
                else:
                    print('load pretrained model failed.')
            elif opt.data_name == 'tiered-Imagenet':
                load_model(model.method.gen, 'pretrained_model/tieredImagenet_resnet12.pth')
            else:
                load_model(model.method.gen, 'pretrained_model/epoch-64.pth')
        else:
            if opt.data_name == 'visda':
                model_path = os.path.join('/home/hsq/Projects/pretrained_models', 'visda_%s_100epoch.pth.tar'%opt.source_domain)
            elif opt.data_name == 'mini-Imagenet':
                model_path = os.path.join('/home/hsq/Projects/pretrained_models', 'mini-Imagenet_%s_50epoch.pth.tar' % opt.source_domain)
            elif opt.data_name == 'officehome':
                model_path = os.path.join('/home/hsq/Projects/pretrained_models', 'officehome_%s_30epoch.pth.tar' % opt.source_domain)
            else:
                raise ValueError('Error dataset name.')
            model_dict = model.method.gen.state_dict()
            ckpt = torch.load(model_path)
            pretrained_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            if len(pretrained_dict) == 0:
                raise ValueError('Error ckpt.')
            model_dict.update(pretrained_dict)
            model.method.gen.load_state_dict(model_dict)
            print('load pretrained model success.')

            # model.method.gen.load_state_dict(ckpt['state_dict'])


    if len(opt.gpu) > 1:
        gid = range(opt.gpu.split(',').__len__())
        model.method.gen = nn.DataParallel(model.method.gen, [int(g) for g in gid],)
        model.method.imgtoclass = nn.DataParallel(model.method.imgtoclass, [int(g) for g in gid],)
        model.method.discri = nn.DataParallel(model.method.discri, [int(g) for g in gid],)
    model.cuda()
    model.eval()

    repeat_num = 5
    best_prec1 = 0.0
    total_accuracy = 0.0
    total_h = np.zeros(repeat_num)
    total_accuracy_vector = []
    for r in range(repeat_num):
        data_dir = opt.dataset_dir+'/'+opt.data_name
        testset = DataGenerator(mode='test', datasource=opt.data_name, data_dir=data_dir, imagesize=opt.imageSize,
                      episode=600, support_num=opt.shot_num, query_num=opt.query_num, split_PATH=opt.split_PATH, s2p=opt.s2p,
                                source_domain=opt.source_domain, target_domain=opt.target_domain, way_num=opt.way_num)
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=opt.testepisodeSize, shuffle=True,
            num_workers=int(opt.workers), drop_last=True, pin_memory=True
        )
        print('Testset: %d' % len(testset))
        print('dataset %s, with 5-way, %d-shot, %s metric.'%(opt.data_name, opt.shot_num, opt.metric))
        prec1, accuracies = meta_test(test_loader, model, r, best_prec1, opt, train='test')
        test_accuracy, h = mean_confidence_interval(accuracies)
        print("Test accuracy", test_accuracy, "h", h)
        total_accuracy += test_accuracy
        total_accuracy_vector.extend(accuracies)
        total_h[r] = h

    aver_accuracy, _ = mean_confidence_interval(total_accuracy_vector)
    print("Aver_accuracy:", aver_accuracy, "Aver_h", total_h.mean())
    return {"aver_accuracy": aver_accuracy, "aver_h": total_h.mean()}


def train(train_loader, model, epoch_index, opt):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ganloss = AverageMeter()
    classloss = AverageMeter()
    top1_m1 = AverageMeter()
    top1_m2 = AverageMeter()
    disc_acc = AverageMeter()

    end = time.time()

    for episode_index, (query_images, query_targets, query_modal, support_images, support_targets, query_global_targets) in tqdm(enumerate(train_loader)):

        # Measure data loading time
        data_time.update(time.time() - end)

        # Convert query and support images

        query_images = query_images.type(torch.FloatTensor)
        # input_var2 = [s[0].cuda() for s in support_images]
        input_var2 = support_images.cuda()

        input_var1 = query_images.cuda()
        query_modal = query_modal.cuda()
        query_targets = query_targets.cuda()
        query_global_targets = query_global_targets.cuda()
        loss_acc = model(query_x=input_var1, support_x=input_var2, query_y=query_targets, query_m=query_modal, global_label=query_global_targets)
        # print(model.method.imgtoclass.module.maps)

        # Measure accuracy and record loss
        ganloss.update(loss_acc['gan_loss'].item(), opt.query_num)
        classloss.update(loss_acc['class_loss'].item(), opt.query_num)
        top1_m1.update(loss_acc['class_acc'][0][0].item(), opt.query_num)
        # top1_m2.update(loss_acc['class_acc'][1][0].item(), query_images.size(0))
        top1_m2.update(-1, opt.query_num//2)
        disc_acc.update(loss_acc['disc_acc'][0].item(), opt.query_num)

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # ============== print the intermediate results ==============#
        if episode_index % opt.print_freq == 0 and episode_index != 0:
            print('Eposide-({0}): [{1}/{2}]\t'
                  'Time  ({batch_time.avg:.3f})\t'
                  'Data  ({data_time.avg:.3f})\t'
                  'CLoss ({loss.avg:.3f})\t'
                  'GLoss  ({ganloss.avg:.3f})\t'
                  'DAcc  ({disc_acc.avg:.3f})\t'
                  'Prec@1 ({top1_m1.avg:.3f}, {top1_m2.avg:.3f} -> {mean:.3f})'.format(
                epoch_index, episode_index, len(train_loader), batch_time=batch_time, data_time=data_time,
                loss=classloss,
                ganloss=ganloss, top1_m1=top1_m1, top1_m2=top1_m2, mean=(top1_m1.avg + top1_m2.avg) / 2, disc_acc=disc_acc))


@torch.no_grad()
def validate(val_loader, model, epoch_index, best_prec1, opt, train='val'):
    batch_time = AverageMeter()
    ganloss = AverageMeter()
    classloss = AverageMeter()
    top1_m1 = AverageMeter()
    top1_m2 = AverageMeter()
    # switch to evaluate mode
    model.eval()
    accuracies = []

    end = time.time()
    for episode_index, (query_images, query_targets, query_modal, support_images, support_targets, query_global_targets) in tqdm(enumerate(val_loader)):

        # Convert query and support images
        query_images = torch.squeeze(query_images.type(torch.FloatTensor))
        support_images = support_images
        input_var1 = query_images.cuda()
        # input_var2 = [s[0].cuda() for s in support_images]
        input_var2 = support_images.cuda()

        # Calculate the output
        query_modal = query_modal.cuda()
        query_targets = query_targets.cuda()
        loss_acc = model(query_x=input_var1, support_x=input_var2, query_y=query_targets, query_m=query_modal, train=train)

        # Measure accuracy and record loss
        ganloss.update(loss_acc['gan_loss'].item(), query_images.size(0))
        classloss.update(loss_acc['class_loss'].item(), query_images.size(0))
        top1_m1.update(loss_acc['class_acc'][0][0].item(), query_images.size(0))
        top1_m2.update(loss_acc['class_acc'][1][0].item(), query_images.size(0))

        accuracies.append(loss_acc['class_acc'])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # ============== print the intermediate results ==============#
        if episode_index % opt.print_freq == 0 and episode_index != 0:
            print('Test-({0}): [{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Prec@1 {top1_m1.val:.3f} ({top1_m1.avg:.3f}, {top1_m2.avg:.3f} -> {mean:.3f})'.format(
                epoch_index, episode_index, len(val_loader), batch_time=batch_time, loss=classloss,
                top1_m1=top1_m1, top1_m2=top1_m2, mean=(top1_m1.avg+top1_m2.avg)/2))

            'Eposide-({0}): [{1}/{2}]\t'
            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
            'classLoss {loss.val:.3f} ({loss.avg:.3f})\t'
            'genLoss {ganloss.val:.3f} ({ganloss.avg:.3f})\t'
            'Prec@1 {top1_m1.val:.3f} ({top1_m1.avg:.3f}, {top1_m2.avg:.3f} -> {mean:.3f})'



    print(' * Prec@1 {mean:.3f} Best_prec1 {best_prec1:.3f}'.format(mean=(top1_m1.avg+top1_m2.avg)/2, best_prec1=best_prec1))

    return top1_m2.avg, accuracies


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


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


# ======================================== Settings of path ============================================



# ========================================== Model Config ===============================================
def main(opt):

    print('loss weight is:', opt.loss_weight)
    if not os.path.exists(opt.outf):
        os.makedirs(opt.outf)

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # save the opt and results to a txt file
    print(opt)
    ngpu = int(opt.ngpu)
    global best_prec1, epoch_index
    best_prec1 = 0
    epoch_index = 0

    model = hyper_model.Trainer(opt=opt)
    # model = hyper_model_178.Trainer(lr=opt.lr, neighbor_k=opt.neighbor_k, beta1=opt.beta1, metric=opt.metric, discri_name=opt.discri, decay_intermin=opt.interim,
    #                            decay_rate=opt.decay_rate, soft_label=opt.softlabel, target_topk=opt.target_topk, usa=opt.usa, loss_weight=opt.loss_weight, use_gda=opt.use_gda, cov_align=opt.cov_align, backbone=opt.backbone)

    model.method.imgtoclass.writer = writer

    if opt.pretrained and opt.metric not in ('MCD', 'DWT', 'DANN'):
        if opt.backbone == 'ResNet12':
            # ckpt = torch.load('pretrained_model/mini_resnet_model_3pool_epoch161.pth.tar')

            def load_model(model, dir):
                model_dict = model.state_dict()
                print('loading model from :', dir)
                pretrained_dict = torch.load(dir)['params']
                if 'encoder' in list(pretrained_dict.keys())[0]:  # load from a parallel meta-trained model
                    if 'module' in list(pretrained_dict.keys())[0]:
                        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items()}
                    else:
                        pretrained_dict = {k: v for k, v in pretrained_dict.items()}
                else:
                    pretrained_dict = { k: v for k, v in
                                       pretrained_dict.items()}  # load from a pretrained model
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                # print(pretrained_dict)
                model_dict.update(pretrained_dict)  # update the param in encoder, remain others still
                model.load_state_dict(model_dict)
            if opt.rotate:
                root_ = '/home/hsq/Projects/rotate_pretrained'
                if opt.data_name == 'visda':
                    model_path = os.path.join(root_, 'visda_%s_to_%s_rotate_resnet_best.pth.tar' % (
                    opt.source_domain, opt.target_domain))
                elif opt.data_name == 'mini-Imagenet':
                    model_path = os.path.join(root_, 'mini-imagenet_ycm_%s_to_%s_conv64f_best.pth.tar' % (
                    opt.source_domain, opt.target_domain))
                elif opt.data_name == 'officehome':
                    model_path = os.path.join(root_, 'officehome_%s_to_%s_conv64f_best.pth.tar' % (
                    opt.source_domain, opt.target_domain))
                else:
                    raise ValueError('Error dataset name.')
                model_dict = model.method.gen.state_dict()
                ckpt = torch.load(model_path, map_location='cuda:0')
                pretrained_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
                a_dict = {}
                for k in pretrained_dict:
                    new_k = k.replace('module.', '')
                    a_dict[new_k] = pretrained_dict[k]
                    # pretrained_dict.pop(k)
                pretrained_dict = a_dict
                # print(pretrained_dict.keys())
                # exit()
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                if len(pretrained_dict) == 0:
                    raise ValueError('Error ckpt.')
                model_dict.update(pretrained_dict)
                model.method.gen.load_state_dict(model_dict)
                print('load pretrained rotate model from %s successly.' % root_)
            elif opt.data_name == 'officehome':
                model_dict = model.method.gen.state_dict()
                ckpt = torch.load('./pretrained_model/officehome_resnet12_model_3pool_epoch90.pth.tar')
                pretrained_dict = ckpt['state_dict']
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                model.method.gen.load_state_dict(model_dict)
            elif opt.data_name == 'visda':
                model_dict = model.method.gen.state_dict()
                if opt.combine_uda_fsl:
                    # e_dir = '/home/hsq/Projects/MCD_DA/classification/pretrained/MCD_%s_to_%s_resnet.pth.tar'%(opt.source_domain, opt.target_domain, )
                    e_dir = 'ADDA_pretrained/%s2%s-ADDA-target-encoder-final.pt'%(opt.source_domain, opt.target_domain, )
                    # e_dir = '/home/hsq/Projects/FSUDA_lds_spatial/ADDA_pretrained/%s2%s-ADDA-target-encoder-final.pt'% (opt.source_domain, opt.target_domain, )
                    ckpt = torch.load(e_dir)
                    n_ckpt = {}
                    for k in ckpt.keys():
                        n_k = k.replace('module.', '')
                        n_ckpt[n_k] = ckpt[k]
                    ckpt = {'state_dict': n_ckpt}
                    print('load from %s' % (e_dir))
                    
                    
                else:

                    ckpt = torch.load('/home/hsq/NAS/Projects/pretrained_model/visda_%s_resnet12_model_3pool.pth.tar' % opt.source_domain)
                pretrained_dict = ckpt['state_dict']
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                model_dict.update(pretrained_dict)
                model.method.gen.load_state_dict(model_dict)
                if pretrained_dict.__len__() != 0:
                    print('load pretrained model success.')
                else:
                    print('load pretrained model failed.')
            elif opt.data_name == 'tiered-Imagenet':
                load_model(model.method.gen, '/home/hsq/NAS/Projects/pretrained_model/tieredImagenet_resnet12.pth')
            else:
                load_model(model.method.gen, '/home/hsq/NAS/Projects/pretrained_model/epoch-64.pth')

        else:
            if opt.data_name == 'visda':
                model_path = os.path.join('/home/hsq/NAS/Projects/pretrained_models', 'visda_%s_100epoch.pth.tar'%opt.source_domain)
            elif opt.data_name == 'mini-Imagenet':
                model_path = os.path.join('/home/hsq/NAS/Projects/pretrained_models', 'mini-Imagenet_%s_50epoch.pth.tar' % opt.source_domain)
            elif opt.data_name == 'officehome':
                model_path = os.path.join('/home/hsq/NAS/Projects/pretrained_models', 'officehome_%s_30epoch.pth.tar' % opt.source_domain)

            model_dict = model.method.gen.state_dict()
            ckpt = torch.load(model_path)
            pretrained_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            if len(pretrained_dict) == 0:
                raise ValueError('Error ckpt.')
            model_dict.update(pretrained_dict)
            model.method.gen.load_state_dict(model_dict)
            print('load pretrained model success.')

            # model.method.gen.load_state_dict(ckpt['state_dict'])

    model.cuda()

    # define loss function (criterion) and optimizer


    # optionally resume from a checkpoint

    if len(opt.gpu) > 1:
        gpuid = range(opt.gpu.split(',').__len__())
        model.method.gen = nn.DataParallel(model.method.gen, [int(g) for g in gpuid],)
        model.method.imgtoclass = nn.DataParallel(model.method.imgtoclass, [int(g) for g in gpuid],)
        model.method.discri = nn.DataParallel(model.method.discri, [int(g) for g in gpuid],)
        # model.method.cluster = nn.DataParallel(model.method.cluster, [int(g) for g in gpuid])
    model.eval()

    # print the architecture of the network
    # print(model)
    # print(model, file=F_txt)

    # ======================================== Training phase ===============================================
    print('\n............Start training............\n')
    start_time = time.time()

    for epoch_item in range(opt.pretrain, opt.epochs):
        if 'Baseline' in opt.metric or opt.metric in ('MCD', 'DWT', 'DANN', 'MCDR2D2', 'ADDA') or opt.finetune:
            break
        if epoch_item >= 0:
            filename = os.path.join(opt.outf, 'epoch_%d.pth.tar' % epoch_item)
            save_checkpoint(
                {
                    'epoch_index': epoch_item,
                    'arch': opt.basemodel,
                    'state_dict': model.method.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer': model.g_optimizer.state_dict(),
                }, filename)
        print('===================================== Epoch %d =====================================' % epoch_item)
        # print('===================================== Epoch %d =====================================' % epoch_item,
        #       file=F_txt)
        model.adjust_learning_rate(epoch_item)

        # ======================================= Folder of Datasets =======================================
        data_dir = opt.dataset_dir+'/'+opt.data_name
        trainset = DataGenerator(mode='train', datasource=opt.data_name,  data_dir=data_dir, imagesize=opt.imageSize,
            episode=opt.episode_train_num, support_num=opt.shot_num, query_num=opt.query_num, split_PATH=opt.split_PATH, s2p=opt.s2p,
                                 source_domain=opt.source_domain, target_domain=opt.target_domain, way_num=opt.way_num)

        valset = DataGenerator(mode='val', datasource=opt.data_name, data_dir=data_dir, imagesize=opt.imageSize,
                      episode=opt.episode_val_num, support_num=opt.shot_num, query_num=opt.query_num, split_PATH=opt.split_PATH, s2p=opt.s2p,
                               source_domain=opt.source_domain, target_domain=opt.target_domain, way_num=opt.way_num)

        testset = DataGenerator(mode='test', datasource=opt.data_name, data_dir=data_dir, imagesize=opt.imageSize,
                      episode=opt.episode_test_num, support_num=opt.shot_num, query_num=opt.query_num, split_PATH=opt.split_PATH, s2p=opt.s2p,
                                source_domain=opt.source_domain, target_domain=opt.target_domain, way_num=opt.way_num)

        print('Trainset: %d' % len(trainset))
        print('Valset: %d' % len(valset))
        print('Testset: %d' % len(testset))

        # ========================================== Load Datasets =========================================
        train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=opt.episodeSize, shuffle=False,
            num_workers=int(opt.workers), drop_last=False, pin_memory=False
        )
        val_loader = torch.utils.data.DataLoader(
            valset, batch_size=opt.testepisodeSize, shuffle=False,
            num_workers=int(opt.workers), drop_last=False, pin_memory=False
        )
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=opt.testepisodeSize, shuffle=False,
            num_workers=int(opt.workers), drop_last=False, pin_memory=False
        )

        # ============================================ Training ===========================================
        # Fix the parameters of Batch Normalization after 10000 episodes (1 epoch)
        if opt.metric in ('DeepEMD', 'MetaBaseline', 'DSN') or (opt.on_bn and epoch_item < opt.bn_epoch):
            # model.method.gen.train()
            model.train()
        else:
            model.eval()

        # Train for 10000 episodes in each epoch
        # model.eval()
        train(train_loader, model, epoch_item, opt)
        print('dataset %s, with 5-way, %d-shot, %s metric.'%(opt.data_name, opt.shot_num, opt.metric))
        model.eval()
        # =========================================== Evaluation ==========================================
        print('============ Validation on the val set ============')
        prec1, _ = validate(val_loader, model, epoch_item, best_prec1, opt, train='val')

        # record the best prec@1 and save checkpoint

        # Testing Prase
        print('============ Testing on the test set ============')
        _, _ = validate(test_loader, model, epoch_item, best_prec1, opt, train='test')

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        # save the checkpoint
        if is_best:
            save_checkpoint(
                {
                    'epoch_index': epoch_item,
                    'arch': opt.basemodel,
                    'state_dict': model.method.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer': model.g_optimizer.state_dict(),
                }, os.path.join(opt.outf, 'model_best.pth.tar'))

    print('............Training is end............')
    # ============================================ Training End ==============================================================

    print('............Now meta testing............')
    res = test(opt)
    res['val_acc'] = best_prec1
    print(res)

    return res







# main(opt)






