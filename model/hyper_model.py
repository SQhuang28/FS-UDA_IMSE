import os

import torchvision.utils

from model.discriminator import Discriminator, DiscriminatorLD
from model.backbone import ResNet, Conv64F
from torch import nn
from model.imse import *
import torch
import torch.optim as optim
from utils import cal_cov_loss, init_weights, cal_FSL_cov_loss, cal_cov_classloss_covdist
from model.losses import *
from tensorboardX import SummaryWriter

class FSUDA_METHOD(nn.Module):
    def __init__(self, opt):
        super(FSUDA_METHOD, self).__init__()
        self.discri_name = opt.discri
        self.soft_label = opt.softlabel
        self.cov_align = opt.cov_align
        self.soft_label_loss = UDASoftLabel_MultiScale_V2(topk=4)
        self.dper = None
        self.opt = opt
        # print(self.opt.record_path, self.opt.source_domain, self.opt.target_domain); exit()
        self.target_topk = opt.target_topk
        self.cov_weight = opt.cov_weight
        writer_path = "{}/{}_to_{}".format(self.opt.record_path, self.opt.source_domain, self.opt.target_domain)
        self.writer = SummaryWriter(writer_path)
        self.global_step = 0
        self.construct()

    def construct(self):
        backbone = self.opt.backbone
        if backbone == 'ResNet12':
            if self.opt.ld_num == 25:
                last_layer = [False, False, False, False]
            elif self.opt.ld_num == 441:
                last_layer = [False, False, True, True]
            else:
                last_layer = [False, False, False, True]
            self.gen = ResNet(last_layer=last_layer)
            self.discri = Discriminator(640)
        elif backbone == 'Conv64F':
            if "IMSE" in self.opt.metric:
                last_layer = [True, True, True, False]
            else:
                last_layer = [True, True, True, True]
            self.gen = Conv64F(leakyrelu=True, downsample=last_layer)
            self.discri = Discriminator(64)
        else:
            raise ValueError("Wrong backbone, no such backbone {}, please check your backbone configuration!".format(backbone))
        if self.opt.metric == "MELDA":
            self.discri = MELDA(discriminator=True)
        if "IMSE" in self.opt.metric:
            self.discri = DiscriminatorLD(640) if backbone == "ResNet12" else DiscriminatorLD(64)
        nm_space = vars(self.opt)
        kwargs = eval(self.opt.metric).__init__.__code__.co_varnames
        keys_ = {k: nm_space[k] for k in nm_space if k in kwargs}
        self.imgtoclass = eval(self.opt.metric)(**keys_)
        init_weights(self.gen, init_type='kaiming')
        init_weights(self.discri, init_type='kaiming')
        init_weights(self.imgtoclass, init_type='kaiming')
        print('===' * 10)
        print('backbone is:%s' % self.gen.__class__.__name__)
        print('discriminator is:%s' % self.discri.__class__.__name__)
        print('metric method is:%s' % self.imgtoclass.__class__.__name__)
        print('feature align method is:%s' % self.soft_label_loss.__class__.__name__)



    def forward(self, input):
        query_y, query, support, train = input
        # obtain deep-features
        q, S = self.forward_gen(query, support)
        B, C, h, w = q.size()
        other_results = {}
        if train == 'train':
            # only use the source domain query sample during training
            trans_source = q.reshape([2, -1, C, h, w])[0]
            trans_target = q.reshape([2, -1, C, h, w])[1]
            tmp_q = q.reshape([2, -1, C, h, w])[0]
            # tmp_q2 = q.reshape([2, -1, C, h, w])[1]
        else:
            trans_source = q.reshape([2, -1, C, h, w])[0]
            trans_target = q.reshape([2, -1, C, h, w])[1]
            tmp_q = q
        new_S = S
        # get the domain prediction from the discriminator
        if 'MELDA' == self.opt.metric:
            modal_predict = self.discri(torch.cat([trans_source, trans_target], 0), new_S)
        else:
            modal_predict = self.discri(torch.cat([trans_source, trans_target], 0))
        # get the class prediction from the few-shot image-to-class (classifier)
        model_output = self.imgtoclass(tmp_q, new_S)
        model_output_target_domain = self.imgtoclass(trans_target, new_S)
        class_predict = model_output["logits"]
        if "cov_list" in model_output:
            cov_source, cov_target = model_output["cov_list"], model_output_target_domain["cov_list"]
            cov_loss= cal_cov_loss(cov_source, cov_target)
            other_results["cov_loss"] = cov_loss
        if "masks" in model_output:
            masks = model_output["masks"]
            entropy = -(torch.log(masks+1e-7)).mean()
            other_results["mask_entropy"] = entropy
        if "cov_list" in model_output:
            if model_output["cov_list"][0].size()[0] == 75:
                self.writer.add_histogram("SP(source)", torch.cat(model_output["cov_list"], 0), global_step=self.global_step)
                self.writer.add_histogram("SP(target)", torch.cat(model_output_target_domain["cov_list"], 0), global_step=self.global_step)
                # self.writer.add_image("original images", torchvision.utils.make_grid(support[0].view(-1, 3, 84, 84), nrow=5, normalize=False), global_step=self.global_step)
                for j, (sc, tc) in enumerate(zip(model_output["cov_list"], model_output_target_domain["cov_list"])):
                    n, _ = sc.size()
                    sc = torchvision.utils.make_grid(sc.view(n, -1, 10, 10)[:10], nrow=10, normalize=True)
                    tc = torchvision.utils.make_grid(tc.view(n, -1, 10, 10)[:10], nrow=10, normalize=True)
                    self.writer.add_image("SP_C{}(source)".format(j), sc, global_step=self.global_step)
                    self.writer.add_image("SP_C{}(target)".format(j), tc, global_step=self.global_step)
            other_results["cov_list"] = model_output["cov_list"]
        if "sps_loss" in model_output:
            other_results["sps_loss"] = model_output["sps_loss"]
            if self.opt.DA:
                other_results["sps_loss"] = (other_results["sps_loss"] + model_output_target_domain['sps_loss'])*.5
        if "mining_info" in model_output:
            labeled_query_num = model_output["mining_info"]["ori_logits"].size(0)
            query_y = query_y[:labeled_query_num]
            query_info_ori = model_output["mining_info"]["ori_logits"]
            query_info_msk = model_output["mining_info"]["msked_logits"]
            other_results["msk_info_loss"] = eval(self.opt.mask_loss_func)(query_info_ori, query_info_msk, query_y, margin=self.opt.mask_margin)
        if "mask_divergence" in model_output:
            other_results["mask_divergence"] = model_output["mask_divergence"]

        if self.opt.softlabel:
            other_results['softloss'] = self.soft_label_loss(q, S)
        self.global_step += 1
        if "MaxCascade" in self.opt.metric:
            if len(self.opt.gpu.split(',')) > 1:
                gaussian_layers = self.imgtoclass.module.gaussian_layers
            else:
                gaussian_layers = self.imgtoclass.gaussian_layers
            for k in range(len(gaussian_layers)):
                sigmas = gaussian_layers[k].sigma_parameters
                gaussian_layers[k].get_gauss_kernel()
                kernel_size = gaussian_layers[k].gaussian_kernels.size()
                gaussian_kernel = gaussian_layers[k].gaussian_kernels.view(kernel_size[0], 1, *kernel_size[-2:])
                gaussian_kernel = gaussian_kernel/gaussian_kernel.view(kernel_size[0], -1).max(dim=-1, keepdim=True)[0].view(kernel_size[0], 1, 1, 1)
                kernel_grid = torchvision.utils.make_grid(gaussian_kernel, nrow=gaussian_layers[k].gaussian_kernels.size()[0])*254
                self.writer.add_image("gaussian_kernel_{}".format(k), kernel_grid, global_step=self.global_step)
                self.writer.add_histogram("gaussian_layer_{}".format(k), sigmas, global_step=self.global_step)
        if "IMSE_MultiConv" in self.opt.metric:
            if len(self.opt.gpu.split(',')) > 1:
                gaussian_layers = self.imgtoclass.module.gaussian_layers
            else:
                gaussian_layers = self.imgtoclass.gaussian_layers
            for k in range(len(gaussian_layers)):
                filter = gaussian_layers[k].kernel.weight
                n, m, _, h, w = filter.size()
                filter = filter.view(n*m, -1, h, w)
                filter_grid = torchvision.utils.make_grid(filter, nrow=n)
                self.writer.add_image("conv_kernel_{}".format(k), filter_grid, global_step=self.global_step)
        return modal_predict, class_predict, other_results




    def forward_gen(self, query, support):
        # return features
        query = query.squeeze()
        try:
            assert len(query.size()) == 4
        except:
            print(query.size()); exit()
        query_num = query.size(0)
        if self.imgtoclass.__class__.__name__ == 'DSN' and self.opt.shot_num == 1:
            support = [_ for _ in support[0]]
            support = torch.stack(support, 0).squeeze()
            support_tmp = support.mean(1, True).expand_as(support)
            support = torch.cat([support, support_tmp], 0)
            datas = torch.cat([query, support])
            datas = self.gen(datas)
            q, S = datas[:query_num], datas[query_num:]
            S = self.gen(support).reshape(2, support_tmp.size(0), -1).permute(1, 0, 2)
        elif 'MELDA' in self.opt.metric:
            # print(self.imgtoclass.lambda_rr.l)
            support = support[0]
            way_num, shot_num, C, h, w = support.size()[1:]
            support_tmp = []
            for domain in support:
                for sample in domain:
                    support_tmp.append(sample)
            support = torch.stack(support_tmp, 0).view(-1, C, h, w)
            datas = torch.cat([query, support])
            datas = self.gen(datas)
            q, S = datas[:query_num], datas[query_num:]
            S = S.view(2, way_num, shot_num, -1, *S.size()[-2:]).permute(1, 0, 2, 3, 4, 5).contiguous().view(way_num, shot_num*2, -1, *S.size()[-2:])
            S = S.repeat(1, torch.cuda.device_count(), 1, 1, 1)
        else:
            support = [_ for _ in support[0]]
            way_num, shot_num, C, h, w = len(support), *support[0].size()
            support = torch.stack(support, 0).view(-1, C, h, w)
            datas = torch.cat([query, support])
            datas = self.gen(datas)
            q, S = datas[:query_num], datas[query_num:]
            S = S.view(way_num, shot_num, -1, *S.size()[-2:]).repeat(1, torch.cuda.device_count(), 1, 1, 1)
        S = [_ for _ in S]
        return q, S


class Trainer(nn.Module):
    def __init__(self, opt):
        super(Trainer, self).__init__()
        self.modallossfn = nn.CrossEntropyLoss()
        self.classlossfn = nn.CrossEntropyLoss()
        self.discri_name = opt.discri
        self.lr = opt.lr
        self.dlr = opt.lr*100
        self.ilr = opt.ilr
        self.globallr = 0.01
        self.method = None
        self.loss_weight = opt.loss_weight
        self.deacy_rate = opt.decay_rate
        self.decay_interim = opt.interim
        self.metric = opt.metric
        self.ld_num = opt.ld_num
        self.soft_weight = opt.soft_weight
        self.cov_align = opt.cov_align
        self.opt = opt
        if self.opt.metric == 'DeepEMD' or self.opt.metric == 'MetaBaseline':
            self.opt.DA = False
        if self.method == None:
            self.method = FSUDA_METHOD(opt)
            self.method.cuda()
        param_list = [{'params': self.method.gen.parameters(), 'weight_decay':1e-4}]
        if self.method.imgtoclass.parameters() is not None:
            print('optimize the imagetoclass.')
            param_list.append({'params': self.method.imgtoclass.parameters(), 'lr': self.ilr, 'weight_decay':1e-4})
        if self.opt.SGD:
            self.g_optimizer = optim.SGD(param_list, lr=self.lr, momentum=0.9, nesterov=True)
        else:
            self.g_optimizer = optim.Adam(param_list, lr=self.lr, betas=(opt.beta1, 0.9), weight_decay=1e-4)
        # print((param_list)); exit()
        self.d_optimizer = optim.Adam(self.method.discri.parameters(), lr=self.dlr, betas=(opt.beta1, 0.9))

        print('===' * 10)
        print('adv weight is:%.4f' % self.loss_weight)
        print('soft weight is:%.2f' % self.soft_weight)
        print('cov weight is:%.4f' % self.method.cov_weight)
        print('===' * 10)


    def adjust_learning_rate(self, epoch_num):
        """Sets the learning rate to the initial LR decayed by 0.05 every 10 epochs"""
        lr = self.lr * (self.deacy_rate ** (epoch_num // self.decay_interim))
        print("learning rate is: %f" % lr)
        optimizer = self.g_optimizer
        for i, param_group in enumerate(optimizer.param_groups):
            if len(optimizer.param_groups) > 1 and i == len(optimizer.param_groups) - 1:
                param_group['lr'] = self.ilr * (self.deacy_rate ** (epoch_num // (self.decay_interim)))
            else:
                param_group['lr'] = self.lr * (self.deacy_rate ** (epoch_num // self.decay_interim))
        optimizer = self.d_optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.dlr * (self.deacy_rate ** (epoch_num // self.decay_interim))


    def accuracy(self, output, target, topk=(1,)):
        """Computes the precaccuracy(output, target, topk=(1,3))ision@k for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))
            res = []
            for k in topk:
                correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def exec_train(self, inp, train):
        all_loss_g = 0.
        all_loss_cls = 0.
        all_acc_cls = 0.
        all_acc_discri = 0.
        all_loss_d = 0.
        for batch_idx in range(self.opt.episodeSize):
            if train == "train":
                query_x, support_x, query_y, query_m = [d[batch_idx].unsqueeze(0) for j, d in enumerate(inp) if j != 4]
            else:
                if batch_idx > 0: break
                query_x, support_x, query_y, query_m = [d for j, d in enumerate(inp) if j != 4]
            support_x = support_x[:, 0] if not "MELDA" in self.opt.metric else support_x
            # process the label data
            query_y = query_y.t().squeeze()
            query_m = query_m.t().squeeze()
            if self.discri_name == 'LD':
                # 'LD' uses all local features, so reproduce the domain label here.
                query_m = query_m.repeat([self.ld_num, 1]).t().reshape([1, -1]).squeeze()
            model_predic, class_predic, other_results = self.method((query_y, query_x, support_x, train))
            if train=='train':
                # compute the top-1 accuracy.
                query_y = query_y.reshape([2, -1])[0]
                class_acc = self.accuracy(class_predic, query_y, topk=(1, ))
                classify_acc = [class_acc, class_acc]
                acc_specific = 0
            else:
                tmp_class_predic = torch.reshape(class_predic, [2, -1, self.opt.way_num])
                tmp_query_y = torch.reshape(query_y, [2, -1])
                classify_acc = [self.accuracy(tmp_class_predic[0], tmp_query_y[0], topk=(1, )), self.accuracy(tmp_class_predic[1], tmp_query_y[1], topk=(1, ))]
                acc_specific = {'pred':[tmp_class_predic[0].argmax(-1), tmp_class_predic[1].argmax(-1)],
                                'gt':[tmp_query_y[0], tmp_query_y[1]]}
            # this label used for train the generator in adversarial training strategy.
            confusion_model_label = torch.ones_like(query_m)-query_m
            # get classification CE loss, the 'temperature' see TADAM.
            classify_loss = self.classlossfn(class_predic*self.opt.temperature, query_y)
            all_loss_cls += classify_loss
            if train=='train' and self.opt.DA:
                # Training phase for the FSUDA.
                g_loss = self.classlossfn(model_predic, confusion_model_label)
                d_loss = self.modallossfn(model_predic, query_m)
                cgloss = self.loss_weight * classify_loss + g_loss*(1-self.loss_weight)
                if "cov_loss" in other_results and self.opt.cov_align:
                    cgloss += self.opt.cov_weight*other_results["cov_loss"]
                if "msk_info_loss" in other_results:
                    cgloss += other_results["msk_info_loss"]*self.opt.mask_info_weight
                if "mask_divergence" in other_results:
                    cgloss += other_results["mask_divergence"][0] * self.opt.mask_info_weight
                if "softloss" in other_results:
                    cgloss += other_results["softloss"]*self.opt.soft_weight

                all_loss_g += cgloss
                all_loss_d += d_loss
                if batch_idx == self.opt.episodeSize - 1:
                    all_loss_g = all_loss_g/self.opt.episodeSize
                    all_loss_d = all_loss_d/self.opt.episodeSize
                    if self.opt.metric != "MELDA":
                        self.d_optimizer.zero_grad()
                        all_loss_d.backward(retain_graph=True)
                    self.g_optimizer.zero_grad()
                    cgloss.backward()
                    grads = torch.cat([_.view(-1)  for x in self.g_optimizer.param_groups[0]['params'] for _ in x.grad ], -1)
                    # avoid the NaN during the updating.
                    if True not in torch.isnan(grads):
                        self.d_optimizer.step()
                        self.g_optimizer.step()
                    else:
                        print("skip the BP update.")
                # discriminator prediction accuracies.
                modal_acc = self.accuracy(model_predic, query_m, topk=(1,))
            elif self.opt.DA:
                # Testing phase for the FSUDA.
                g_loss = self.classlossfn(model_predic, confusion_model_label)
                d_loss = self.modallossfn(model_predic, query_m)
                modal_acc = self.accuracy(model_predic, query_m, topk=(1,))
            elif train=='train':
                # Training phase for the FSL.
                cgloss = classify_loss
                # Computing the L1 term.
                if "mask_entropy" in other_results:
                    cgloss += other_results["mask_entropy"]*self.opt.mask_entropy_weight
                if "cov_list" in other_results:
                    cov_loss = cal_FSL_cov_loss(other_results["cov_list"])*self.opt.cov_weight
                    cgloss += cov_loss
                if "msk_info_loss" in other_results:
                    cgloss += other_results["msk_info_loss"]*self.opt.mask_info_weight
                if "mask_divergence" in other_results:
                    cgloss += other_results["mask_divergence"][0] * self.opt.mask_info_weight
                if "sps_loss" in other_results:
                    cgloss += other_results["sps_loss"]*self.opt.sploss_weight
                # reg_l1 = 0
                # if self.method.imgtoclass.parameters():
                #     for params in self.method.imgtoclass.parameters():
                #         reg_l1 += params.abs().sum()
                self.g_optimizer.zero_grad()
                # (reg_l1*1e-5+cgloss).backward()
                (cgloss).backward()
                # torch.nn.utils.clip_grad_value_(self.method.gen.parameters(), 5)
                self.g_optimizer.step()
                g_loss = d_loss = modal_acc = cgloss.unsqueeze(0)
            else:
                cgloss = classify_loss
                g_loss = d_loss = modal_acc = cgloss.unsqueeze(0)

        return {'class_loss': classify_loss, 'gan_loss':g_loss, 'd_loss':d_loss,
                'class_acc':classify_acc, 'disc_acc':modal_acc, 'acc_specific':acc_specific}

    def forward(self, query_x, support_x, query_y, query_m, train='train', global_label=None):
        query_y = query_y.cuda()
        inp = (query_x, support_x, query_y, query_m, global_label)
        loss_acc = self.exec_train(inp, train)
        return loss_acc

