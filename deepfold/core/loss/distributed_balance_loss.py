
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter, defaultdict

import pandas as pd

from deepfold.data.utils.ontology import Ontology

class ResampleLoss(nn.Module):
    """
    [Submitted on 10 Sep 2021 (v1), last revised 15 Oct 2021 (this version, v2)]
    Balancing Methods for Multi-label Text Classification with Long-Tailed Class Distribution
    Yi Huang, Buse Giledereli, Abdullatif Köksal, Arzucan Özgür, Elif Ozkirimli
    https://github.com/Roche/BalancedLossNLP
    """

    def __init__(self,
                 use_sigmoid=True,
                 loss_weight=1.0, reduction='mean', #"none", "mean" and "sum"
                 reweight_func=None,  # None, 'inv', 'sqrt_inv', 'rebalance', 'CB'
                 weight_norm=None, # None, 'by_instance', 'by_batch'
                 focal=dict(focal=True,alpha=0.5,gamma=2,),
                 map_param=dict(alpha=10.0,beta=0.2,gamma=0.1),
                 CB_loss=dict(CB_beta=0.9,CB_mode='average_w'),  #'by_class', 'average_n', 'average_w', 'min_n'
                 logit_reg=dict(neg_scale=5.0,init_bias=0.1),
                 class_freq=None,
                 train_num=None):
        super(ResampleLoss, self).__init__()
        assert use_sigmoid
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.cls_criterion = binary_cross_entropy
        # reweighting function
        self.reweight_func = reweight_func

        # normalization (optional)
        self.weight_norm = weight_norm

        # focal loss params
        self.focal = focal['focal']
        self.gamma = focal['gamma']
        self.alpha = focal['alpha'] # change to alpha

        # mapping function params
        self.map_alpha = map_param['alpha']
        self.map_beta = map_param['beta']
        self.map_gamma = map_param['gamma']

        # CB loss params (optional)
        self.CB_beta = CB_loss['CB_beta']
        self.CB_mode = CB_loss['CB_mode']

        self.class_freq = torch.from_numpy(np.asarray(class_freq)).float().cuda()
        self.num_classes = self.class_freq.shape[0]
        self.train_num = train_num # only used to be divided by class_freq
        # regularization params
        self.logit_reg = logit_reg
        self.neg_scale = logit_reg[
            'neg_scale'] if 'neg_scale' in logit_reg else 1.0
        init_bias = logit_reg['init_bias'] if 'init_bias' in logit_reg else 0.0
        self.init_bias = - torch.log(
            self.train_num / self.class_freq - 0.9999) * init_bias ########################## bug fixed https://github.com/wutong16/DistributionBalancedLoss/issues/8
        self.freq_inv = torch.ones(self.class_freq.shape).cuda() / self.class_freq
        self.propotion_inv = self.train_num / self.class_freq

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        weight = self.reweight_functions(label)
        cls_score, weight = self.logit_reg_functions(label.float(), cls_score, weight)
        
        if self.focal:
            logpt = self.cls_criterion(
                cls_score.clone(), label, weight=None, reduction='none',
                avg_factor=avg_factor)
            # pt is sigmoid(logit) for pos or sigmoid(-logit) for neg
            pt = torch.exp(-logpt)
            wtloss = self.cls_criterion(
                cls_score, label.float(), weight=weight, reduction='none')
            alpha_t = torch.where(label==1, self.alpha, 1-self.alpha)
            loss = alpha_t * ((1 - pt) ** self.gamma) * wtloss ####################### balance_param should be a tensor
            loss = reduce_loss(loss, reduction)             ############################ add reduction
        else:
            loss = self.cls_criterion(cls_score, label.float(), weight,
                                      reduction=reduction)
        loss = self.loss_weight * loss
        return loss

    def reweight_functions(self, label):
        if self.reweight_func is None:
            return None
        elif self.reweight_func in ['inv', 'sqrt_inv']:
            weight = self.RW_weight(label.float())
        elif self.reweight_func in 'rebalance':
            weight = self.rebalance_weight(label.float())
        elif self.reweight_func in 'CB':
            weight = self.CB_weight(label.float())
        else:
            return None

        if self.weight_norm is not None:
            if 'by_instance' in self.weight_norm:
                max_by_instance, _ = torch.max(weight, dim=-1, keepdim=True)
                weight = weight / max_by_instance
            elif 'by_batch' in self.weight_norm:
                weight = weight / torch.max(weight)

        return weight

    def logit_reg_functions(self, labels, logits, weight=None): 

        if not self.logit_reg:
            return logits, weight
        if 'init_bias' in self.logit_reg:
            logits += self.init_bias
        if 'neg_scale' in self.logit_reg:
            logits = logits * (1 - labels) * self.neg_scale  + logits * labels
            if weight is not None:
                weight = weight / self.neg_scale * (1 - labels) + weight * labels
        return logits, weight

    def rebalance_weight(self, gt_labels):
        repeat_rate = torch.sum( gt_labels.float() * self.freq_inv, dim=1, keepdim=True)
        pos_weight = self.freq_inv.clone().detach().unsqueeze(0) / repeat_rate
        # pos and neg are equally treated
        weight = torch.sigmoid(self.map_beta * (pos_weight - self.map_gamma)) + self.map_alpha
        return weight

    def CB_weight(self, gt_labels):
        if  'by_class' in self.CB_mode:
            weight = torch.tensor((1 - self.CB_beta)).cuda() / \
                     (1 - torch.pow(self.CB_beta, self.class_freq)).cuda()
        elif 'average_n' in self.CB_mode:
            avg_n = torch.sum(gt_labels * self.class_freq, dim=1, keepdim=True) / \
                    torch.sum(gt_labels, dim=1, keepdim=True)
            weight = torch.tensor((1 - self.CB_beta)).cuda() / \
                     (1 - torch.pow(self.CB_beta, avg_n)).cuda()
        elif 'average_w' in self.CB_mode:
            weight_ = torch.tensor((1 - self.CB_beta)).cuda() / \
                      (1 - torch.pow(self.CB_beta, self.class_freq)).cuda()
            weight = torch.sum(gt_labels * weight_, dim=1, keepdim=True) / \
                     torch.sum(gt_labels, dim=1, keepdim=True)
        elif 'min_n' in self.CB_mode:
            min_n, _ = torch.min(gt_labels * self.class_freq +
                                 (1 - gt_labels) * 100000, dim=1, keepdim=True)
            weight = torch.tensor((1 - self.CB_beta)).cuda() / \
                     (1 - torch.pow(self.CB_beta, min_n)).cuda()
        else:
            raise NameError
        return weight

    def RW_weight(self, gt_labels, by_class=True):
        if 'sqrt' in self.reweight_func:
            weight = torch.sqrt(self.propotion_inv)
        else:
            weight = self.propotion_inv
        if not by_class:
            sum_ = torch.sum(weight * gt_labels, dim=1, keepdim=True)
            weight = sum_ / torch.sum(gt_labels, dim=1, keepdim=True)
        return weight
    

def reduce_loss(loss, reduction):
    """Reduce loss as specified.
    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".
    Return:
        Tensor: Reduced loss tensor.
    """
    if reduction == 'none':
        return loss
    elif reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.
    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.
    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss


def binary_cross_entropy(pred,
                         label,
                         weight=None,
                         reduction='mean',
                         avg_factor=None):

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()

    loss = F.binary_cross_entropy_with_logits(
        pred, label.float(), weight, reduction='none')
    loss = weight_reduce_loss(loss, reduction=reduction, avg_factor=avg_factor)

    return loss

def statistic_terms(train_data_path):
    """get frequency dict from train file."""
    train_data = pd.read_pickle(train_data_path)
    cnt = Counter()
    for i, row in train_data.iterrows():
        for term in row['prop_annotations']:
            cnt[term] += 1
    print('Number of annotated terms:', len(cnt))
    sorted_by_freq_tuples = sorted(cnt.items(), key=lambda x: x[0])
    sorted_by_freq_tuples.sort(key=lambda x: x[1], reverse=True)
    freq_dict = {go: count for go, count in sorted_by_freq_tuples}
    return freq_dict

if __name__ == '__main__':
    import os
    data_path = '/share/home/niejianzheng/xbiome/datasets/protein/cafa3/'
    namespace = 'mfo'
    go_file = os.path.join(data_path, 'go_cafa3.obo')
    train_data_file = os.path.join(data_path, namespace,
                                   namespace + '_train_data.pkl')
    freq_dict = statistic_terms(train_data_file)
    freq_df = pd.DataFrame(data={'terms':freq_dict.keys(),'freq':freq_dict.values()})
    terms = pd.read_pickle(os.path.join(data_path,namespace,namespace+'_terms.pkl'))
    terms = terms.merge(freq_df,on='terms')
    freq_array = np.array(terms.freq.values)
    train_num = len(pd.read_pickle(train_data_file))
    loss_fn = ResampleLoss(reweight_func='rebalance', loss_weight=1.0,
                            focal=dict(focal=True, alpha=0.5, gamma=2),
                            logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                            map_param=dict(alpha=0.1, beta=10.0, gamma=0.05), 
                            class_freq=freq_array, train_num=train_num)
    label = torch.randint(0, 2, (4,len(freq_dict)))
    pred = torch.randn((4,len(freq_dict)))*100
    label = label.cuda()
    pred = pred.cuda()
    loss = loss_fn(pred,label)

    # loss_fn = ResampleLoss()
    # loss = loss_fn(pred,label)