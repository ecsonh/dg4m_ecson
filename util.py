from __future__ import print_function

import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_class_weight

from pprint import pprint
import os

class NormalizeTransform(torch.nn.Module):
    """Normalize data"""
    
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean
        self.std = std

    def forward(self, data):
        data_clone = data.copy()
        data_clone = np.divide(np.subtract(data, np.asarray(self.mean).reshape(-1, 1)), np.asarray(self.std).reshape(-1,1))
        return data_clone

class RandomAmplitudeTransform(torch.nn.Module):
    """Applies amplitude transformation"""
    
    def __init__(self, num_sensors, axs=['x', 'y', 'z'], r=torch.arange(0.55, 1.05, 0.05), p=0.5):
            super().__init__()
            self.num_sensors = num_sensors
            self.axs = [ax.lower() for ax in axs]
            self.r = r
            self.p = p
    
    def amplify(self, data, axs, r):
        if axs == 'x':
            data[0] /= r 
        elif axs == 'y':
            data[1] /= r
        elif axs == 'z':
            data[2] /= r
        return data

    def forward(self, data):
        data_clone = data.copy()
        if torch.rand(1) < self.p:
            num_axs = torch.randint(0, len(self.axs), (1, )) + 1                   # number of axs to apply
            axs_idxs = torch.randint(0, len(self.axs), (num_axs, ))        # axs to apply
            r_idxs = torch.randint(0, len(self.r), (num_axs, ))          # ratios to choose
            for sensor in range(self.num_sensors):
                for axs_idx in axs_idxs:
                    for r_idx in r_idxs:
                        start_idx = sensor * 3
                        end_idx = start_idx + 3
                        data_clone[start_idx:end_idx] = self.amplify(torch.tensor(data_clone[start_idx:end_idx]), self.axs[axs_idx], self.r[r_idx])
        return data_clone

class RandomRotationTransform(torch.nn.Module):
    """Rotates a 3-axis sensor(s) along a specified dimension(s) and rotation in degrees
       assumes data has the structure (axis, segment) 
       performs rotation with probability p, otherwise no rotation is performed
       default: rotation list = (0, 35, 5) 0 to 30 degrees by 5 increment
       Two cases:
               1: If axs is not provided, randomly choose axs(s) and random angle(s) from rotation list
               2: Rotates along a specified axis and rotation list"""
    
    def __init__(self, num_sensors, axs=['x', 'y', 'z'], r=torch.arange(0, 35, 5), p=0.5):
        super().__init__()
        self.num_sensors = num_sensors
        self.axs = [ax.lower() for ax in axs]
        self.r = r * math.pi / 180 ## convert to radians
        self.p = p

    def rotate(self, data, axs, r):
        if axs == 'x':
            """Around X-axis"""
            data[1] = data[1] * torch.cos(r) - data[2] * torch.sin(r)
            data[2] =  data[1] * torch.sin(r) + data[2] * torch.cos(r)
        elif axs == 'y':
            """Around Y-axis"""
            data[0] = data[0] * torch.cos(r) + data[2] * torch.sin(r)
            data[2] = data[2] * torch.cos(r) - data[0] * torch.sin(r)
        elif axs == 'z':
            """Around Z-axis"""
            data[0] = data[0] * torch.cos(r) - data[1] * torch.sin(r)
            data[1] = data[0] * torch.sin(r) + data[1] * torch.cos(r)
        return data

    def forward(self, data):
        data_clone = data.copy()
        if torch.rand(1) < self.p:
            num_axs = torch.randint(0, len(self.axs), (1, )) + 1                   # number of axs to apply
            axs_idxs = torch.randint(0, len(self.axs), (num_axs, ))        # axs to apply
            r_idxs = torch.randint(0, len(self.r), (num_axs, ))          # rotations to choose
            for sensor in range(self.num_sensors):
                for axs_idx in axs_idxs:
                    for r_idx in r_idxs:
                        start_idx = sensor * 3
                        end_idx = start_idx + 3
                        data_clone[start_idx:end_idx] = self.rotate(torch.tensor(data_clone[start_idx:end_idx]), self.axs[axs_idx], self.r[r_idx])
        return data_clone

class TwoTransforms:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        data1 = self.transform(x).squeeze()
        data2 = self.transform(x).squeeze()
        # while(torch.equal(data1, data2)):
        #     data2 = self.transform(x).squeeze()
        return [data1, data2]

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

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


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
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

def adjust_lambd(opt, revgrad, epoch, lambd_p, factor=1.0):
    lambd_ = ((2 / (1 + math.exp(-opt.gamma * lambd_p[epoch]))) - 1) * factor
    revgrad.adjust_lambda_(lambd_)
    return lambd_

def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def set_optimizer(opt, encoder):
    optimizer = optim.SGD(encoder.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer

def set_optimizers(opt, encoder, classifier_discriminator):
    optimizer_enc = optim.SGD(encoder.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    if opt.classifier_optimizer.lower() == 'adam':
        optimizer_class = optim.Adam(classifier_discriminator.parameters(),
                                    lr=opt.classifier_lr, weight_decay=opt.classifier_wd)
    else:
        optimizer_class = optim.SGD(classifier_discriminator.parameters(),
                            lr=opt.classifier_lr,
                            momentum=opt.classifier_momentum,
                            weight_decay=opt.classifier_wd)
    merged_params = list(encoder.parameters()) + list(classifier_discriminator.parameters())
    if opt.optimizer_type.lower() == 'adam':
        optimizer_whole = optim.Adam(merged_params,
                                      lr=opt.learning_rate, weight_decay=opt.weight_decay)
    elif opt.optimizer_type.lower() == 'sgd':
        optimizer_whole = optim.SGD(merged_params,
                            lr=opt.learning_rate,
                            momentum=opt.momentum,
                            weight_decay=opt.weight_decay)
    return optimizer_enc, optimizer_class, optimizer_whole

def metric_calc(loss, labels, preds, metric_type):
    return {'loss': loss,
            'acc': accuracy_score(labels, preds),
            'f1': f1_score(labels, preds, average=metric_type, zero_division=0),
            'conf_mtx': str(confusion_matrix(labels, preds)).replace('\n', ','),
            'prec': precision_score(labels, preds, average=metric_type, zero_division=0),
            'rec': recall_score(labels, preds, average=metric_type, zero_division=0),
    }

def log_result(wb, wandb, metrics, epoch_idx):
    print("")
    print("Mini Test for Epochs %d:"%epoch_idx)
    pprint(metrics)
    if wb:
        keys2pop = [key for key in metrics.keys() if "BT" in key or "DT" in key or "conf_mtx" in key]
        for key in keys2pop:
            metrics.pop(key)
        wandb.log(metrics)

def get_class_weights(opt, train_data, test_data):
    train_labels = list(np.asarray(train_data.data, dtype=object)[:, 1])
    if opt.verbose:
        test_labels = [label for _, label, _ in test_data]
        from collections import Counter
        train_class_ratios = np.array(list(Counter(train_labels).values())) / len(train_labels)
        test_class_ratios = np.array(list(Counter(test_labels).values())) / len(test_labels)
        print(f'train_class_ratios: {train_class_ratios}')
        print(f'test_class_ratios: {test_class_ratios}')
    class_weights = torch.from_numpy(compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels))
    print(f'class_weights: {class_weights}')
    _, remap_dict = remap_discontinuous_labels(train_labels)
    return class_weights, remap_dict

def get_domain_weights(train_data, test_data):
    domain_labels_train = list(np.asarray(train_data.data, dtype=object)[:, 2])
    domain_labels_test = list(np.asarray(test_data.data, dtype=object)[:, 2])
    domains = np.concatenate((domain_labels_train, domain_labels_test))
    domain_weights = torch.from_numpy(compute_class_weight('balanced', classes=np.unique(domains), y=domains))
    # domain_weights = compute_class_weight('balanced', classes=np.unique(domain_labels_train), y=domain_labels_train)
    # domain_weights = torch.from_numpy(np.insert(domain_weights, np.unique(domain_labels_test), 1))
    return domain_weights

def get_domain_weights_ignore(opt, train_data, test_data):
    domain_labels_train = list(np.asarray(train_data.data, dtype=object)[:, 2])
    _, remap_dict = remap_discontinuous_labels(domain_labels_train)
    # domain_labels_test = [domain for _, _, domain in test_data]

    if opt.verbose:
        from collections import Counter
        train_domain_ratios = np.array(list(Counter(domain_labels_train).values())) / len(domain_labels_train)
        # test_domain_ratios = np.array(list(Counter(domain_labels_test).values())) / len(domain_labels_test)
        print(f'train_domain_ratios: {train_domain_ratios}')
        # print(f'test_domain_ratios: {test_domain_ratios}')
        
    # domains = np.concatenate((domain_labels_train, domain_labels_test))
    # domain_weights = torch.from_numpy(compute_class_weight('balanced', classes=np.unique(domain_labels_train), y=domain_labels_train))
    domain_weights = torch.from_numpy(compute_class_weight('balanced', classes=np.unique(domain_labels_train), y=domain_labels_train))
    # domain_weights = torch.from_numpy(np.insert(domain_weights, np.unique(domain_labels_test), 1))

    print(f'domain_weights: {domain_weights}')
    return domain_weights, remap_dict

def save_encoder(encoder, optimizer, opt, epoch, save_file):
    state = {
        'opt': opt,
        'encoder': encoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

def save_classifier(classifier_discriminator, optimizer, opt, epoch, save_file):
    state = {
        'opt': opt,
        'classifier_discriminator': classifier_discriminator.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

def save_model(encoder, classifier_discriminator, optimizer, optimizer2, opt, epoch, save_file):
    state = {
        'opt': opt,
        'encoder': encoder.state_dict(),
        'classifier_discriminator': classifier_discriminator.state_dict(),
        'optimizer': optimizer.state_dict(),
        'optimizer2': optimizer2.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state

def load_encoder(encoder, optimizer, load_path, ckpt=None):
    if ckpt is not None:
        if os.path.exists(ckpt):
            print(f'model ckpt loaded: {ckpt}')
            state = torch.load(ckpt)
            encoder.load_state_dict(state['encoder'])
            optimizer.load_state_dict(state['optimizer'])
            epoch = state['epoch']
        else:
            print(f'model ckpt: {ckpt} not found')
    else:
        ckpts = [os.path.join(path, name) for path, subdirs, files in os.walk(load_path) for name in files]
        ckpts.sort()
        epoch = 0
        if len(ckpts) != 0:
            print(f'model ckpt loaded: {ckpts[-1]}')
            state = torch.load(ckpts[-1])
            encoder.load_state_dict(state['encoder'])
            optimizer.load_state_dict(state['optimizer'])
            epoch = state['epoch']
        else:
            print(f'model ckpt: {load_path} not found')
    return encoder, optimizer, epoch

def load_classifier(classifier_discriminator, optimizer, load_path, direct_ckpt=None):
    if direct_ckpt is not None:
        if os.path.exists(direct_ckpt):
            print(f'model ckpt loaded: {direct_ckpt}')
            state = torch.load(direct_ckpt)
            classifier_discriminator.load_state_dict(state['classifier_discriminator'])
            optimizer.load_state_dict(state['optimizer'])
            epoch = state['epoch']
        else:
            print(f'model ckpt: {direct_ckpt} not found')
    else:
        ckpts = [os.path.join(path, name) for path, subdirs, files in os.walk(load_path) for name in files]
        ckpts.sort()
        epoch = 0
        if len(ckpts) != 0:
            print(f'model ckpt loaded: {ckpts[-1]}')
            state = torch.load(ckpts[-1])
            classifier_discriminator.load_state_dict(state['classifier_discriminator'])
            optimizer.load_state_dict(state['optimizer'])
            epoch = state['epoch']
        else:
            print(f'model ckpt: {load_path} not found')
    return classifier_discriminator, optimizer, epoch

def load_model(encoder, classifier_discriminator, optimizer, optimizer2, load_path, direct_ckpt=None):
    if direct_ckpt is not None:
        if os.path.exists(direct_ckpt):
            print(f'model ckpt loaded: {direct_ckpt}')
            state = torch.load(direct_ckpt)
            encoder.load_state_dict(state['encoder'])
            classifier_discriminator.load_state_dict(state['classifier_discriminator'])
            optimizer.load_state_dict(state['optimizer'])
            optimizer2.load_state_dict(state['optimizer2'])
            epoch = state['epoch']
        else:
            print(f'model ckpt: {direct_ckpt} not found')
    else:
        ckpts = [os.path.join(path, name) for path, subdirs, files in os.walk(load_path) for name in files]
        ckpts.sort()
        epoch = 0
        if len(ckpts) != 0:
            print(f'model ckpt loaded: {ckpts[-1]}')
            state = torch.load(ckpts[-1])
            encoder.load_state_dict(state['encoder'])
            classifier_discriminator.load_state_dict(state['classifier_discriminator'])
            optimizer.load_state_dict(state['optimizer'])
            optimizer2.load_state_dict(state['optimizer2'])
            epoch = state['epoch']
        else:
            print(f'model ckpt: {load_path} not found')
    return encoder, classifier_discriminator, optimizer, optimizer2, epoch

def remap_discontinuous_labels(labels):

    remap_dict = dict()
    ret = list()

    unique_labels = np.unique(labels)
    for idx, label in enumerate(unique_labels):
        remap_dict[label] = idx
    ret = np.array([remap_dict[label] for label in labels])
    return ret, remap_dict

def init_layer_weights(layer, init_func=None):
    if init_func is not None:
        init_func(layer.weight)
    else:
        nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
