import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.autograd import Function
import numpy as np

# from networks.diva import DIVA

class GradientReversalF(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = None
        dx = -lambda_ * grads
        return dx, None, None, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def adjust_lambda_(self, new_lambda):
        self.lambda_ = new_lambda

    def forward(self, x):
        return GradientReversalF.apply(x, self.lambda_)

class Encoder_dsads64(nn.Module):
    def __init__(self, opt, n_feature=64):
        super(Encoder_dsads64, self).__init__()
        self.baseline = opt.baseline
        self.baseline_norm = opt.baseline_norm
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=45, out_channels=16, kernel_size=(1, 9)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 9)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        self.proj = nn.Sequential(
            nn.Linear(in_features=32*25, out_features=32*25),
            nn.ReLU(),
            nn.Linear(in_features=32*25, out_features=n_feature),
        )

    def forward(self, x):
        features = self.conv1(x)
        features = self.conv2(features)
        features = features.reshape(-1, 32*25)
        if self.baseline and self.baseline_norm:
            features = F.normalize(features, dim=1)
        elif not self.baseline:
            features = F.normalize(features, dim=1)
        proj = self.proj(features)
        proj = F.normalize(proj, dim=1)
        if self.baseline and self.baseline_norm:
            features = F.normalize(features, dim=1)
        elif not self.baseline:
            features = F.normalize(features, dim=1)
        return proj, features

class Classifier_dsads64(nn.Module):
    def __init__(self, opt, n_feature=64):
        super(Classifier_dsads64, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(in_features=32*25, out_features=int(32*25/2)),
            nn.ReLU(),
            nn.Dropout(p=opt.classifier_dropout)
        )
        self.classifier = nn.Linear(in_features=int(32*25/2), out_features=opt.num_classes)
        self.discriminator = torch.nn.ModuleList()
        self.revgrad = GradientReversal()

        self.discriminator.append(nn.Sequential(nn.Dropout(p=opt.discriminator_dropout), nn.Linear(in_features=int(32*25/2), out_features=opt.num_domains-1)))
    
    def forward(self, features, output_dm=False):
        feat_output = self.fc(features)
        if output_dm:
            rev_output = self.revgrad(feat_output)
            domain_output = self.discriminator[0](rev_output)
        else:
            domain_output = None  
        class_output = self.classifier(feat_output)
        return class_output, domain_output, feat_output

class Encoder_uschad64(nn.Module):
    def __init__(self, opt, n_feature=64):
        super(Encoder_uschad64, self).__init__()
        self.baseline = opt.baseline
        self.baseline_norm = opt.baseline_norm
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(1, 6)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 6)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        self.proj = nn.Sequential(
            nn.Linear(in_features=32*27, out_features=32*27),
            nn.ReLU(),
            nn.Linear(in_features=32*27, out_features=n_feature),
        )

    def forward(self, x):
        features = self.conv1(x)
        features = self.conv2(features)
        features = features.reshape(-1, 32*27)
        if self.baseline and self.baseline_norm:
            features = F.normalize(features, dim=1)
        elif not self.baseline:
            features = F.normalize(features, dim=1)
        proj = self.proj(features)
        proj = F.normalize(proj, dim=1)
        if self.baseline and self.baseline_norm:
            features = F.normalize(features, dim=1)
        elif not self.baseline:
            features = F.normalize(features, dim=1)
        return proj, features

class Classifier_uschad64(nn.Module):
    def __init__(self, opt, n_feature=64):
        super(Classifier_uschad64, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(in_features=32*27, out_features=int(32*27/2)),
            nn.ReLU(),
            nn.Dropout(p=opt.classifier_dropout)
        )
        self.classifier = nn.Linear(in_features=int(32*27/2), out_features=opt.num_classes)
        self.discriminator = torch.nn.ModuleList()
        self.revgrad = GradientReversal()

        self.discriminator.append(nn.Sequential(nn.Dropout(p=opt.discriminator_dropout), nn.Linear(in_features=int(32*27/2), out_features=opt.num_domains-1)))

    def forward(self, features, output_dm=False):
        feat_output = self.fc(features)
        if output_dm:
            rev_output = self.revgrad(feat_output)
            domain_output = self.discriminator[0](rev_output)
        else:
            domain_output = None
        class_output = self.classifier(feat_output)
        return class_output, domain_output, feat_output

class Encoder_pamap64(nn.Module):
    def __init__(self, opt, n_feature=64):
        super(Encoder_pamap64, self).__init__()
        self.baseline = opt.baseline
        self.baseline_norm = opt.baseline_norm
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=27, out_channels=16, kernel_size=(1, 9)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, 9)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2), stride=2)
        )
        self.proj = nn.Sequential(
            nn.Linear(in_features=32*25, out_features=32*25),
            nn.ReLU(),
            nn.Linear(in_features=32*25, out_features=n_feature),
        )

    def forward(self, x):
        features = self.conv1(x)
        features = self.conv2(features)
        features = features.reshape(-1, 32*25)
        if self.baseline and self.baseline_norm:
            features = F.normalize(features, dim=1)
        elif not self.baseline:
            features = F.normalize(features, dim=1)
        proj = self.proj(features)
        proj = F.normalize(proj, dim=1)
        if self.baseline and self.baseline_norm:
            features = F.normalize(features, dim=1)
        elif not self.baseline:
            features = F.normalize(features, dim=1)
        return proj, features

class Classifier_pamap64(nn.Module):
    def __init__(self, opt, n_feature=64):
        super(Classifier_pamap64, self).__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(in_features=32*25, out_features=int(32*25/2)),
            nn.ReLU(),
            nn.Dropout(p=opt.classifier_dropout)
        )
        self.classifier = nn.Linear(in_features=int(32*25/2), out_features=opt.num_classes)
        self.discriminator = torch.nn.ModuleList()
        self.revgrad = GradientReversal()

        self.discriminator.append(nn.Sequential(nn.Dropout(p=opt.discriminator_dropout), nn.Linear(in_features=int(32*25/2), out_features=opt.num_domains-1)))

    def forward(self, features, output_dm=False):
        feat_output = self.fc(features)
        if output_dm:
            rev_output = self.revgrad(feat_output)
            domain_output = self.discriminator[0](rev_output)
        else:
            domain_output = None
        class_output = self.classifier(feat_output)
        return class_output, domain_output, feat_output