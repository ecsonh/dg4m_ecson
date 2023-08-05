from __future__ import print_function
import pdb
import os
import sys
import argparse
import time
import math

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms

from util import TwoTransforms, NormalizeTransform, RandomRotationTransform, RandomAmplitudeTransform
from util import AverageMeter, metric_calc, log_result, get_class_weights, get_domain_weights_ignore
from util import adjust_learning_rate, warmup_learning_rate, adjust_lambd
from util import set_optimizers, save_model, save_encoder, save_classifier, load_model, load_encoder, load_classifier, init_layer_weights
from networks.network import Encoder_dsads64, Classifier_dsads64, Encoder_uschad64, Classifier_uschad64, Encoder_pamap64, Classifier_pamap64
from losses import SupConLoss

from dataset import DSADS, USC_HAD, PAMAP2
from pprint import pprint
import numpy as np

from sklearn.metrics import confusion_matrix

# try:
#     import apex
#     from apex import amp, optimizers
# except ImportError:
#     pass

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # misc.
    parser.add_argument('--wb', type=int, default=0,
                        help='enable wandb')
    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency for training metrics')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='frequency for saving models')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='num of workers to use for dataloader')
    parser.add_argument('--verbose', type=int, default=0,
                        help='print out additional information')
    parser.add_argument('--no_cuda', type=int, default=0,
                        help='use cuda if available')
    parser.add_argument('--cuda_device', type=int, default=0,
                        help='use the specified cuda device')

    # training SupCon
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of training epochs for SupCon')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size for SupCon')
    parser.add_argument('--optimizer_type', type=str, default='sgd')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate for SupCon')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')        
    parser.add_argument('--weighted_supcon', type=int, default=0,
                        help='turn on class weight for supcon loss')
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')
    parser.add_argument('--base_temp', type=float, default=0.07,
                        help='base temperature for loss function')
    parser.add_argument('--cosine', type=int, default=1,
                        help='using cosine annealing')
    parser.add_argument('--syncBN', type=int, default=0,
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', type=int, default=0,
                        help='warm-up for large batch training')
    
    # domain discriminator
    parser.add_argument('--double_train', type=int, default=0,
                        help='2 pass to discriminator')
    parser.add_argument('--lambda_factor', type=float, default=1.0,
                        help='factor to multiply the adaptive lambda')
    parser.add_argument('--discriminator_dropout', type=float, default=0.2,
                        help='classifiers dropout probability')

    # training Classifier
    parser.add_argument('--classifier_optimizer', type=str, default='adam',
                    help='optimizer choice for classifier block')
    parser.add_argument('--classifier_epochs', type=int, default=250,
                        help='classifiers training epochs')
    parser.add_argument('--classifier_bs', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--classifier_lr', type=float, default=0.001,
                        help='classifiers learning rate')
    parser.add_argument('--classifier_momentum', type=float, default=0.9,
                        help='classifiers momentum')
    parser.add_argument('--classifier_wd', type=float, default=0.0,
                        help='classifiers weight decay')
    parser.add_argument('--classifier_dropout', type=float, default=0.0,
                        help='classifiers dropout probability')
    parser.add_argument('--patience', type=int, default=50,
                        help='early stopping patience')
    parser.add_argument('--supcon_loss', type=int, default=0,
                        help='include supcon loss in classifier training')    

    # dataset
    parser.add_argument('--model', type=str, default='uschad64', 
                        choices=['dsads64', 'uschad64', 'pamap64'], help='model')
    parser.add_argument('--dataset', type=str, default='uschad',
                        choices=['dsads', 'uschad', 'pamap2'], help='dataset')
    parser.add_argument('--data_folder', type=str, default=None, help='path to dataset')
    parser.add_argument('--data_cache_path', type=str, default='./', help='path to preprocessed data pickle')
    parser.add_argument('--window', type=int, default=126, help='window size')
    parser.add_argument('--slide', type=float, default=0.5, help='slide size')
    parser.add_argument('--val_split', type=float, default=0.2, help='validiation split ratio')

    # experiments
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')
    parser.add_argument('--seed', type=int, default=0, help='program seed')
    parser.add_argument('--trial', type=str, default='0', help='id for recording multiple runs')
    parser.add_argument('--LODO', type=int, default=0, help='domain to leave-out')
    parser.add_argument('--eval', type=int, default=0,
                            help='for training classifier and evaluating performance')
    parser.add_argument('--baseline', type=int, default=0,
                            help='enable baseline training')
    parser.add_argument('--baseline_norm', type=int, default=0,
                            help='enable baseline normalization')
    parser.add_argument('--transforms', type=int, default=0,
                            help='enable transforms in baseline training')
    parser.add_argument('--rotations', nargs="+", type=int, default=[0, 5, 10, 15, 20, 25, 30],
                        help='rotation degrees for rotation transformations')
    parser.add_argument('--loro', nargs="+", type=int, default=None,
                        help='the rotation(s) to leave out of training')
    parser.add_argument('--amplitudes', nargs="+", type=float, default=[0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0],
                        help='rotation degrees for rotation transformations')
    parser.add_argument('--loao', nargs="+", type=float, default=None,
                        help='the rotation(s) to leave out of training')
    #ecson positions
    parser.add_argument('--position', type=str, default= None, choices = ['t', 'ra', 'la', 'll', 'rl'],help='Test one of 5 positions(T, RA, LA, LL, RL)')
    

    opt = parser.parse_args()

    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)

    # check if dataset is path that passed required arguments
    if opt.dataset == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './datasets/{}'.format(opt.dataset)
    opt.model_path = './save/DiSCovHAR/{}_models'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))
    #ecson add
    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_temp_{}_lambda_{}_lodo_{}_trial_{}_seed_{}_position_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, opt.lambda_factor, opt.LODO, opt.trial, opt.seed, opt.position)
    
    if opt.loro is not None:
        opt.model_name = '{}_loro_{}'.format(opt.model_name, opt.loro)

    if opt.loao is not None:
        opt.model_name = '{}_loao_{}'.format(opt.model_name, opt.loao)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size >= 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)

    opt.classifier_name = 'classifier'

    if opt.classifier_dropout > 0.0:
        opt.classifier_name = '{}_dropout_{}'.format(opt.classifier_name, opt.classifier_dropout)

    if opt.classifier_wd > 0.0:
        opt.classifier_name = '{}_wd_{}'.format(opt.classifier_name, opt.classifier_wd)

    if opt.supcon_loss:
        opt.classifier_name = '{}_supcon_loss{}'.format(opt.classifier_name)

    opt.classifier_folder = os.path.join(opt.save_folder, opt.classifier_name)
    if not os.path.isdir(opt.classifier_folder):
        os.makedirs(opt.classifier_folder)

    if not opt.baseline and not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt

def set_loader(opt):
    # construct data loader
    if opt.dataset == 'dsads':
        normstats = {
            0:[(7.76576563e+00, -8.11036481e-01, 2.76884486e+00,
                -2.79630795e-03, 1.36950428e-02, -3.31175602e-03,
                -5.98435613e-01, 6.17291609e-02, -2.72517018e-01,
                4.26059529e+00, 4.93877870e+00, 3.11926240e+00,
                1.63019858e-02, -2.14838962e-02, -3.18008369e-03,
                -2.11230457e-01, -3.25661652e-01, -2.61232185e-01,
                4.27899464e+00, -3.91063565e+00, 3.82216535e+00,
                -1.82191779e-02, -2.28597151e-02, 2.56695212e-03,
                -2.78319085e-01, 3.23281981e-01, -3.43634856e-01,
                -7.37938968e+00, 2.71898689e+00, -1.51953386e+00,
                2.08523942e-02, 3.05308583e-02, -4.07740508e-03,
                4.28238878e-01, -2.70120375e-01, 6.88012434e-02,
                -7.32448907e+00, -3.21791634e+00, -5.30367493e-01,
                -1.56389119e-02, 2.78643739e-02, -2.45366806e-03,
                4.58107217e-01, 2.95544236e-01, 6.45039725e-02),
                (5.63788423, 2.62302609, 3.53825845, 0.79401035, 0.69103979,
                0.31076554, 0.35609989, 0.34043553, 0.37341144, 5.82133691,
                4.58021943, 3.86947263, 0.86944219, 0.76474872, 1.0231465 ,
                0.46358023, 0.38304311, 0.43755373, 6.02394276, 4.43640874,
                4.39969299, 0.93968047, 0.81155208, 0.98234379, 0.4683943 ,
                0.43804756, 0.44817703, 5.8802031 , 5.99695836, 3.12366522,
                0.85781175, 0.49406364, 1.14297172, 0.36441279, 0.38321877,
                0.36163722, 5.81075766, 5.92555299, 3.39261819, 0.86829163,
                0.48352606, 1.15895247, 0.38023965, 0.38389433, 0.35738408)],
            1:[(7.88676261e+00, -8.46201643e-01, 2.79531970e+00, -4.26877382e-03,
                1.45943971e-02, -3.21713396e-03, -6.15848012e-01, 6.97999630e-02,
                -2.73398089e-01, 4.01546283e+00, 4.86946388e+00, 3.52104882e+00,
                1.36204909e-02, -2.10627441e-02, -5.57639332e-03, -2.20602072e-01,
                -3.05205715e-01, -2.77493261e-01, 4.07373934e+00, -4.01719584e+00,
                3.92113633e+00, -1.93111220e-02, -2.16747348e-02, 2.70996869e-03,
                -2.44216395e-01, 3.34470909e-01, -3.45881190e-01, -7.37990517e+00,
                2.73218160e+00, -1.50464423e+00, 2.13437062e-02, 3.16049939e-02,
                -3.34099193e-03, 4.12766748e-01, -2.68111033e-01, 7.22907081e-02,
                -7.29834950e+00, -3.22115592e+00, -4.25477472e-01, -1.38134558e-02,
                2.91049957e-02, -2.41185861e-03, 4.42643849e-01, 2.85040737e-01,
                5.10171369e-02),
                (5.60808185, 2.55363438, 3.64434647, 0.82658264, 0.75398717, 0.31473178,
                0.34565629, 0.34315192, 0.37722215, 5.77125312, 4.56495506, 3.71520623,
                0.86299895, 0.74541419, 0.99092552, 0.45285512, 0.39175436, 0.42173126,
                5.98628932, 4.34571215, 4.48667549, 0.97206728, 0.83689147, 0.97920184,
                0.46469666, 0.43081683, 0.45860063, 5.97076711, 6.03608422, 3.10389232,
                0.90426648, 0.49141103, 1.19309074, 0.36396717, 0.38752637, 0.35217974,
                5.90649726, 6.03181727, 3.34009377, 0.89629307, 0.47679377, 1.2167641,
                0.37655735, 0.37521268, 0.35404438)],
            2:[(7.82932117e+00, -8.09211191e-01, 2.56654260e+00, -1.05692903e-03,
                1.39381651e-02, -2.65277176e-03, -6.17582880e-01, 5.84409098e-02,
                -2.59910416e-01, 4.16092497e+00, 5.28971390e+00, 2.75988711e+00,
                1.96783623e-02, -1.88208723e-02, -1.00760762e-03, -1.99160739e-01,
                -3.57227445e-01, -2.44159663e-01, 4.28305130e+00, -3.88527881e+00,
                3.69867303e+00, -1.84035804e-02, -2.57720282e-02, 1.08078384e-03,
                -2.93098734e-01, 3.22032653e-01, -3.47902200e-01, -7.46456380e+00,
                2.69836791e+00, -1.33909127e+00, 1.93058364e-02, 3.08547333e-02,
                -4.95528678e-03, 4.38197336e-01, -2.68902478e-01, 4.87193057e-02,
                -7.38855065e+00, -3.33120504e+00, -5.50562245e-01, -1.85327598e-02,
                2.98068969e-02, -1.56579695e-03, 4.68117760e-01, 3.07205419e-01,
                6.98253617e-02),
                (5.68591965, 2.71196868, 3.70483782, 0.84829712, 0.66452359, 0.3305742,
                0.35806576, 0.34961627, 0.36240799, 5.84213025, 4.67392863, 3.9848496,
                0.8791802, 0.79245744, 1.07815209, 0.46713583, 0.3742102, 0.45872826,
                6.08127151, 4.65493413, 4.48673984, 0.96143516, 0.83748334, 1.02147134,
                0.46319593, 0.42780081, 0.44964463, 5.89099545, 6.06495791, 3.24946301,
                0.87159302, 0.52571388, 1.13776578, 0.36014365, 0.38429376, 0.37262456,
                5.82264086, 5.97165119, 3.43255861, 0.89752654, 0.51463928, 1.15914477,
                0.37877159, 0.38551603, 0.35988225)],
            3:[(7.67943856e+00, -6.33912528e-01, 2.83581162e+00, -3.11036888e-03,
                1.30962890e-02, -3.21317485e-03, -5.81359117e-01, 4.40107844e-02,
                -2.85993501e-01, 4.24669811e+00, 4.59133313e+00, 3.42127975e+00,
                1.66027822e-02, -2.17369333e-02, -5.02681108e-03, -2.02064203e-01,
                -3.16342628e-01, -2.85566747e-01, 4.24120613e+00, -3.96807519e+00,
                3.74663048e+00, -1.75443676e-02, -1.84919995e-02, 1.77074857e-03,
                -2.72707467e-01, 3.19687052e-01, -3.41049291e-01, -7.25940193e+00,
                2.85078229e+00, -1.66621991e+00, 2.09954696e-02, 2.95296159e-02,
                -5.42235384e-03, 4.12680211e-01, -2.93558698e-01, 8.01105250e-02,
                -7.23059240e+00, -3.25253752e+00, -5.59956328e-01, -1.57833427e-02,
                2.50392601e-02, -2.87406473e-03, 4.45225274e-01, 3.06899199e-01,
                6.78153294e-02),
                (5.51442338, 2.54522447, 3.29081732, 0.71954457, 0.64683637, 0.28329772,
                0.3716294, 0.32421059, 0.37503553, 5.78062817, 4.65438752, 3.85821517,
                0.85926021, 0.76462222, 1.00713724, 0.46909505, 0.38608511, 0.43407823,
                5.89526627, 4.31020613, 4.40392918, 0.85056874, 0.75972408, 0.96073417,
                0.46913487, 0.44666897, 0.45515748, 5.74361394, 5.79102379, 2.92329299,
                0.77383745, 0.45161024, 1.11060598, 0.37562493, 0.37737325, 0.35052179,
                5.70191402, 5.70756081, 3.31570908, 0.78866809, 0.4540143, 1.12202816,
                0.38877164, 0.37774044, 0.35239513)]
        }
        mean = normstats[opt.LODO][0]
        std = normstats[opt.LODO][1]
        #opt.num_domains = 5
        opt.num_domains = 5
        opt.num_classes = 19
        opt.num_sensors = 15
    elif opt.dataset == 'uschad':
        normstats = {
            0:[(0.78195487,  0.21496111, -0.05487249, -0.37075692, -0.27835201, -0.17473576),
                (0.63814527,  0.39143873,  0.34252298, 53.24449367, 27.35744257, 48.40558098)],
            1:[(0.79484242, 0.18859915, -0.05742496, -0.57311041, -0.36449177, -0.20760476),
                (0.63949637, 0.4010962, 0.333326, 54.20311886, 28.81606737, 49.6689737)],
            2:[(0.78216536, 0.22556399, -0.05679427, -0.37597559, -0.35365674, -0.16439425),
                (0.63309788, 0.37599506, 0.34045601, 52.36997098, 28.00270321, 47.71026656)],
            3:[(0.79563342, 0.167658, -0.07053719, -0.47839441, -0.47536046, -0.0974736),
                (0.64701061, 0.39135867, 0.3421972, 52.8076551, 28.14726955, 49.29461002)],
            4:[(0.75981298, 0.20958271, -0.06981445, -0.27366824, -0.28789361, -0.22871619),
                (0.66559139, 0.38202959, 0.31930651, 49.15944476, 25.46235909, 43.42030691)]
        }
        mean = normstats[opt.LODO][0]
        std = normstats[opt.LODO][1]
        opt.num_domains = 5
        opt.num_classes = 12
        opt.num_sensors = 2
    elif opt.dataset == 'pamap2':
        normstats = {
            0:[(-4.99929302e+00, 2.75587789e+00, 3.81259445e+00, -1.61402932e-02,
                7.44083529e-03, -1.26886649e-02, 2.11288701e+01, -1.01282113e+01,
                -2.38378507e+01, 3.16456526e-01, 8.11424621e+00, -3.57627607e-02,
                5.55709508e-03, 3.43472491e-03, -2.07347815e-02, 5.78804203e+00,
                -2.74119556e+01,-1.25418006e+00, 8.78476996e+00, -6.54844278e-01,
                -2.65717782e+00, 8.49568780e-03, -2.98568986e-02, 3.16875642e-03,
                -2.69051075e+01, 3.88483771e+00, 1.86841621e+01),
                (5.64124374, 4.38224646, 3.47488598, 1.17572423, 0.78374766, 1.16166721,
                22.3614014, 22.29905886, 18.92523223, 1.57037529, 3.86584931, 4.466007,
                0.28048905, 0.48429178, 0.22880419, 16.3480743, 16.42753774, 17.46291967,
                5.19875343, 6.16752571, 3.06770948, 0.98774851, 0.52437006, 1.52866332,
                16.15232045, 18.60149495, 18.82146034)],
            1:[(-4.81667677e+00, 3.04196926e+00, 3.60920667e+00, -1.56335702e-02,
                2.06524797e-02, -6.90882543e-03, 1.92020635e+01, -1.20531820e+01,
                -2.38496743e+01, 4.38004495e-01, 8.05572828e+00, -8.13613841e-01,
                5.66477069e-03, 9.73597739e-04,-1.79421427e-02, 4.54499911e+00,
                -2.96855473e+01, 4.10082714e-01, 8.60510419e+00, -8.65080946e-01,
                -2.92555419e+00, 2.76822234e-03,-2.36298230e-02, 5.38567962e-04,
                -3.06761322e+01, 4.35481299e+00, 1.85463388e+01),
                (5.79098708, 4.41637668, 3.47843666, 1.19698014, 0.71616424, 1.2751304,
                25.69081473, 24.83409698, 21.44376907, 1.6025641, 3.67914149, 4.66839042,
                0.30075238, 0.47576357, 0.25259502, 18.23028565, 18.84818776, 21.22456778,
                5.3878967, 6.10217271, 2.97879443, 0.99850879, 0.55425381, 1.53993348,
                19.9448595, 21.4316061, 21.76911167)],
            2:[(-5.00206979e+00, 2.85841399e+00, 3.64812705e+00, -5.91240143e-03,
                1.31574324e-02, -1.60735951e-02, 2.00265747e+01, -1.17186269e+01,
                -2.42132533e+01, 3.24785312e-01, 7.95995210e+00, -8.96921802e-01,
                5.30483416e-03, -2.91329407e-03, -1.84776813e-02, 5.06360029e+00,
                -2.92124679e+01, 1.07285952e+00, 8.74760872e+00, -6.66292672e-01,
                -2.47610382e+00, -1.97288482e-04, -2.82305024e-02, 9.58654165e-04,
                -2.92614281e+01, 2.63012340e+00, 1.71901189e+01),
                (5.88894674, 4.31815075, 3.54686772, 1.25225279, 0.7828057, 1.33200876,
                24.89891058, 23.2024172, 21.30919997, 1.62071626, 3.74698043, 4.80200094,
                0.29318551, 0.47546889, 0.25867543, 18.19474956, 18.52201441, 21.28984968,
                5.4137657, 6.39713877, 3.2762976, 0.97525619, 0.55287842, 1.55207253,
                20.51003427, 20.79551012, 21.66008368)],
            3:[(-4.58726844e+00, 4.27771288e+00, 3.79835112e+00, -1.60815617e-02,
                1.43103965e-02, -1.37412577e-02, 1.92513013e+01, -1.59341243e+01,
                -2.48659135e+01, 2.81962182e-01, 7.99210514e+00, -7.96534325e-01,
                6.17310279e-03, -4.72069676e-04, -1.93079132e-02, 5.30299021e+00,
                -2.88502912e+01, 4.02971734e-01, 8.80803302e+00, -8.78646661e-01,
                -2.21260929e+00, -3.51281380e-03, -2.41492044e-02, 7.20209689e-03,
                -3.04290217e+01, 3.48711053e+00, 1.82703934e+01),
                (5.89506156, 3.37340844, 3.50301126, 1.27856532, 0.79434568, 1.32523547,
                23.6199011, 23.19391768, 21.84609581, 1.62178629, 3.72921185, 4.75849816,
                0.30605879, 0.49724774, 0.2609901, 17.97815317, 18.46056075, 21.06716032,
                5.04445625, 5.97413987, 3.14943671, 0.95220023, 0.48869875,  1.51794299,
                20.00797922, 20.89595627, 21.7880029)]
        }
        mean = normstats[opt.LODO][0]
        std = normstats[opt.LODO][1]
        #need to modify
        opt.num_domains = 4
        opt.num_classes = 8
        opt.num_sensors = 9
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    #Apply sporadic D.S. augmentations here
    if opt.baseline:
        if opt.transforms:
            rot = torch.tensor(list(set(opt.rotations) - set(opt.loro))) if opt.loro is not None else torch.tensor(opt.rotations)
            amps = torch.tensor(list(set(opt.amplitudes) - set(opt.loao))) if opt.loao is not None else torch.tensor(opt.amplitudes)
            train_transform = transforms.Compose([
            RandomRotationTransform(opt.num_sensors, r=rot),
            RandomAmplitudeTransform(opt.num_sensors, r=amps),
            NormalizeTransform(mean, std),
        ])
        else:
            train_transform = transforms.Compose([
            NormalizeTransform(mean, std),
        ])

    else:
        rot = torch.tensor(list(set(opt.rotations) - set(opt.loro))) if opt.loro is not None else torch.tensor(opt.rotations)
        amps = torch.tensor(list(set(opt.amplitudes) - set(opt.loao))) if opt.loao is not None else torch.tensor(opt.amplitudes)
        train_transform = transforms.Compose([
            RandomRotationTransform(opt.num_sensors, r=rot),
            RandomAmplitudeTransform(opt.num_sensors, r=amps),
            NormalizeTransform(mean, std),
        ])

    test_transform = []

    if opt.loro is not None:
        test_transform.append(RandomRotationTransform(opt.num_sensors, r=torch.tensor(opt.loro)))
    if opt.loao is not None:
        test_transform.append(RandomAmplitudeTransform(opt.num_sensors, r=torch.tensor(opt.loao)))
    test_transform.append(NormalizeTransform(mean, std))
    test_transform = transforms.Compose(test_transform)

    if opt.dataset == 'dsads':
        opt.window = 125
        opt.slide = 1.0
        if opt.baseline:
            dataset = DSADS(opt, transform=train_transform)
            
        else:
            dataset = DSADS(opt, transform=TwoTransforms(train_transform))
    elif opt.dataset == 'uschad':
        opt.window = 126
        opt.slide = 0.5
        if opt.baseline:
            dataset = USC_HAD(opt, transform=train_transform)
        else:
            dataset = USC_HAD(opt, transform=TwoTransforms(train_transform))
    elif opt.dataset == 'pamap2':
        ## between 124 - 127
        opt.window = 125
        opt.slide = 0.5
        if opt.baseline:
            dataset = PAMAP2(opt, transform=train_transform)
        else:
            dataset = PAMAP2(opt, transform=TwoTransforms(train_transform))
    else:
        raise ValueError(opt.dataset)
    #ecson if the data loaded is split by position and the position arg is specified
    
    
    if opt.position is not None:
        #dataset.pos_load_data()
        dataset.load_data()
        dataset.restructure()
        train_dataset, test_dataset = dataset.position_split()
    else:
        dataset.load_data()
        train_dataset, test_dataset = dataset.LODO_split()
    # dataset.load_data()
    # train_dataset, test_dataset = dataset.LODO_split()
    if opt.val_split:
        train_dataset, val_dataset = train_dataset.val_split(opt.val_split)
        ## FIXME: should the validation dataset see transforms?
        # val_dataset.transform = test_transform
    test_dataset.transform = test_transform

    train_sampler = None
    # if opt.classifier_bs != 256:
    #     train_loader = torch.utils.data.DataLoader(
    #         train_dataset, batch_size=opt.classifier_bs, shuffle=(train_sampler is None),
    #         num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    #     train_loader2 = torch.utils.data.DataLoader(
    #         train_dataset, batch_size=opt.classifier_bs, shuffle=(train_sampler is None),
    #         num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    # else:
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
    train_loader2 = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
        num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)
        
    val_loader = None
    if opt.val_split:
        val_sampler = None
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=opt.batch_size, shuffle=(val_sampler is None),
            num_workers=opt.num_workers, pin_memory=True, sampler=val_sampler)

    test_sampler = None
    test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, shuffle=(test_sampler is None),
    num_workers=opt.num_workers, pin_memory=True, sampler=test_sampler)
    
    class_weights, opt.class_remap_dict = get_class_weights(opt, train_dataset, test_dataset)
    opt.class_weights = class_weights.float()
    domain_weights, opt.domain_remap_dict = get_domain_weights_ignore(opt, train_dataset, test_dataset)
    opt.domain_weights = domain_weights.float()

    return train_loader, train_loader2, val_loader, test_loader


def set_model(opt):
    if opt.model == 'dsads64':
        opt.x_dim = 32*25
        opt.zd_dim = int(32*25/2)
        opt.zx_dim = int(32*25/2)
        opt.zy_dim = int(32*25/2)

        encoder = Encoder_dsads64(opt)
        classifier = Classifier_dsads64(opt)
    elif opt.model == 'uschad64':
        opt.x_dim = 32*27
        opt.zd_dim = int(32*27/2)
        opt.zx_dim = int(32*27/2)
        opt.zy_dim = int(32*27/2)
        
        encoder = Encoder_uschad64(opt)
        classifier = Classifier_uschad64(opt)
    elif opt.model == 'pamap64':
        opt.x_dim = 32*25
        opt.zd_dim = int(32*25/2)
        opt.zx_dim = int(32*25/2)
        opt.zy_dim = int(32*25/2)
        
        encoder = Encoder_pamap64(opt)
        classifier = Classifier_pamap64(opt)
    
    if opt.weighted_supcon:
        criterion = SupConLoss(opt, weight=opt.class_weights)
    else:
        criterion = SupConLoss(opt)

    criterion_class = torch.nn.CrossEntropyLoss(weight=opt.class_weights)
    criterion_domain = torch.nn.CrossEntropyLoss(weight=opt.domain_weights)

    if opt.cuda:
        encoder = encoder.to(opt.device)
        classifier = classifier.to(opt.device)
        criterion = criterion.to(opt.device)
        criterion_class = criterion_class.to(opt.device)
        criterion_domain = criterion_domain.to(opt.device)
        cudnn.benchmark = True

    if opt.wb:
        wandb.watch(encoder, log="all")
        wandb.watch(classifier, log="all")

    return encoder, classifier, criterion, criterion_class, criterion_domain

def train_baseline(train_loader, encoder, classifier, criterion_class, criterion_domain, optimizer, epoch, opt):
    """one epoch train"""
    encoder.train()
    classifier.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    domain_losses = AverageMeter()

    end = time.time()
    for idx, (data, label, domain) in enumerate(train_loader):
        data_time.update(time.time() - end)
        domain = [opt.domain_remap_dict[d.item()] for d in domain]
        label = [opt.class_remap_dict[l.item()] for l in label]
        data = torch.as_tensor(data).type(torch.float).unsqueeze(2)
        label = torch.as_tensor(label).type(torch.int64)
        domain = torch.as_tensor(domain).type(torch.int64)
        if opt.cuda:
            data = data.to(opt.device)
            label = label.to(opt.device)
            domain = domain.to(opt.device)
        bsz = label.shape[0]

        # compute loss
        _, features = encoder(data)
        classifier_output, domain_output = classifier(features)
        class_loss = criterion_class(classifier_output, label)
        loss = class_loss
        if opt.domain:
            if opt.conditioned:
                expanded_domain_loss = criterion_domain(domain_output, domain)
                for idx, losses in enumerate(expanded_domain_loss):
                    dm_loss = torch.masked_select(dm_loss, torch.eq(label, idx)).mean()
                    dm_loss = dm_loss if not torch.isnan(dm_loss) else 0
                    domain_loss += dm_loss
            else:
                domain_loss = criterion_domain(domain_output, domain)
            domain_losses.update(domain_loss.item(), bsz)
            loss = class_loss + domain_loss
        # update metric
        losses.update(loss.item(), bsz)
      
        # Adam
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # print info
        if (idx + 1) % opt.print_freq == 0:
            if not opt.domain:
                print('Baseline Train: [{0}][{1}/{2}]\t'
                        'BT {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                        'DT {data_time.val:.2f} ({data_time.avg:.2f})\t'
                        'loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                        epoch, idx + 1, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses))
            else:
                print('Baseline Train: [{0}][{1}/{2}]\t'
                        'BT {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                        'DT {data_time.val:.2f} ({data_time.avg:.2f})\t'
                        'loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'domain_loss {domain_loss.val:.4f} ({domain_loss.avg:.4f})'.format(
                        epoch, idx + 1, len(train_loader), batch_time=batch_time,
                        data_time=data_time, loss=losses, domain_loss=domain_losses))
            sys.stdout.flush()
    return losses.avg, domain_losses.avg

def train_encoder_dg(train_loader, encoder, classifier, criterion, criterion_domain, optimizer, epoch, opt):
    """one epoch training"""
    encoder.train()
    classifier.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    supcon_losses = AverageMeter()
    domain_losses = AverageMeter()

    domains = []
    domain_outputs = []

    end = time.time()
    len_dataloader = len(train_loader)
    for idx, (data, label, domain) in enumerate(train_loader):

        data_time.update(time.time() - end)

        data = torch.from_numpy(np.concatenate([data[0], data[1]], axis=0)).type(torch.float).unsqueeze(2)

        label = [opt.class_remap_dict[l.item()] for l in label]
        label = torch.as_tensor(label).type(torch.int64)

        domain = [opt.domain_remap_dict[d.item()] for d in domain]       
        domain = torch.as_tensor(domain).type(torch.int64)

        if opt.cuda:
            data = data.to(opt.device)
            label = label.to(opt.device)
            domain = domain.to(opt.device)
        bsz = label.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        unsplit_proj, features = encoder(data)
        f1, f2 = torch.split(unsplit_proj, [bsz, bsz], dim=0)
         
        proj = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        if opt.method == 'SupCon':
            supcon_loss = criterion(proj, label)
            supcon_losses.update(supcon_loss.item(), bsz)
        elif opt.method == 'SimCLR':
            loss = criterion(proj)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))

        if opt.lambda_factor > 0:
            p = float(idx + epoch * len_dataloader) / opt.epochs / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            classifier.revgrad.adjust_lambda_(alpha * opt.lambda_factor)

            _, domain_output, _ = classifier(features, output_dm=True)
            domain = domain.repeat(1, 2).squeeze(0)
            
            domain_loss = criterion_domain(domain_output, domain)
            domain_losses.update(domain_loss.item(), bsz)

            loss = supcon_loss + domain_loss

            domain_outputs.append(domain_output.cpu())
            domains += domain.tolist()
        else:
            loss = supcon_loss

        # update metric
        losses.update(loss.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('SupCon Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                  'DT {data_time.val:.2f} ({data_time.avg:.2f})\t'
                  'loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'supcon_loss {supcon_loss.val:.4f} ({supcon_loss.avg:.4f})\t'
                  'domain_loss {domain_loss.val:.4f} ({domain_loss.avg:.4f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, supcon_loss=supcon_losses, domain_loss=domain_losses))
            sys.stdout.flush()

    if opt.lambda_factor > 0:
        metrics = dict()
        domain_outputs = torch.cat(domain_outputs).reshape(-1, opt.num_domains-1).detach()
        domains_tensor = torch.LongTensor(domains).detach()
        domain_outputs_tensor = torch.FloatTensor(domain_outputs).detach()
        domain_preds = domain_outputs_tensor.max(1)[1].type_as(domains_tensor).detach()
        metric_type = "binary" if opt.num_domains < 2 else "weighted"
        for key, value in metric_calc(domain_losses.avg, domains_tensor, domain_preds, metric_type).items():
            metrics["train/enc_dm_%s"% (key)] = value
        print(metrics)

    return supcon_losses.avg

def train_classifier(train_loader, encoder, classifier, criterion_class, optimizer, epoch, opt):
    """one epoch train"""

    encoder.eval()
    classifier.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    class_losses = AverageMeter()

    end = time.time()
    for idx, (data, label, domain) in enumerate(train_loader):
        data_time.update(time.time() - end)
        domain = [opt.domain_remap_dict[d.item()] for d in domain]
        label = [opt.class_remap_dict[l.item()] for l in label]
        data = torch.from_numpy(np.concatenate([data[0], data[1]], axis=0)).type(torch.float).unsqueeze(2)
        label = torch.as_tensor(label).type(torch.int64)
        domain = torch.as_tensor(domain).type(torch.int64)

        if opt.cuda:
            data = data.to(opt.device)
            label = label.to(opt.device)
            domain = domain.to(opt.device)
        bsz = label.shape[0]

        with torch.no_grad():
            unsplit_proj, features = encoder(data)      
        label = label.repeat(1, 2).squeeze(0)
        classifier_output, _, _ = classifier(features)
        class_loss = criterion_class(classifier_output, label)
        loss = class_loss
        class_losses.update(class_loss.item(), bsz)
        
        # update metric
        losses.update(loss.item(), bsz)   

        # SGD
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('SupCon Classifier Train: [{0}][{1}/{2}]\t'
                    'BT {batch_time.val:.2f} ({batch_time.avg:.2f})\t'
                    'DT {data_time.val:.2f} ({data_time.avg:.2f})\t'
                    'loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                    epoch, idx + 1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses))
            sys.stdout.flush()
    return losses.avg, class_losses.avg

def eval_enc(val_loader, encoder, criterion, epoch, opt):
    valid_loss, valid_data_time, valid_batch_time = eval_enc_one_epoch(val_loader, encoder, criterion, opt)
    
    metrics = {}
    metrics["valid/e_loss"] = valid_loss
    metrics['valid/BT'] = [valid_batch_time.val, valid_batch_time.avg]
    metrics['valid/DT'] = [valid_data_time.val, valid_data_time.avg]
    log_result(opt.wb, wandb, metrics, epoch)

    return metrics

def eval_enc_one_epoch(data_loader, encoder, criterion, opt):
    """one epoch training"""
    encoder.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()
    
    with torch.no_grad():
        for idx, (data, label, _) in enumerate(data_loader):
            data_time.update(time.time() - end)
            data = torch.from_numpy(np.concatenate([data[0], data[1]], axis=0)).type(torch.float).unsqueeze(2)
            label = [opt.class_remap_dict[l.item()] for l in label]
            label = torch.as_tensor(label).type(torch.int64)
            if opt.cuda:
                data = data.to(opt.device)
                label = label.to(opt.device)
            bsz = label.shape[0]
            
            # compute loss
            unsplit_proj, _ = encoder(data)
            f1, f2 = torch.split(unsplit_proj, [bsz, bsz], dim=0)
            
            proj = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

            if opt.method == 'SupCon':
                loss = criterion(proj, label)
                losses.update(loss.item(), bsz)
            elif opt.method == 'SimCLR':
                loss = criterion(proj)
            else:
                raise ValueError('contrastive method not supported: {}'.
                                format(opt.method))
                
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            bsz = label.shape[0]
            losses.update(loss.item(), bsz)


    return losses.avg, data_time, batch_time

def eval_class(train_loader, val_loader, test_loader, encoder, classifier, criterion_class, epoch, opt):
    # backup_transform = train_loader.dataset.transform
    # train_loader.dataset.transform = test_loader.dataset.transform
    # train_loss, train_class_loss, train_labels, train_class_preds, train_data_time, train_batch_time = eval_class_one_epoch(train_loader, encoder, classifier, criterion_class, opt)
    # train_loader.dataset.transform = backup_transform
    backup_transform = val_loader.dataset.transform
    val_loader.dataset.transform = test_loader.dataset.transform
    valid_loss, valid_class_loss, valid_labels, valid_class_preds, valid_data_time, valid_batch_time = eval_class_one_epoch(val_loader, encoder, classifier, criterion_class, opt)
    val_loader.dataset.transform = backup_transform
    test_loss, test_class_loss, test_labels, test_class_preds, test_data_time, test_batch_time = eval_class_one_epoch(test_loader, encoder, classifier, criterion_class, opt)

    metrics = {}
    # metric_type = "binary" if max(train_labels) < 2 else "weighted"
    metric_type = "weighted"
    # metrics["train/total_loss"] = train_loss
    metrics["valid/total_loss"] = valid_loss
    metrics["test/total_loss"] = test_loss
    # for key, value in metric_calc(train_class_loss, train_labels, train_class_preds, metric_type).items():
    #     metrics["train/c_%s"% (key)] = value
    for key, value in metric_calc(valid_class_loss, valid_labels, valid_class_preds, metric_type).items():
        metrics["valid/c_%s"% (key)] = value
    for key, value in metric_calc(test_class_loss, test_labels, test_class_preds, metric_type).items():
        metrics["test/c_%s"% (key)] = value

    # metrics['train/BT'] = [train_batch_time.val, train_batch_time.avg]
    # metrics['train/DT'] = [train_data_time.val, train_data_time.avg]
    metrics['valid/BT'] = [valid_batch_time.val, valid_batch_time.avg]
    metrics['valid/DT'] = [valid_data_time.val, valid_data_time.avg]
    metrics['test/BT'] = [test_batch_time.val, test_batch_time.avg]
    metrics['test/DT'] = [test_data_time.val, test_data_time.avg]
    log_result(opt.wb, wandb, metrics, epoch)

    return metrics

def eval_class_one_epoch(data_loader, encoder, classifier, criterion_class, opt):
    """one epoch training"""
    encoder.eval()
    classifier.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    class_losses = AverageMeter()

    labels = []
    class_outputs = []
    
    end = time.time()
    
    with torch.no_grad():
        for idx, (data, label, _) in enumerate(data_loader):
            data_time.update(time.time() - end)
            label = [opt.class_remap_dict[l.item()] for l in label]
            data = torch.as_tensor(data).type(torch.float).unsqueeze(2)
            label = torch.as_tensor(label).type(torch.int64)
            if opt.cuda:
                data = data.to(opt.device)
                label = label.to(opt.device)
            bsz = label.shape[0]
            
            # compute loss
            _, features = encoder(data)
            
            class_output, _, _ = classifier(features, output_dm=False)
            class_loss = criterion_class(class_output, label)
            class_losses.update(class_loss.item(), bsz)

            loss = class_loss          
                
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            bsz = label.shape[0]
            losses.update(loss.item(), bsz)
            
            class_outputs.append(class_output.cpu())
            labels += label.tolist()
 
        labels_tensor = torch.LongTensor(labels).detach()

        class_outputs = torch.cat(class_outputs).reshape(-1, opt.num_classes).detach()
        class_outputs_tensor = torch.FloatTensor(class_outputs).detach()
        class_preds = class_outputs_tensor.max(1)[1].type_as(labels_tensor).detach()

    return losses.avg, class_losses.avg, labels_tensor, class_preds, data_time, batch_time
    
def main():
    opt = parse_option()

    opt.cuda = not opt.no_cuda and torch.cuda.is_available()
    opt.device = torch.device(f"cuda:{opt.cuda_device}" if opt.cuda else "cpu")

    if opt.wb:
        wandb.init(project="project", entity="dg4mhealth")

    # build data loader
    train_loader, train_loader2, val_loader, test_loader = set_loader(opt)

    # build model and criterion
    encoder, classifier, criterion, criterion_class, criterion_domain = set_model(opt)

    # build optimizer
    _, optimizer_class, optimizer_whole = set_optimizers(opt, encoder, classifier)
    print('opt here:\n')
    print(opt)
    print('over')
    patience = opt.patience
    early_stop_count = 0
    min_valid_loss = np.inf

    # training routine
    metrics = {}
    save_file = os.path.join(opt.save_folder, 'best.pth')
    if not os.path.exists(save_file):
        for epoch in range(1, opt.epochs+1):
            adjust_learning_rate(opt, optimizer_whole, epoch)

            # train for one epoch
            time1 = time.time()
            loss_e = train_encoder_dg(train_loader, encoder, classifier, criterion, criterion_domain, optimizer_whole, epoch, opt)
            if opt.wb:              
                metrics["train/loss_e"] = loss_e
                wandb.log(metrics)
            time2 = time.time()
            print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

            metrics = eval_enc(val_loader, encoder, criterion, epoch, opt)

            if min_valid_loss >= metrics['valid/e_loss']:
                best_epoch = epoch
                min_valid_loss = metrics['valid/e_loss']
                early_stop_count = 0
                save_encoder(encoder, optimizer_whole, opt, epoch, save_file)
            else:
                early_stop_count += 1
            if early_stop_count >= patience:
                break

            # if min_train_loss >= loss_e:
            #     best_epoch = epoch
            #     min_train_loss = loss_e
            #     early_stop_count = 0
            #     save_encoder(encoder, optimizer_whole, opt, epoch, save_file)
            # else:
            #     early_stop_count += 1
            # if early_stop_count >= patience:
            #     break
    encoder, _, _ = load_encoder(encoder, optimizer_whole, opt.save_folder, ckpt=save_file)

    best_metrics = None
    best_epoch = None

    min_valid_loss = np.Inf
    early_stop_count = 0
    classifier_save_file = os.path.join(opt.classifier_folder, 'best.pth')
    for epoch in range(1, opt.classifier_epochs+1):
        time1 = time.time()
        if opt.baseline:
            train_baseline(train_loader, encoder, classifier, criterion_class, criterion_domain, optimizer_class, epoch, opt)
        else:
            train_classifier(train_loader, encoder, classifier, criterion_class, optimizer_class, epoch, opt)
        metrics = eval_class(train_loader, val_loader, test_loader, encoder, classifier, criterion_class, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        if min_valid_loss >= metrics['valid/c_loss']:
            best_metrics = metrics
            best_epoch = epoch
            min_valid_loss = metrics['valid/c_loss']
            early_stop_count = 0
            save_classifier(classifier, optimizer_class, opt, best_epoch, classifier_save_file)
        else:
            early_stop_count += 1
        if early_stop_count >= patience:
            break

    pprint('****Best Metrics****')
    log_best = dict()
    for key, value in best_metrics.items():
        log_best["best/%s"% (key)] = value
    log_result(opt.wb, wandb, log_best, best_epoch)

if __name__ == '__main__':
    #import wandb
    main()
