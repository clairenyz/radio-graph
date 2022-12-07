import torch
import argparse
import numpy as np
from torchvision.models import resnet34
import torch.nn as nn

from data import KneeGradingDatasetNew, KneeGradingDataSetUnsupervised, dataloaders_dict_init
from train import *
from utils import *
from attn_resnet.models.model_resnet import ResidualNet
from attn_resnet.focal import FocalLoss
from attn_resnet.wide_resnet import wide_resnet50_2
# Arguments
parser = argparse.ArgumentParser(description='Networks training')
parser.add_argument('-d', '--data', action="store", dest="data_folder", type=str,
                    help="Training data directory", default=None)
parser.add_argument('-dc', '--data-contents', action="store", dest="contents", type=str,
                    help="Validation data directory", default=None)
parser.add_argument('-unspc', '--unsup-content', action="store", dest="unsup_contents", type=str,
                    help="unsup data directory", default='/gpfs/data/denizlab/Users/bz1030/data/unsupervised_method_contents/nyu_data_unsup.csv')
parser.add_argument('-us', '--unsupervised', action="store_true", dest="unsupervised",
                    help="Whether to use unsupervised training")
parser.add_argument('-nus', '--no-unsupervised', action="store_false", dest="unsupervised",
                    help="Whether to use unsupervised training")
parser.set_defaults(unsupervised=False)
parser.add_argument('-bs', '--batch-size', action="store", dest="batch_size", type=int, 
                    help="Batch size for training", default=8)
parser.add_argument('-b', '--beta', action="store", dest="beta", type=float,
                    help="Beta coefficient for KLD loss", default=1)
parser.add_argument('-vi', '--val-interval', action="store", dest="val_interval", type=int, 
                    help="Number of epochs between validation", default=1)
parser.add_argument('-ep', '--epochs', action="store", dest="epochs", type=int, 
                    help="Number of epochs for training", default=15)
parser.add_argument('-lr', '--learning-rate', action="store", dest="learning_rate", type=float, 
                    help="Learning rate", default=0.0001)
parser.add_argument('-lm', '--load-model', action="store", dest="load_model", type=bool, 
                    help="Whether to load model", default=False)
parser.add_argument('-md', '--model-dir', action="store", dest="model_dir", type=str, 
                    help="Where to load the model", default=None)
parser.add_argument('-n', '--run-name', action="store", dest="run_name", type=str, 
                    help="Name for this run", default='default')
parser.add_argument('-m', '--model', action='store', dest='model_type', type=str,
                    default='baseline', choices=['baseline', 'CBAM', 'wide_resnet'],
                    help='Choose which model to train')
parser.add_argument('-au', '--augmentation', action='store_true',
                    dest='augmentation', help='If apply augmentation on training')
parser.add_argument('-nau', '--no-augmentation', action='store_false',
                    dest='augmentation', help='If apply augmentation on training')
parser.set_defaults(augmentation=False)
parser.add_argument('-do','--dropout',  action='store_true',
                    dest='dropout', help='If apply dropout from linear layer (Oulu lab)')
parser.add_argument('-ndo','--no-dropout',  action='store_false',
                    dest='dropout', help='If apply dropout from linear layer (Oulu lab)')
parser.set_defaults(dropout=False)
parser.add_argument('-fl', '--freeze-layer',  action='store', type=int, default=0,
                    dest='freeze_layers', help='If only train layer number > this number')
parser.add_argument('-lf', '--loss-function',  action='store', default='CE',
                    dest='loss_function', help='FL or CE', choices=['FL', 'CE', 'wCE'])
parser.add_argument('-fg', '--focal-gamma',  action='store', default=1, type=int,
                    dest='fl_gamma', help='Focal loss gamma parameters')
parser.add_argument('-dt', '--data-type',  action='store', default='float', type=str,
                    dest='data_type', help='Use integer or float.')
parser.add_argument('-exp', '--experiment', action='store', default='normal', type=str,
                    choices=['normal', 'leq2', 'gcn'], dest='exp')
parser.add_argument('-dm', '--demo', action='store', dest='demo', type=str2bool,
                    default='yes')
parser.add_argument('-unsn', '--unsupervised-number', action='store', dest='unsup_num', type=int,
                    default=None)

# CMD 20211118 --> matching the UDA implementation in github
parser.add_argument('-ust', '--uda_softmax_temp', action='store', dest='uda_softmax_temp', type=float, default=0.4,
                    help='The temperature of the Softmax when making prediction on unlabeled '
                    'examples. -1 means to use normal Softmax')
parser.add_argument('-usct', '--uda_confidence_thresh', action='store', dest='uda_confidence_thresh', type=float, default=0.6,
                    help='The threshold on predicted probability on unsupervised data. If set,'
                    'UDA loss will only be calculated on unlabeled examples whose largest'
                    'probability is larger than the threshold.')

parser.add_argument('-lam0', '--lam0', action='store', dest='lam1', type=float, default=1,
                    help='Weight of the KL loss')
parser.add_argument('-lam1', '--lam1', action='store', dest='lam1', type=float, default=-1,
                    help='Weight of the entropy loss')
parser.add_argument('-lam2', '--lam2', action='store', dest='lam2', type=float, default=0,
                    help='Weight of the divertisty term loss')

def main():

    seed = 1234
    torch.manual_seed(seed)
    np.random.seed(seed)
    # Import parameters
    np.set_printoptions(suppress=True)
    args = parser.parse_args()
    assert args.loss_function in ['CE','FL', 'wCE']
    # Check device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Computation device: ", device)
    # Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

    # Instantiate network
    print('Model configuration: model_type:{}; dropout:{}; Training augmentation:{}'\
          .format(args.model_type, args.dropout, args.augmentation))

    if args.exp in ['normal', 'gcn']:
        out_label = 5
    elif args.exp == 'leq2':
        out_label = 3

    if args.model_type == 'baseline':
        net = resnet34(pretrained=True)
        num_ftrs = net.fc.in_features
        if args.dropout:
            net.fc = nn.Sequential(nn.Dropout(0.4),
                                   nn.Linear(num_ftrs, out_label))
        else:
            net.fc = nn.Sequential(nn.Linear(num_ftrs, out_label))
        net.avgpool = nn.AvgPool2d(28)
    elif args.model_type == 'CBAM':
        model = resnet34(pretrained=True)
        net = ResidualNet('ImageNet', 34, 1000, args.model_type)
        load_my_state_dict(net, model.state_dict())
        del model
        num_ftrs = net.fc.in_features
        if args.dropout:

            net.fc = nn.Sequential(nn.Dropout(0.4),
                                   nn.Linear(num_ftrs, out_label))
        else:
            net.fc = nn.Sequential(nn.Linear(num_ftrs, out_label))
    elif args.model_type == 'wide_resnet':
        net = wide_resnet50_2(pretrained=True)
        net.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(2048, out_label)
        )
    else:
        raise ValueError('Check the model_type arguments. Wrong input:', args.model_type)

    # print(net)

    ct = 0
    for child in net.children():
        # print(ct, child)
        ct += 1
        if ct < args.freeze_layers:
            for param in child.parameters():
                param.requires_grad = False

    net = net.to(device)

    # Dataloaders
    csv_dir_dict = args.contents
    data_dir_dict = args.data_folder
    print('Get contents from {}; Get data from {}'.format(csv_dir_dict, data_dir_dict))
    dataloaders_dict = dataloaders_dict_init(csv_dir_dict, data_dir_dict, args)

    # Optimizer and loss function
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    if args.loss_function == 'CE':
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(device)
    elif args.loss_function == 'FL':
        criterion = FocalLoss(gamma=args.fl_gamma, alpha=0.5, num_classes=5)
    elif args.loss_function =='wCE':
        train_csv = clean(pd.read_csv(csv_dir_dict + 'train_DONE.csv'))
        #####################
        # Construct Weights #
        #####################
        class_count_df = train_csv.groupby('KLG').count()
        print('&' *20)
        print(class_count_df)

        n_0, n_1, n_2, n_3, n_4 = class_count_df.iloc[0, 0], class_count_df.iloc[1, 0], class_count_df.iloc[2, 0], class_count_df.iloc[3, 0], class_count_df.iloc[4, 0]
        n_sum = n_0 + n_1 + n_2 + n_3 + n_4
        w_0 = n_sum / (5.0 * n_0)
        w_1 = n_sum / (5.0 * n_1)
        w_2 = n_sum / (5.0 * n_2)
        w_3 = n_sum / (5.0 * n_3)
        w_4 = n_sum / (5.0 * n_4)

        print('Weighted CE is used with following weights:')
        print(n_0, w_0, n_1, w_1, n_2, w_2, n_3, w_3, n_4, w_4)

        # Important: Convert Weights To Float Tensor
        class_weights=torch.FloatTensor([w_0, w_1, w_2, w_3, w_4]).cuda()
        criterion = nn.CrossEntropyLoss(weight=class_weights)


    criterion = criterion.to(device)
    # Model training, comments corresponding line to train different method for semi-supervise learning
    # UDA
    train(net, dataloaders_dict, criterion, optimizer, args.val_interval, args.epochs, args.load_model,
             args.model_dir,
             args.run_name, args.unsupervised, args, args.beta, device)
    # multi task learning
    #train_mt(net, dataloaders_dict, criterion, optimizer, args.val_interval, args.epochs,
     #        args.load_model, args.model_dir,
    #        args.run_name, args.unsupervised,
     #        args, args.beta, device)


if __name__ == "__main__":
    main()
