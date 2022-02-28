import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    # parser.add_argument('--model_path', type=str, default='Model_1', help='one model per file')
    parser.add_argument('--gpu', type=str, default='1', help='id of gpu device(s) to be used')
    # parser.add_argument('--log_info', type=str, default='Exp1')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--base_lr', type=float, default=0.001, help='basic learning rate')
    parser.add_argument('--lr_factor', type=float, default=1, help='lr_factor')
    parser.add_argument('--fix_bn', type=bool, default=True, help='fix batch normalization')
    parser.add_argument('--alpha', type=float, default=1, help='mes loss weight')
    parser.add_argument('--beta', type=float, default=5, help='ce loss weight')

    # rgs_branch
    parser.add_argument('--topk', type=int, default=5, help='number of patterns that need to be chosen in rgs branch')
    parser.add_argument('--drop_rate', type=float, default=0.5, help='dropout rate in rgs branch')


    # Batchsize and epochs
    parser.add_argument('--num_epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--num_workers', type=int, default=2, help='number of subprocesses for dataloader')
    parser.add_argument('--train_batch_size', type=int, default=32, help='batch size for training phase')
    parser.add_argument('--test_batch_size', type=int, default=16, help='batch size for test phase')

    # Dataset
    parser.add_argument('--dataset_path', type=str, default='/mnt/gdata/AQA/AQA-7', help='path to AQA7 dataset')

    # pretrained model
    parser.add_argument("--pretrained_i3d_weight", type=str,
                        default='/home/proto/liym/pretrained_models/i3d_model_rgb.pth',
                        help='pretrained i3d model')
    return parser

# 第一步：克隆Repo，配置环境
# 第二步：创建data文件夹，下载pt文件，下载FRFS数据集，添加ln链接，保证格式如下

# python train_net.py --gpu 0,1,2,3