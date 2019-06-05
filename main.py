import os
import argparse
import datetime
import torch
import torch.nn as nn
import model
import torch.optim as optim
import torch.backends.cudnn as cudnn
import train
import mydataset
from tensorboardX import SummaryWriter

def load_chechpoint(filename):
    print("=> loading checkpoint '{}'".format(filename))
    checkpoint = torch.load(filename)
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(filename, checkpoint['epoch']))
    return checkpoint

def main():
    parser = argparse.ArgumentParser(description='CNN Home Eneregy Management System')
    # learning
    parser.add_argument('-lr', type=float, default=0.0001, help='initial learning rate [default: 0.001]')
    parser.add_argument('-adjust-lr', type=list,default=[0.001,0.0001],help='if you use adjust lr')
    parser.add_argument('-epochs', type=int, default=50, help='number of epochs for train [default: 256]')
    parser.add_argument('-test-batch-size',type=int, default=10000, help='batch size when you eval')
    parser.add_argument('-batch-size', type=int, default=128, help='batch size for training [default: 64]')
    parser.add_argument('-log-interval',  type=int, default=25,   help='how many steps to wait before logging training status [default: 1]')
    parser.add_argument('-test-interval', type=int, default=10, help='how many steps to wait before testing [default: 100]')
    parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
    parser.add_argument('-save-dir', type=str, default='model', help='where to save the snapshot')
    parser.add_argument('-early-stop', type=int, default=1000, help='iteration numbers to stop without performance increasing')
    parser.add_argument('-save-best', type=bool, default=False, help='whether to save when get best performance')
    parser.add_argument('-save-prefix',type=str,default='checkpoint_cnn')
    # data
    parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
    parser.add_argument('-output-normalization',default=True,help='is output_normalized? [default:False]')
    parser.add_argument('-normalization-factor-path',type=str,default='../data/seq_data/all/stat/normalization_factors5.csv')
    # model
    parser.add_argument('-dropout', type=float, default=0, help='the probability for dropout [default: 0.5]')
    parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
    parser.add_argument('-input-dim', type=int, default=31, help='number of embedding dimension [default: 128]')
    parser.add_argument('-input-num',type=int, default=15, help='input size')
    parser.add_argument('-output-num',type=int, default=5, help='output size')
    parser.add_argument('-kernel-num', type=int, default=300, help='number of each kind of kernel')
    parser.add_argument('-kernel-sizes', type=str, default='3', help='comma-separated kernel size to use for convolution')
    parser.add_argument('-static', action='store_true', default=True, help='fix the embedding')
    # device
    parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
    parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
    # option
    parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
    parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
    parser.add_argument('-test', action='store_true', default=False, help='train or test')
    parser.add_argument('--start-epoch', default=1, type=int,
                        help='If you load a model, program write log from loaded epoch')
    parser.add_argument('--t',type=int,default=15)
    parser.add_argument('-resume',type=str,default='')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    cudnn.benchmark = True
    cudnn.deterministic = True
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    writer = SummaryWriter('./logs/test_121314_50/')#0.01,0.001,0.0001
    args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
    args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    #call the dataset
    train_dataset = mydataset.CustomDataset(args,"../data/dataset_seq/normalized/dataset_train.csv")
    test_dataset = mydataset.CustomDataset(args,"../data/dataset_seq/normalized/dataset_test_"+str(args.t)+".csv")
    train_loss_dataset = mydataset.CustomDataset(args,"../data/dataset_seq/normalized/dataset_train_loss.csv")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)
    train_loss_loader = torch.utils.data.DataLoader(train_loss_dataset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

    cnn = model.CNN_HEMS(args).to(device)
    if args.snapshot is not None:
        print('\nLoading model from {}...'.format(args.snapshot))
        cnn.load_state_dict(torch.load(args.snapshot))
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(cnn.parameters(), lr=args.lr)


    if args.resume:
        if os.path.isfile(args.resume):
            checkpoint=load_chechpoint(args.resume)
            cnn.load_state_dict(checkpoint)
            # optimizer.load_state_dict(checkpoint['optimizer'])
            # args.start_epoch = checkpoint['epoch']
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    for epoch in range(args.start_epoch,args.epochs):
        print('Epoch : ', epoch)
        adjust_learning_rate(optimizer,epoch,args.adjust_lr)
        train.train(args,train_loader,cnn,optimizer,criterion,test_loader,train_loss_loader,writer,epoch)
        for name, param in cnn.named_parameters():
            try:
                writer.add_histogram(name,param.clone().cpu().data.numpy(),epoch)
                writer.add_histogram(name+'grad',param.grad.clone().cpu().data.numpy(),epoch)
            except:
                print('param err')
                continue
        # test_loss,test_mae_loss, prediction_mae,prediction_rmse =train.eval(args,test_loader,cnn,criterion)
        # train_loss,train_mae_loss,_,_ =train.eval(args,train_loss_loader,cnn,criterion)
        # write_energy2tensorboard(train_loss,test_loss,prediction_rmse,epoch,writer,name='mse')
        # write_energy2tensorboard(train_mae_loss,test_mae_loss,prediction_mae,epoch,writer,name='_mae')
        train.save(cnn,args.save_dir,args.save_prefix,epoch)

def write_energy2tensorboard(train_loss,test_loss,err,epoch,writer,name=""):
    writer.add_scalars('Loss'+name,{'train loss': train_loss,'test loss': test_loss},epoch)
    writer.add_scalars('energy_prediction_mae'+name,{'elec':err[0],'water':err[1],'gas':err[2],'hot_water':err[3],'heating':err[4]},epoch)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def adjust_learning_rate(optimizer, epoch,adjust_lr):
    lr_idx = epoch//15

    try:
        lr = adjust_lr[lr_idx]
    except:
        lr = adjust_lr[-1]
    print("learning rate : {}".format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    main()