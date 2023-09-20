import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch import optim
import torch.backends.cudnn as cudnn
import glob
from torch.utils.tensorboard import SummaryWriter
import os.path
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import LambdaLR,MultiStepLR
import json
import argparse
import numpy as np
import random
import os
from dataloader import myDataset
from model import Model_test,Model
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def set_global_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(state, working_directory, epoch, step=None):
    directory = os.path.join(os.path.abspath(working_directory), 'weights')
    if not os.path.exists(directory):
        os.makedirs(directory)
    if step is None:
        torch.save(state, os.path.join(directory, 'epoch_{}.checkpoint.pth.tar'.format(epoch)))
    else:
        torch.save(state, os.path.join(directory, 'epoch_{}_step_{}.checkpoint.pth.tar'.format(epoch, step)))

def splitTrainAndVal(isUseTest,path):
    saved_path = "./"
    ftrainval = open(saved_path + 'trainval2.txt', 'w')
    ftest = open(saved_path + 'test2.txt', 'w')
    ftrain = open(saved_path + 'tra2.txt', 'w')
    fval = open(saved_path + 'val2.txt', 'w')

    total_files = os.listdir(path)
    print(len(total_files))
    trainval_files = []
    test_files = []

    if isUseTest:
        trainval_files, test_files = train_test_split(total_files, test_size=0.001, random_state=55)
    else:
        trainval_files = total_files

    for file in trainval_files:
        ftrainval.write(file + "\n")

    train_files, val_files = train_test_split(trainval_files, test_size=0.2, random_state=55)
    # train
    for file in train_files:
        ftrain.write(file + "\n")
    # val
    for file in val_files:
        fval.write(file + "\n")
    for file in test_files:
        print(file)
        ftest.write(file + "\n")
    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()


def train(epoch, train_loader, model, criterion, optimizer, logger, args):
    model.train()
    running_loss = 0
    all_loss = 0
    for i,(input_data,label) in enumerate(train_loader):
        if args.use_gpu:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if (device != 'cpu'):
                input_data = input_data.cuda('cuda:0')
                label = label.cuda('cuda:0')
            else:
                print("This device doesn`t have GPU, the model will run at CPU")           
        optimizer.zero_grad()
        outputs = model(input_data)
        label = torch.squeeze(label, dim  = 1)
        loss = criterion(outputs, label)
        global_step = epoch * len(train_loader) + i
        loss.backward()
        optimizer.step()
        n=1000
        running_loss += loss.item()
        all_loss += loss.item()
        
        if(i % n == n-1):
            print('[%d,%5d] loss: %0.3f' %(epoch + 1,i+1,running_loss/n))
     
            running_loss = 0
        if logger is not None and (global_step + 1) % args.print_freq == 0:
            logger.add_scalar('train/loss', loss.item(), global_step + 1) 
    
    print("平均loss：",all_loss/i)
        

def evaluate(val_loader, model, criterion, logger, epoch, args):
    model.eval()
    acc = []
    total,correct = 0,0
    tp,fp,tn,fn = 0,0,0,0
    label_list = []
    score_list = []
    with torch.no_grad():
        for step, (input_data,label) in enumerate(val_loader):
            if args.use_gpu:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                if (device == 'cpu'):
                    print("This device doesn`t have GPU, the model will run at CPU")
                input_data = input_data.to(device)
                label = label.to(device)
            outputs = model(input_data)
            _, predicted = torch.max(outputs.data, dim=1)
            total +=label.size(0)
            label = torch.squeeze(label, dim=1)
            label_list.extend(predicted.detach().cpu().numpy())
            score_list.extend(label.cpu().numpy())
            correct +=(predicted==label).sum().item()
            correct_rate =100 *  correct/total
            acc.append(correct_rate)
            for i in range(input_data.size()[0]):
                if predicted[i].item() == 1:
                    if label[i].item() == 1:
                        tp += 1
                    else:
                        fp += 1
                else:
                    if label[i].item() == 0:
                        tn += 1
                    else:
                        fn += 1
        print("tp:",tp,"fp:",fp,"tn:",tn,"fn:",fn)
        precision = tp / (tp + fp)
        specificity = tn / (tn + fp)
        recall = tp / (tp + fn)
        '''
        fpr, tpr, threshold = roc_curve(label_list, score_list)
        roc_auc = auc(fpr, tpr)  # 准确率代表所有正确的占所有数据的比值
        print('roc_auc:', roc_auc)
        lw = 2
        plt.subplot(1, 1, 1)
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='%d ROC curve (area = %0.2f)' % (epoch + 1,roc_auc))  # Specificity为横坐标，Sensitivity为纵坐标做曲线
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('1 - specificity')
        plt.ylabel('Sensitivity')
        plt.title('ROC', y=0.5)
        plt.legend(loc="lower right")
        figure_save_path =os.path.join(args.output_path, "file_fig")
        if not os.path.exists(figure_save_path):
            os.makedirs(figure_save_path)
        plt.savefig(os.path.join(figure_save_path, str(epoch + 1) + '.png'))
        plt.clf()
        '''
        if logger is not None:
            logger.add_scalar('eval/acc_mean', np.mean(acc), epoch + 1)
            logger.add_scalar('eval/acc_median', np.median(acc), epoch + 1)
        
        print('GPU{}/epoch{} precision={:.4f}%  specificity={:.4f}% recall={:.4f}%'.format('cuda:0', epoch + 1, precision*100,  specificity*100, recall*100))
        print('GPU{}/epoch{} fscore={:.4f}'.format('cuda:0', 1/((1/precision + 1/recall)/2)))    
        print('GPU{}/epoch{} mean_acc={:.4f}% median_acc={:.4f}%'.format('cuda:0', epoch + 1, np.mean(acc), np.median(acc)))


def main(args):
    #splitTrainAndVal(False,args.input_path)
    #train_set = myDataset_test(args.input_path,"tra2.txt")
    #valid_set = myDataset_test(args.input_path,"val2.txt")
    dataset = myDataset(args.input_path)
    train_set, valid_set = random_split(dataset, [int(len(dataset) * 0.8),len(dataset)-int(len(dataset) * 0.8)])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,)
    valid_loader = DataLoader(valid_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers,)
 
   
    model = Model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    logger_path = os.path.join(os.path.abspath(args.output_path), 'log')
    if not os.path.exists(logger_path):
        os.makedirs(logger_path)
    logger = SummaryWriter(logger_path)

    if args.use_gpu:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if(device != 'cpu'):
            model = model.to(device)
            criterion = criterion.to(device)
        else:
            print("This device doesn`t have GPU, the model will run at CPU")

    if args.seed is not None:
        set_global_seed(args.seed)
        print('setting seed={}'.format(args.seed))

    for epoch in range(args.start_epoch, args.epochs):
        print("This epoch is ",epoch)
        print("第%d个epoch的学习率：%f" % (epoch+1, optimizer.state_dict()['param_groups'][0]['lr']))

        if 0 <= epoch < 4:
            optimizer = optim.AdamW(model.parameters(), lr=args.lr)
        if 4 <= epoch < 8:
            optimizer = optim.AdamW(model.parameters(), lr=args.lr/2)
        if 8 <= epoch < 12:
            optimizer = optim.AdamW(model.parameters(), lr=args.lr/2/2)
        if epoch >= 12:
            optimizer = optim.AdamW(model.parameters(), lr=args.lr/2/2/2)

        train(epoch, train_loader, model, criterion, optimizer, logger, args)

        if (epoch + 1) % args.eval_freq == 0:
            evaluate(valid_loader, model, criterion, logger, epoch, args)

       
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, args.output_path, epoch + 1)



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Predictting the Assembly errors')

    parser.add_argument('--input_path',help='input_path of ')
    parser.add_argument('--output_path',help='the folder to save log and model weights')
    parser.add_argument('--restore',metavar='PATH', type=str, default=None,help='restore checkpoint')
    parser.add_argument('--num_workers',default=0,type=int,help='the number of cpu')

    parser.add_argument('--seed', default=40, type=int, help='seed for deterministic training (default: 40)')
    parser.add_argument('--epochs',default=10, type=int, metavar='N',)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',help='manual epoch number (useful on restart training)')
    parser.add_argument('--batch_size', default=64, type=int, metavar='N')
    parser.add_argument('--eval_batch_size',default=64, type=int, metavar='N')
    parser.add_argument('--lr',default=0.0001,type=float,metavar='LR',help='learning rate')

    parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--eval-freq', default=1, type=int, help='evaluation frequency (default: 1)')
    parser.add_argument('--use_gpu',default=True, help='if use gpu default is True')


    args_ = parser.parse_args()

    if not os.path.exists(args_.input_path):
        raise NotADirectoryError('input dataset directory is not valid')

    if not os.path.exists(args_.output_path):
        os.makedirs(args_.output_path)
        print('output directory has been created in ',args_.output_path)
    if os.path.exists(args_.output_path):
        print('output directory has been in ',args_.output_path)
    with open(os.path.join(args_.output_path, 'config.json'), 'w') as f:
        json.dump(args_.__dict__, f, indent=2)
    args = args_
    main(args)
