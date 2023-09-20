import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import random
import json
from torch.utils.data import Dataset, DataLoader
import os
from typing import TextIO
from model import Model_test,Model
from pre_dataloader import testDataset

def mk(path):
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + ' output directory was created successfully')
    else:
         print(path + ' output directory has been created ')

def resolve_path(
    path: str,
    bam_file_path: str,
    **kwargs
):
    """ 把参数拼接到路径上 """
    args = [path, bam_file_path] + [
        '-%s %d' % (k, v)
        for k, v in kwargs.items()
    ]
    return ' '.join(args)

def handle_one_batch(fp: TextIO,nametemp):
   #header = fp.readline()
    #if not header:
    #    return None
    data = []
    name = nametemp
    sum_0,sum_1,sum_2,sum_3,sum_4,sum_5 = 0,0,0,0,0,0
    print_line = []
    while True:
        line = fp.readline().strip()
        if len(line) != 0:
            if '>' in line:
            
                if name == []:
                    name.append(line.strip('>'))
                elif len(data) == 2000:
                    nametemp = line.strip('>')
                    avg_0,avg_1,avg_2,avg_3,avg_4,avg_5 = sum_0/2000,sum_1/2000,sum_2/2000,sum_3/2000,sum_4/2000,sum_5/2000
                    print_line.extend((avg_0,avg_1,avg_2,avg_3,avg_4,avg_5))

                    return name,data,print_line,nametemp
                else:
                    data.clear()
                    next_name = []
                    next_name.append(line.strip('>'))
                    name = next_name
            else:
                tmp = line.split('\t')
                line = [float(j) for j in tmp[2:]]
               
                sum_0 += line[0]
                sum_1 += line[1]
                sum_2 += line[2]
                sum_3 += line[3]
                sum_4 += line[4]
                sum_5 += line[5]
              
                data.append(line)
                if len(data)==2000:
                    avg_0, avg_1, avg_2, avg_3, avg_4, avg_5 = sum_0 / 2000, sum_1 / 2000, sum_2 / 2000, sum_3 / 2000, sum_4 / 2000, sum_5 / 2000
                    print_line.extend((avg_0, avg_1, avg_2, avg_3, avg_4, avg_5))
                    return name,data,print_line,nametemp
        else:
            break
    return None,None,None,None 

def to_tensor(batches_data):
    """ 把Python的数据结构转化为Tensor """
    batches_data = torch.Tensor(batches_data)
    return batches_data

def load_data(
    path: str,
    batch_size: int
):
    with os.popen(path, 'r') as fp:
        finish = False
        while not finish:
            batches_name = []
            batches_data = []
            batches_line = []
            nametemp = []
            for i in range(batch_size):
                batch_name, batch_data, batch_line,nametemp = handle_one_batch(fp,nametemp)
         
                if batch_name is None or batch_data is None:
                    finish = True
                    break
                batches_name.append(batch_name)
                batches_data.append(batch_data)
                batches_line.append(batch_line)
            if batches_name:
                yield batches_name, to_tensor(batches_data),batches_line
        return 0,0,0



def test(module,args,abs_result_path,load_data,path):
    module.eval()
    tt = 0
    ff = 0
    file = open(abs_result_path,'w')
    with torch.no_grad():
        module.load_state_dict(torch.load(args.model)["state_dict"], False)
        for names, data, lines in load_data(path, args.batch_size):
            if names == 0:
                break
            if args.use_gpu:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                if(device != 'cpu'):
                    data = data.to(device)
                else:
                    print("This device doesn`t have GPU, the model will run at CPU")
 
            if data.size()[1] == 2000:
                outputs = module(data)
        
                _, predicted = torch.max(outputs.data, dim=1)
                for i in range(data.size()[0]):
                    m = nn.Softmax(dim=1)
                    outputs_softmax = m(outputs[:][i].reshape(1,2))
                    out_list = outputs_softmax.tolist()[0]
                    s_tr = "\t".join(str(i) for i in out_list)
                    if predicted[i].item() == 1:
                        tt += 1
                        names[i] = ",".join(str(x) for x in names[i])
                        lines[i] = ",".join(str(x) for x in lines[i])
                        lines[i] = lines[i].replace(',','\t')
                        file.write(names[i] + '\t' + str(1) + '\t' + s_tr + '\t' + lines[i] + '\n')
                        '''for j in range(data[i].shape[0]):
                            datas = str(data[i][j, :]).replace('.','')[8:-19]
                            file.write(datas.replace(',', '\t',5))
                            file.write('\n')'''
                    else:
                        ff += 1
                        names[i] = ",".join(str(x) for x in names[i])
                        lines[i] = ",".join(str(x) for x in lines[i])
                        lines[i] = lines[i].replace(',','\t')
                        file.write(names[i] + '\t' + str(0) + '\t' + s_tr + '\t' + lines[i] + '\n')
                        '''for j in range(data[i].shape[0]):
                            datas = str(data[i][j, :]).replace('.','')[8:-19]
                            file.write(datas.replace(',', '\t',5))
                            file.write('\n')'''
            else:
                print(data.size()[1])
                print('不足2000：',names)
                continue
    file.close()
    print("The number Predicted to have breakpoints", tt)
    print("The number predicted to have no breakpoints：", ff)

def test_2(module,args,abs_result_path,load_data,path):
    module.eval()
    tt = 0
    ff = 0
    file = open(abs_result_path,'w')
    with torch.no_grad():
        module.load_state_dict(torch.load(args.model)["state_dict"], False)
        for names, data, lines in load_data(path, args.batch_size):
            if names == 0:
                break
            if args.use_gpu:
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                if(device != 'cpu'):
                    data = data.to(device)
                else:
                    print("This device doesn`t have GPU, the model will run at CPU")

            if data.size()[1] == 2000:
                outputs = module(data)

                _, predicted = torch.max(outputs.data, dim=1)
                for i in range(data.size()[0]):
                    m = nn.Softmax(dim=1)
                    outputs_softmax = m(outputs[:][i].reshape(1,2))
                    out_list = outputs_softmax.tolist()[0]
                    s_tr = " ".join(str(i) for i in out_list)
                    names[i] = ",".join(str(x) for x in names[i])
                    file.write(names[i] + '\t' + s_tr + '\t' + str(predicted[i].item()) + '\n')

            else:
                print(data.size()[1])
                print('不足2000：',names)
                continue
    file.close()
       

def main(args):
    exec_path = args.input_path1
    bam_file_path = args.input_path2
    path = resolve_path(
        path=exec_path,
        bam_file_path=bam_file_path,
        i=args.num_i,
        d=args.num_d,
        w=args.num_w,
        s=args.num_s,
        b=args.num_b,
      
        
    )
  
    module = Model()
    if args.use_gpu:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if(device != 'cpu'):
            module = module.to(device)
        else:
            print("This device doesn`t have GPU, the model will run at CPU")
    mk(args.output_path)
    result_path = args.output_path + '/normal-mis.txt'
    f = open(result_path,'w')
    f.close()
    abs_result_path = os.path.abspath(result_path)
    
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    test(module,args,abs_result_path,load_data,path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Predictting the Assembly errors')

    parser.add_argument('--input_path1',default='/scripts/bkpws/bkpws',help='test_path of ')
    parser.add_argument('--input_path2',default='/tgs_mapping_20653_TGS_mapping.bam',help='test_path of ')
    parser.add_argument('--num_i',default=500,  type=int, metavar='N')
    parser.add_argument('--num_d',default=500, type=int, metavar='N')
    parser.add_argument('--num_w',default=2000, type=int, metavar='N')
    parser.add_argument('--num_s',default=90, type=int, metavar='N')
    parser.add_argument('--num_b',default=1000, type=int, metavar='N')
    parser.add_argument('--output_path',help='results of the model predicts')
    parser.add_argument('--model',default='/result5/weights/epoch_16.checkpoint.pth.tar',help='test_path of trained model ')
    parser.add_argument('--restore',metavar='PATH', type=str, default=None,help='restore checkpoint')
    parser.add_argument('--num_workers',default=10,type=int,help='the number of cpu')

    parser.add_argument('--batch_size', default=64, type=int, metavar='N')
    parser.add_argument('--use_gpu',default=True, help='if use gpu default is True')


    args_ = parser.parse_args()

    if not os.path.exists(args_.output_path):
        os.makedirs(args_.output_path)
        print('output directory has been created in ',args_.output_path)
    else:
        print('output directory has been in ',args_.output_path)

    args = args_
    print("i:", args.num_i)
    print("d:", args.num_d)
    print("w:", args.num_w)
    print("s:", args.num_s)
    print("b:", args.num_b)
 
    main(args)
