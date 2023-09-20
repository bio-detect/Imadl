import glob
import os
import shutil
import glob

import numpy as np


def is_exist(path):
    return os.path.isdir(path)
def ensure_exist(path):
    if not is_exist(path):
        os.makedirs(path)


f = open('label.txt','r')
lines = f.readlines()
count_1 = 0
count_0 = 0
for line in lines:
    data = line.split()
    path_txt = data[0]
    label_ = data[-1]
   
    txt_name = os.path.basename(path_txt)
    file_name = os.path.splitext(txt_name)[0]
    try:
        pos_name = file_name.split(':')[-1]
        pos_start = pos_name.split('-')[0]
        pos_end = pos_name.split('-')[-1]
        start = int(pos_start)
        end = int(pos_end)
    except:
        continue
    if label_ =='1':
        count_1 += 1
    if label_ == '0':
        count_0 += 1
    raw = []
    label = []
    data_start, end_start = None, None
    with open(path_txt) as f:
        if os.path.getsize(path_txt) != 0:
            for lines in f:
                data = lines.strip('\n').split()
                if '>' not in data[0]:
                    chorm, pos, raw_data = data[0], data[1], [float(i) for i in data[2:]]
                    raw_data = raw_data if len(raw_data) == 6 else raw_data + [0]
                    if data_start is None:
                        data_start = int(pos)
                    data_end = int(pos)

                    if int(pos) > end:
                        break
                    raw.append(raw_data)
                if label_ == '1':
                    label.append(1)
                    
                if label_ == '0':
                    label.append(0)
                     

        else:
            continue
    padding = lambda :[[0] * 6]
    if data_start <= start and data_end >= end:
        raw = raw[start - data_start : end - data_start + 1]
        label = label[start - data_start : end - data_start + 1]
    elif data_start <= start and data_end <= end:
        raw = raw[start - data_start:data_end - data_start + 1] + padding() * (end - data_end)
        label = label[start - data_start: data_end - data_start + 1] + [0] * (end - data_end)
    elif data_start >= start and data_end <= end:
        raw = padding() * (data_start - start) + raw + padding() * (end - data_end)
        label = [0] * (data_start - start) + label + [0] * (end - data_end)
    else:
        raw = padding() * (data_start - start) + raw
        label = [0] * (data_start - start) + label
    raw_arr = np.array(raw)
    label_arr = np.array(label)
    label_arr = label_arr[:,np.newaxis]
    raw_all = np.hstack((raw_arr, label_arr))
    path_raw_name = (file_name + '_raw.txt')
    chr_path = "chr_dataset"
    if os.path.exists(chr_path + '/' + path_raw_name):
        continue
    f_raw = open(path_raw_name, "w")
    for i in raw_all:
        i = str(i).strip('[').strip(']').replace("'", "") + '\n'
        f_raw.writelines(i)
    ensure_exist(chr_path)
    f_raw.close()

    shutil.move(path_raw_name, chr_path)
#print('label1:',count_1)
#print('label0:',count_0)
