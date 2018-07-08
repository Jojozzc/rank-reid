from keras.models import load_model
from keras.layers import *
from keras.utils import *
from keras import engine
import numpy as np
from keras.preprocessing import image
import os
from keras import engine
from keras.applications.resnet50 import preprocess_input
import keras.engine.training

from keras.preprocessing.image import ImageDataGenerator
import json
# 0575_c3s2_009987_00.jpg
# 0002    ———->每一个人的独特标签，如上三个人是同一个人，所以标签值相同。
# 这是另一个人，标签值为0023，不同。

# c1s1  是camera1 sequence1的缩写，共有c1,c2,c3,c4,c5,c6六个摄像机，每个摄像机又有数个的录像段，  这里是摄像机一的                                             第 一个录像段

# 000451 是c1s1的第000451帧图片
#
# 03  每一帧可能会框出好几个这样的bboxes，所以这是这一帧上第三个框。


class single_model_output:
    output_array = []
    predict_label = '0002'
    def __init__(self, arr):
        self.output_array = arr

def label_list(train_list):
    labels = []
    with open(train_list) as f:
        for fname in f:
            label = str(fname)
            label = label.strip()
            label = label.split('_')
            label = label[0]
            if(label not in labels):
                labels.append(label)
    return labels
def label_dict(train_list):
    labels = []
    label_map = {}
    with open(train_list) as f:
        for fname in f:
            label = str(fname)
            label = label.strip()
            label = label.split('_')
            label = label[0]
            if(label not in labels):
                labels.append(label)
        for index in range(len(labels)):
            label_map[labels[index]] = index
    return label_map

def load_test_data(test_path, test_list):
    '''
    :param test_list: 文件名list
    :param test_path: 测试图片所在路径
    :return:
    '''
    images = []
    label_map = {}
    labels = []
    for fname in test_list:
        fname = str(fname)
        label = fname.strip()
        label = fname.strip('_')
        label = fname[0]
        if(label not in labels):
            labels.append(label)
        img = image.load_img(os.path.join(test_path, fname), target_size=[224, 224])
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        images.append(img[0])
    for index in range(len(labels)):
        label_map[labels[index]] = index
    return images, label_map
def load_single_img(img_path):
    img = image.load_img(img_path, target_size=[224, 224, 3])
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return np.array(img)
def extract_label_from_name(file_name):
    label = file_name.strip()
    label = label.split('_')
    label = label[0]
    return label

# def get_probe_data(probe_dir):


def evaluate(model_path, train_list_path, probe_list_path, probe_dir, log_dir=None, threadhold = 0.1, write_log_batch=20):
    labels = label_list(train_list_path)
    model = load_model(filepath=model_path)
    model.summary()
    probe_img_list = []
    identify_accurity = 0.0
    vertification_accurity = 0.0
    identify_test_num = 0
    vertification_num = 0
    identify_correct_num = 0
    vertification_correct_num = 0
    batch_count = 0
    log_name = 'market_probe.txt'
    log_path = os.path.join(log_dir, log_name)
    with open(probe_list_path, mode='r') as probe_list, open(log_path, 'a+') as log_file:
        for line in probe_list:
            probe_img_list.append(str(line).strip())

        for i in range(len(probe_img_list)):
            for j in range(i + 1, len(probe_img_list)):
                # identify_test_num += 2
                vertification_num += 1
                img_1 = load_single_img(os.path.join(probe_dir, probe_img_list[i]))
                img_2 = load_single_img(os.path.join(probe_dir, probe_img_list[j]))
                model_output = model.predict([img_1, img_2])
                out1 = np.array(model_output[0])
                out2 = np.array(model_output[1])
                index1 = out1.argmax();
                index2 = out2.argmax();
                bin_out = np.array(model_output[2])
                pdt_id1 = labels[index1]
                pdt_id2 = labels[index2]
                real_label1 = extract_label_from_name(probe_img_list[i])
                real_label2 = extract_label_from_name(probe_img_list[j])
                is_img1_in_train_set = real_label1 in labels
                is_img2_in_train_set = real_label2 in labels
                score1 = out1[0][index1]
                score2 = out2[0][index2]

                if is_img1_in_train_set:
                    identify_test_num += 1
                    if pdt_id1 == real_label1:
                        identify_correct_num += 1
                if is_img2_in_train_set:
                    identify_test_num += 1
                    if pdt_id2 == real_label2:
                        identify_correct_num += 1
                if (real_label1 == real_label2 and bin_out >= threadhold) or (real_label1 != real_label2 and bin_out <= threadhold / 2):
                    vertification_correct_num += 1

                input_str = 'input: img1:{img1}, in train set?({is_in1}), img2:{img2}, ({is_in2})'.format(img1=real_label1,
                                                                                                          img2=real_label2,
                                                                                                          is_in1=is_img1_in_train_set,
                                                                                                          is_in2=is_img2_in_train_set)
                output_str = 'output: predict1={}({}), score1={}, predict2={}({}), score2={}'\
                    .format(pdt_id1, pdt_id1==real_label1, score1, pdt_id2, pdt_id2==real_label2, score2)
                log_file.write(input_str + '\n')
                log_file.write(output_str + '\n')
                log_file.write(str(bin_out[0]) + '\n')
                log_file.write('\n')
                batch_count += 1
                if batch_count == write_log_batch:
                    log_file.flush()
                    batch_count = 0
                print(input_str)
                print(output_str)
                print('bin_out:', bin_out)
                print()
        if identify_test_num != 0:
            identify_accurity = identify_correct_num / identify_test_num
        if vertification_num != 0:
            vertification_accurity = vertification_correct_num / vertification_num
        log_file.write('identify num:{}, correct num:{}\n'.format(identify_test_num, identify_correct_num))
        log_file.write('vertification num:{}, correct num:{}\n'.format(vertification_num, vertification_correct_num))
        log_file.write('identify accurity:{}, vertification:{}\n'.format(identify_accurity, vertification_accurity))
        probe_list.close()
        log_file.close()
        print('identify num:{}, correct num:{}'.format(identify_test_num, identify_correct_num))
        print('vertification num:{}, correct num:{}'.format(vertification_num, vertification_correct_num))
        print('identify accurity:{}, vertification:{}'.format(identify_accurity, vertification_accurity))


def predict_test():
    model_path = '/media/jojo/Code/rank-reid/pretrain/market_pair_pretrain.h5'
    model = load_model(model_path)
    model.summary()
    probe_dir = "/media/jojo/Code/rank-reid/Market-1501/probe"

    img1 = load_single_img(os.path.join(probe_dir, '0001_c5s3_072862_00.jpg'))
    img2 = load_single_img(os.path.join(probe_dir, '0002_c2s1_068521_00.jpg'))
    img3 = load_single_img(os.path.join(probe_dir, '0004_c3s3_065819_00.jpg'))
    img4 = load_single_img(os.path.join(probe_dir, '0113_c6s1_018876_00.jpg'))

    res = model.predict([img1, img2], 2)

    print(type(res[0]))
    img1_output = np.array(res[0])
    for i in range(len(img1_output[0])):
        print(i)
        print(img1_output[0][i])
    print(img1_output.shape)
    print(img1_output.argmax(1))
    print(img1_output.max())

if __name__ == '__main__':
    # predict_test()
    # label_map = label_dict('/media/jojo/Code/rank-reid/dataset/market_train.list')
    # print(label_map)
    # arr = np.array([[0, 5, 9, 1, 15, 7]])
    # print(arr.shape)
    # print(arr.argmax())
    # print(arr.max())
    # print(arr[0][arr.argmax()])


    model_path = '/media/jojo/Code/rank-reid/pretrain/market_pair_pretrain.h5'
    evaluate(model_path, train_list_path='/media/jojo/Code/rank-reid/dataset/market_train.list',
             probe_list_path='/media/jojo/Code/rank-reid/dataset/probe_1.list',
             probe_dir='/media/jojo/Code/rank-reid/Market-1501/probe',
             log_dir='/media/jojo/Code/rank-reid/pretrain')