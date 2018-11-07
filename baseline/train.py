from __future__ import division, print_function, absolute_import

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from random import shuffle
from utils import class_helper
import numpy as np
import tensorflow as tf
from config import path_config
from config import train_config
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.backend.tensorflow_backend import set_session
from keras.initializers import RandomNormal
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Input
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

# 0002    ———->每一个人的独特标签，如上三个人是同一个人，所以标签值相同。

# 这是另一个人，标签值为0023，不同。

# c1s1  ———–>是camera1 sequence1的缩写，共有c1,c2,c3,c4,c5,c6六个摄像机，每个摄像机又有数个的录像段，  这里是摄像机一的                                             第 一个录像段

# 000451      ————>是c1s1的第000451帧图片
#
# 03   ————>每一帧可能会框出好几个这样的bboxes，所以这是这一帧上第三个框。

def load_mix_data(LIST, TRAIN):

    print('load mix data...')
    images, labels = [], []
    with open(LIST, 'r') as f:
        last_label = -1
        label_cnt = -1
        last_type = ''
        for line in f:
            line = line.strip()
            img = line
            lbl = line.split('_')[0]
            cur_type = line.split('.')[-1]
            if last_label != lbl or last_type != cur_type:
                label_cnt += 1
            last_label = lbl
            last_type = cur_type
            img = image.load_img(os.path.join(TRAIN, img), target_size=[224, 224])
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            images.append(img[0])
            labels.append(label_cnt)

    img_cnt = len(labels)
    shuffle_idxes = range(img_cnt)
    shuffle(shuffle_idxes)
    shuffle_imgs = list()
    shuffle_labels = list()
    for idx in shuffle_idxes:
        shuffle_imgs.append(images[idx])
        shuffle_labels.append(labels[idx])
    images = np.array(shuffle_imgs)
    labels = to_categorical(shuffle_labels)
    print('load mix data finished.')
    return images, labels



def load_data(LIST, TRAIN):
    '''

    :param LIST:  .list文件路径
    :param TRAIN: 训练集路径
    :return:      图片和标签
    '''
    print('load_data...')
    print('list:', LIST)
    print('train set:', TRAIN)
    images, labels = [], []
    with open(LIST, 'r') as f:
        last_label = -1
        label_cnt = -1
        for line in f:
            line = line.strip()
            img = line
            print('image:', img)
            lbl = line.split('_')[0]
            if last_label != lbl:
                label_cnt += 1
            last_label = lbl
            print('label:', label_cnt)
            img = image.load_img(os.path.join(TRAIN, img), target_size=[224, 224])
            print('target size:','224 x 224')
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            images.append(img[0])
            labels.append(label_cnt)

    img_cnt = len(labels)
    print('image count:', img_cnt)
    shuffle_idxes = list(range(img_cnt))

    # shuffle：将序列或元组随机排列
    shuffle(shuffle_idxes)
    shuffle_imgs = list()
    shuffle_labels = list()
    for idx in shuffle_idxes:
        shuffle_imgs.append(images[idx])
        shuffle_labels.append(labels[idx])
    images = np.array(shuffle_imgs)
    labels = to_categorical(shuffle_labels)
    print('load_data finished.')
    return images, labels


def softmax_model_pretrain(train_list, train_dir, class_count, target_model_path):
    '''

    :param train_list: .list文件，列举所有图片，例如market_train.list
    :param train_dir:  训练图片位置，即train_list 中所有文件的位置 如Market-1501/train/
    :param class_count: 分类数
    :param target_model_path:
    :return:
    '''
    print('softmax_model pre_training...')
    images, labels = load_data(train_list, train_dir)
    config = tf.ConfigProto()

    # 改成False by zzc
    config.gpu_options.allow_growth = False

    sess = tf.Session(config=config)
    set_session(sess)

    # load pre-trained resnet50
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))

    x = base_model.output
    x = Flatten(name='flatten')(x)
    x = Dropout(0.5)(x)
    x = Dense(class_count, activation='softmax', name='fc8', kernel_initializer=RandomNormal(mean=0.0, stddev=0.001))(x)
    net = Model(inputs=[base_model.input], outputs=[x])

    for layer in net.layers:
        layer.trainable = True

    # pretrain
    batch_size = train_config.get_batch_size()
    print('batch size:', batch_size)
    train_datagen = ImageDataGenerator(
        shear_range=0.2,
        width_shift_range=0.2,  # 0.
        height_shift_range=0.2)

    net.compile(optimizer=SGD(lr=0.001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    net.fit_generator(
        train_datagen.flow(images, labels, batch_size=batch_size),
        steps_per_epoch=len(images) / batch_size + 1, epochs=train_config.get_epcho(),# epochs = 40 edited by zzc
    )
    print('Net saving...')
    net.save(target_model_path)
    print('Net saved, location:', target_model_path)
    print('softmax model pre_training finished')


def softmax_pretrain_on_dataset(source, project_path='/media/jojo/Code/rank-reid', dataset_parent\
        ='/media/jojo/Code/rank-reid'):
    # 原先代码：source, project_path='/home/cwh/coding/rank-reid', dataset_parent='/home/cwh/coding'
    '''

    :param source:  训练集名字
    :param project_path:
    :param dataset_parent:
    :return:
    '''
    print('source:', source)

    #class_count:分类数，这里就是被识别人的个数
    if source == 'market':
        # train_list = project_path + '/dataset/market_train_test.list'
        # train_dir = dataset_parent + '/Market-1501-test/train'
        train_list = project_path + path_config.get_test_lists(source)
        train_dir = project_path + path_config.get_train_dir(source)
        class_count = class_helper.count_class_num_from_data_list(train_list)
    elif 'tumor' == source:
        class_count = 2
        train_list = os.path.join(project_path, 'dataset/tumor_train.list')
        train_dir = os.path.join(project_path, 'datasource/tumor-data/train')
    elif source == 'grid':
        train_list = project_path + '/dataset/grid_train.list'
        train_dir = dataset_parent + '/grid_label'
        class_count = 250
    elif source == 'cuhk':
        train_list = project_path + '/dataset/cuhk_train.list'
        train_dir = dataset_parent + '/cuhk01'
        class_count = 971
    elif source == 'viper':
        train_list = project_path + '/dataset/viper_train.list'
        train_dir = dataset_parent + '/viper'
        class_count = 630
    elif source == 'duke':
        train_list = project_path + '/dataset/duke_train.list'
        train_dir = dataset_parent + '/DukeMTMC-reID/train'
        class_count = 702
    elif 'grid-cv' in source:
        cv_idx = int(source.split('-')[-1])
        train_list = project_path + '/dataset/grid-cv/%d.list' % cv_idx
        train_dir = dataset_parent + '/underground_reid/cross%d/train' % cv_idx
        class_count = 125
    elif 'mix' in source:
        train_list = project_path + '/dataset/mix.list'
        train_dir = dataset_parent + '/cuhk_grid_viper_mix'
        class_count = 250 + 971 + 630
    else:
        train_list = 'unknown'
        train_dir = 'unknown'
        class_count = -1
    softmax_model_pretrain(train_list, train_dir, class_count, '../pretrain/' + source + '_softmax_pretrain.h5')


if __name__ == '__main__':
    # sources = ['market', 'grid', 'cuhk', 'viper']
    # sources = ['market']
    sources = path_config.get_sources()
    for source in sources:
        softmax_pretrain_on_dataset(source, project_path='/run/media/kele/DataSSD/Code/multi-task/rank-reid', dataset_parent='/run/media/kele/DataSSD/Code/multi-task/rank-reid/dataset')
