import os
sources = ['voc2012']
project_path = '/run/media/kele/DataSSD/Code/multi-task/rank-reid'
dataset_parent = os.path.join(project_path, 'datasource/voc-data')


train_dirs = {'market':'/Market-1501-1/train'}
probe_dirs = {'market':'/Market-1501-1/probe'}
test_dir = {'market':'/Market-1501-1/test'}

test_lists = {'market':'/dataset/market_train_1.list'}
def get_sources():
    return sources
def get_train_dir(source):
    return train_dirs.get(source)
def get_test_lists(source):
    return test_lists.get(source)
def get_project_path():
    return project_path
def get_dataset_parent():
    return dataset_parent
