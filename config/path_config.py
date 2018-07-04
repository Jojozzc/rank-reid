sources = ['market']
project_path = '/media/jojo/Code/rank-reid'
dataset_parent = '/media/jojo/Code/rank-reid'
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
