from config import path_config
# 计算有几个类别
def count_class_num_from_data_list(LIST):
    class_count = 0
    lables = {'':0}
    with open(LIST, 'r') as f:
        for line in f:
            line = line.strip()
            lable = line.split('_')[0]
            if lable in lables:
                pass
            else:
                class_count += 1
                lables[lable] = 1
    return class_count

if __name__ == '__main__':
    print(count_class_num_from_data_list(path_config.get_project_path() + path_config.get_test_lists('market')))