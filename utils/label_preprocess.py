import os
import shutil

# 将tumor文件名重新命名成label
# good 命名成 0_casexx.bmp
# bad 命名成 0_casexx.bmp

def label_tran(good_dir, bad_dir, target_dir):
    """

    :param good_dir: 良性目录
    :param bad_dir: 恶性目录
    :param target_dir: 最终图片存放的目录
    """
    for file in os.listdir(good_dir):
        if is_img_file(file):
            shutil.copy(os.path.join(good_dir, file), os.path.join(target_dir, '0_good_' + file))
    for file in os.listdir(bad_dir):
        if is_img_file(file):
            shutil.copy(os.path.join(bad_dir, file), os.path.join(target_dir, '1_bad_' + file))

def is_img_file(file_name):
    if file_name is None:
        return False
    file_name = str(file_name).lower()
    parts = file_name.split('.')
    if len(parts) == 0:
        return False
    return parts[len(parts) - 1] in ['bmp', 'jpg', 'jpeg', 'png']

def rename_test():
    '''
    测试重命名函数os.rename是否可以移动文件
    测试结论：相当于shell mv命令
    :return:
    '''
    source_path = '/run/media/kele/DataSSD/Code/multi-task/rank-reid/test_re.txt'
    target_path = '/run/media/kele/DataSSD/Code/multi-task/rank-reid/config/test_re2.txt'
    os.rename(source_path, target_path)

def copy_test():
    """
    测试shutil.copyfile
    测试结论:可以拷贝文件
    """
    source_path = '/run/media/kele/DataSSD/Code/multi-task/rank-reid/test_re.txt'
    target_path = '/run/media/kele/DataSSD/Code/multi-task/rank-reid/config/test_re2.txt'
    shutil.copy(source_path, target_path)

def tumor_label_process():
    # label_tran(good_dir='/run/media/kele/DataSSD/Code/multi-task/rank-reid/datasource/cls-samples/train-image/original-image/good',
    #            bad_dir='/run/media/kele/DataSSD/Code/multi-task/rank-reid/datasource/cls-samples/train-image/original-image/bad',
    #            target_dir='/run/media/kele/DataSSD/Code/multi-task/rank-reid/datasource/tumor-data')
    label_tran(
        good_dir='/run/media/kele/DataSSD/Code/multi-task/rank-reid/datasource/cls-samples/test-image/after-resize432x432/good',
        bad_dir='/run/media/kele/DataSSD/Code/multi-task/rank-reid/datasource/cls-samples/test-image/after-resize432x432/bad',
        target_dir='/run/media/kele/DataSSD/Code/multi-task/rank-reid/datasource/tumor-data/test')

if __name__ == '__main__':
    tumor_label_process()
