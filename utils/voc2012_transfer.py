# Select useful from voc2012 dataset.
import os

VOC2012_BASE_PATH = '/run/media/kele/Data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012'


def count_animal_seg_cls(voc2012_base):
	'''
	print:
	     eg.
	           cow:
	                   seg_train:45
	                   seg_trainval:82
	                   seg_val:23
	                   seg_train_index_save:xxx/xxx.txt
	                   seg_trainval_index_save:xxx/xxx.txt
	                   seg_val_index_save:xxx/xxx.txt

	:param voc2012_base:
	:return:
	'''
	seg_txt_dir = os.path.join(voc2012_base, 'ImageSets', 'Segmentation')
	main_txt_dir = os.path.join(voc2012_base, 'ImageSets', 'Main')
	animals = ['cat', 'cow', 'dog', 'horse', 'sheep']

	save_dir = './cls-seg'
	if(not os.path.exists(save_dir)):
		os.makedirs(save_dir)

	seg_train_set = set()
	seg_val_set = set()
	seg_trainval_set = set()

	seg_train_txt_path = os.path.join(seg_txt_dir, 'train.txt')
	seg_trainval_txt_path = os.path.join(seg_txt_dir, 'trainval.txt')
	seg_val_txt_path = os.path.join(seg_txt_dir, 'val.txt')

	with open(seg_train_txt_path) as file:
		line  = file.readline()
		while line:
			line = line.split(' ')[0]
			seg_train_set.add(line)
			line = file.readline()

	with open(seg_trainval_txt_path) as file:
		line = file.readline()
		while line:
			# line = line.split(' ')[0]
			seg_trainval_set.add(line)
			line = file.readline()

	with open(seg_val_txt_path) as file:
		line = file.readline()
		while line:
			line = line.split(' ')[0]
			# print(line)
			seg_val_set.add(line)
			line = file.readline()

	for anm in animals:
		txt_path = os.path.join(main_txt_dir, anm + '_train.txt')
		# train data
		train_set = set()
		trainval_set = set()
		val_set = set()
		with open(txt_path) as file:
			line = file.readline()
			while line:
				line =	line.split(' ')[0]
				# print(line)
				if line in seg_train_set:
					train_set.add(line)
				elif line in seg_trainval_set:
					trainval_set.add(line)
				elif line in seg_val_set:
					val_set.add(line)
				line = file.readline()
		txt_path = os.path.join(main_txt_dir, anm + '_trainval.txt')
		with open(txt_path) as file:
			line = file.readline()
			while line:
				line =	line.split('\s')[0]
				if line in seg_train_set:
					train_set.add(line)
				elif line in seg_trainval_set:
					trainval_set.add(line)
				elif line in seg_val_set:
					val_set.add(line)
				line = file.readline()
		txt_path = os.path.join(main_txt_dir, anm + '_val.txt')
		with open(txt_path) as file:
			line = file.readline()
			while line:
				line = line.split('\s')[0]
				if line in seg_train_set:
					train_set.add(line)
				elif line in seg_trainval_set:
					trainval_set.add(line)
				elif line in seg_val_set:
					val_set.add(line)
				line = file.readline()
		anm_save_dir = os.path.join(save_dir, anm)
		try:
			os.makedirs(anm_save_dir)
		except:
			print(anm_save_dir + 'already existed')
		finally:
			print('Save to ' + anm_save_dir)
		train_file = open(os.path.join(anm_save_dir, 'train.txt'), mode='w+')
		trainval_file = open(os.path.join(anm_save_dir, 'trainval.txt'), mode='w+')
		val_file = open(os.path.join(anm_save_dir, 'val.txt'), mode='w+')
		for idx in train_set:
			train_file.writelines(idx)
		for idx in trainval_set:
			trainval_file.writelines(idx)
		for idx in val_set:
			val_file.writelines(idx)


if __name__ == '__main__':
	count_animal_seg_cls(voc2012_base=VOC2012_BASE_PATH)
	print('ok')