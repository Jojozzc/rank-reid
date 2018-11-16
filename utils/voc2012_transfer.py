# Select useful from voc2012 dataset.
import os
import shutil
from scipy import misc
from xml.dom.minidom import parse
import xml.dom.minidom


from utils import str_util, bounding_core


VOC2012_BASE_PATH = '/run/media/kele/Data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012'


def copy_file_by_list_rename(source_dir, target_dir, file_list, class_id):
	print('copy start')
	for file in file_list:
		if file:
			file = file + '.jpg'
			shutil.copy(os.path.join(source_dir, file), os.path.join(target_dir, str(class_id) + '_'+ file))

def gen_voc_data():
	voc2012_base_path = VOC2012_BASE_PATH
	main_dir = os.path.join(voc2012_base_path, 'ImageSets/Main')
	rank_reid_datasource_voc_base_dir = '/run/media/kele/DataSSD/Code/multi-task/rank-reid/datasource/voc-data'
	img_source_dir = os.path.join(voc2012_base_path, 'JPEGImages')

	train_num = 1500
	test_num = 2000
	objs = ['cat', 'cow', 'dog', 'horse', 'sheep']
	for i in range(len(objs)):
		train_txt_path = os.path.join(main_dir, objs[i] + '_train.txt')
		val_txt_path = os.path.join(main_dir, objs[i] + '_val.txt')
		train_list = []
		test_list = []
		with open(train_txt_path) as txt:
			line = txt.readline()
			count = 0
			while line and count < train_num:
				sp = line.strip().split(' ')
				print(sp)
				if len(sp) <= 2 or not sp[2] == '1':
					line = txt.readline()
					continue
				print(sp[0] + sp[1])
				line = sp[0]
				count = count + 1
				train_list.append(line)
				line = txt.readline()
			copy_file_by_list_rename(
				source_dir=img_source_dir,
				target_dir=os.path.join(rank_reid_datasource_voc_base_dir, 'train'),
				file_list=train_list,
				class_id=i)

		with open(val_txt_path) as txt:
			line = txt.readline()
			count = 0
			while line and count < test_num:
				sp = line.strip().split(' ')
				print(sp)
				if len(sp) <= 2 or not sp[2] == '1':
					line = txt.readline()
					continue
				print(sp[0] + sp[1])
				line = sp[0]
				count = count + 1
				test_list.append(line)
				line = txt.readline()

			copy_file_by_list_rename(
				source_dir=img_source_dir,
				target_dir=os.path.join(rank_reid_datasource_voc_base_dir, 'test'),
				file_list=test_list,
				class_id=i)


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
			line = line.strip()
			seg_train_set.add(line)
			line = file.readline()

	with open(seg_trainval_txt_path) as file:
		line = file.readline()
		while line:
			# line = line.split(' ')[0]
			line = line.strip()
			seg_trainval_set.add(line)
			line = file.readline()

	with open(seg_val_txt_path) as file:
		line = file.readline()
		while line:
			line = line.strip()
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
				print(line)
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
			train_file.write(idx + '\n')
		for idx in trainval_set:
			trainval_file.write(idx + '\n')
		for idx in val_set:
			val_file.write(idx + '\n')


def batch_img_resize_save(source_dir, target_dir, target_size):
	suffixes = ['jpg', 'png', 'jpeg', 'bmp']
	for file in os.listdir(source_dir):
		if str_util.confirm_file_type(file, *suffixes):
			img = misc.imread(os.path.join(source_dir, file))
			img = misc.imresize(img, target_size)
			misc.imsave(os.path.join(target_dir, file), img)

def batch_resize_test():
	source_dir = '/run/media/kele/DataSSD/Code/multi-task/rank-reid/datasource/voc-data/tttest'
	target_dir = source_dir
	target_size = [224, 224, 3]
	batch_img_resize_save(source_dir, target_dir, target_size)


def seg_img_by_xml(source_img_path, xml_path, seg_obj_name, save_to_dir, file_pre_name, suffix):
	img = misc.imread(source_img_path)
	bnds = get_bound_rect_from_xml(xml_path, seg_obj_name)
	for i in range(len(bnds)):
		bnd = bnds[i]
		seg_img = bounding_core.img_segmentation(img, bnd)
		misc.imsave(os.path.join(save_to_dir, file_pre_name + str(i) + suffix), seg_img)


def get_bound_rect_from_xml(xml_path, object_name):
	DOMTree = xml.dom.minidom.parse(xml_path)
	collection = DOMTree.documentElement
	objs = collection.getElementsByTagName('object')
	res = []
	for obj in objs:
		name = obj.getElementsByTagName('name')[0]
		if name is not None:
			if name.childNodes[0].nodeValue == object_name:
				print(object_name)
				bnd = obj.getElementsByTagName('bndbox')[0]

				x_min = bnd.getElementsByTagName('xmin')[0]
				x_min = int(x_min.childNodes[0].nodeValue)

				y_min = bnd.getElementsByTagName('ymin')[0]
				y_min = int(y_min.childNodes[0].nodeValue)

				x_max = bnd.getElementsByTagName('xmax')[0]
				x_max = int(x_max.childNodes[0].nodeValue)

				y_max = bnd.getElementsByTagName('ymax')[0]
				y_max = int(y_max.childNodes[0].nodeValue)

				# bound_rect: [[most_up, most_left], [most_down, most_right]]
				bnd_rect = [[y_min - 1, x_min - 1], [y_max - 1, x_max - 1]]
				res.append(bnd_rect)
	return res


def xml_seg_test():
	source_img_path = '/run/media/kele/DataSSD/Code/multi-task/rank-reid/datasource/voc-data/train/0_2008_000096.jpg'
	xml_path = '/run/media/kele/Data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/Annotations/2008_000096.xml'
	save_dir = '/run/media/kele/DataSSD/Code/multi-task/rank-reid/datasource/voc-data/temp'
	seg_img_by_xml(
		source_img_path=source_img_path, xml_path=xml_path,
		seg_obj_name='cat', save_to_dir=save_dir, file_pre_name='2008000096_seg_test', suffix='.jpg')


def xml_test():
	xml_path = '/run/media/kele/Data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/Annotations/2007_000027.xml'
	get_bound_rect_from_xml(xml_path, 'person')


def batch_seg_by_xml():
	source_dir = '/run/media/kele/DataSSD/Code/multi-task/rank-reid/datasource/voc-data/test'
	xml_dir = os.path.join(VOC2012_BASE_PATH, 'Annotations')
	save_to_dir = '/run/media/kele/DataSSD/Code/multi-task/rank-reid/datasource/voc-data/test_temp'
	objects = ['cat', 'cow', 'dog', 'horse', 'sheep']
	for file in os.listdir(source_dir):
		if str_util.confirm_file_type(file, *['jpg']):
			idx = file[2:-4]
			xml_path = os.path.join(xml_dir, idx + '.xml')
			seg_img_by_xml(
				source_img_path=os.path.join(source_dir, file),
				xml_path=xml_path,
				seg_obj_name=objects[int(file[0])],
				save_to_dir=save_to_dir,
				file_pre_name=file[0:-4],
				suffix=file[-4:])


if __name__ == '__main__':
	pass
	# batch_img_resize_save(source_dir='/run/media/kele/DataSSD/Code/multi-task/rank-reid/datasource/voc-data/test_temp',
	# 					  target_dir='/run/media/kele/DataSSD/Code/multi-task/rank-reid/datasource/voc-data/test_temp',
	# 					  target_size=[224, 224, 3])
	# batch_seg_by_xml()
	# xml_seg_test()
	# xml_test()
	# count_animal_seg_cls(voc2012_base=VOC2012_BASE_PATH)
	# batch_resize_test()