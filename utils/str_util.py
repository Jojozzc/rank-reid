import re

def confirm_file_type(file_name=None, *suffixes):
	if file_name is None:
		return False
	if len(suffixes) == 0 or suffixes is None:
		return True
	file_name = str(file_name).lower()
	for sfx in suffixes:
		if re.search(sfx + '$', file_name):
			return True
	return False

def file_name_test():
	file_name = 'fad.jpg'
	suffixes = ['.jpg']
	print(confirm_file_type(file_name, *suffixes))

if __name__ == '__main__':
	file_name_test()