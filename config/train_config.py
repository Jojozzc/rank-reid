batch_size = 4
epcho = 15
class_coutns = {'market':4}
def get_batch_size():
    return batch_size
def get_epcho():
    return epcho
def get_class_count(source):
    return class_coutns.get(source)