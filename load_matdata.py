import scipy.io as io
import numpy as np

def load_matdata(path, key):
    data=io.loadmat(path)
    data = data[key]
    print('载入数据的大小为：',np.shape(data))
    return data