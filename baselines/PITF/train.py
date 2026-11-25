import pickle
import random
import numpy as np
import argparse
from pitf import PITF
from utils import *


def read_data(filepath):
    return np.genfromtxt(filepath, delimiter=' ', dtype=int)


# fix random seed
def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数


p = argparse.ArgumentParser()
# p.add_argument("-i", "--infile", help="input tensor file", type=argparse.FileType('r'), default='sample.train')
p.add_argument("-o", "--outfile", help="output model file", type=argparse.FileType('wb'), default='save_model/OpenRice_2017')
p.add_argument("-a", "--alpha", help="learning rate (default=0.05)", type=float, nargs='?', default=0.05)
p.add_argument("-l", "--lamb", help="lambda (default=0.00005)", type=float, nargs='?', default=0.00005)
p.add_argument("-k", help="dimension (default=10)", type=int, nargs='?', default=30)
p.add_argument("-m", "--max_iter", help="max_iter (default=100)", type=int, nargs='?', default=150)
p.add_argument("-v", "--verbose", help="verbosity", action='store_true')
args = p.parse_args()

dataset = 'HotelRec'
seed = 2017
data_shape, train_tuple_list, test_tuple_list, user2items_test, user2index, item2index, exp2index = \
    load_data('data/' + dataset + '/' + dataset + '.mat')

# with open('OpenRice.test', 'w') as f:
#     for i in test_tuple_list:
#         f.write(str(i[0]) + ' ' + str(i[1]) + ' ' + ' '.join(str(i) for i in list(i[2])) + '\n')
same_seeds(seed)
model = PITF(args.alpha, args.lamb, args.k, args.max_iter, data_shape, args.verbose)

model.fit(train_tuple_list)
pickle.dump(model, args.outfile)
