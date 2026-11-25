from bperj import BPERJ
from utils import *
import numpy as np
import argparse
import sys
import pickle
from visdom import Visdom


# fix random seed
def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='path for loading raw data',
                    default='data/OpenRice/OpenRice.mat')
parser.add_argument('--data_ratio', type=str, help='ratio of train:test, default=7:3', default='7:3')
parser.add_argument("-o", "--outfile", help="output model file", type=argparse.FileType('wb'), 
                    default='save_model/OpenRice_2024')
parser.add_argument('--top_k', type=int, help='select top k to evaluate, default=10', default=10)
parser.add_argument('--max_iter', type=int, help='max iteration number, default=500', default=150)
parser.add_argument('--learning_rate', type=float, help='learning rate, default=0.01', default=0.1)
parser.add_argument('--dimension', type=int, help='dimension of latent factors, default=20', default=30)
parser.add_argument('--alpha', type=float, help='reg for explanation, default=1.0', default=1.0)
parser.add_argument('--reg_rate', type=float, help='rate for regularization, default=0.01', default=0.01)
parser.add_argument('--mu_on_user', type=float,
                    help='ratio on user for score prediction, default=0.7 (-1 means from 0, 0.1, ..., 1)', default=0.5)
args = parser.parse_args()

print('-----------------------------ARGUMENTS-----------------------------')
for arg in vars(args):
    value = getattr(args, arg)
    if value is None:
        value = ''
    print('{:30} {}'.format(arg, value))
print('-----------------------------ARGUMENTS-----------------------------')

if args.data_path is None:
    sys.exit(get_now_time() + 'provide data_path for loading data')
print(get_now_time() + 'Program Starts')
train_tuple_list, test_tuple_list, user2items_test, user2index, item2index, exp2index = load_our_data(args.data_path)
same_seeds(2024)
print(len(test_tuple_list))
rec = BPERJ(train_tuple_list, len(user2index), len(item2index), len(exp2index), args.learning_rate, args.dimension,
            args.alpha, args.reg_rate)
wind = Visdom(port=8001)
pre_loss = 10e7
for it in range(1, args.max_iter + 1):
    # if it % 100 == 0:
    #     print(get_now_time() + 'iteration {}'.format(it))
    #     rec.learning_rate *= 0.1
    loss = rec.train_one_epoch()
    print(it, loss) 
    wind.line([loss], [it], win='train_loss', update='append')
pickle.dump(rec, args.outfile)
