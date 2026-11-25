import pickle
import argparse
import numpy as np
from utils import *


def read_data(filepath):
    return np.genfromtxt(filepath, delimiter=' ', dtype=int)


p = argparse.ArgumentParser()
# p.add_argument("-i", "--infile", help="input data file", type=argparse.FileType('r'), default='OpenRiceText.test')
p.add_argument("-m", "--modelfile", help="input model file", type=argparse.FileType('rb'), default='save_model/HotelRec_2023')
args = p.parse_args()

user2items_test = {}
test_tuple_list = []
with open('HotelRec.test', 'r') as f:
    lines = f.readlines()
    for line in lines:
        lis = line.strip().split()
        u = int(lis[0])
        i = int(lis[1])
        exps = [int(i) for i in lis[2:]]
        test_tuple_list.append([u, i, set(exps)])
        if u not in user2items_test:
            user2items_test[u] = [i]
        else:
            user2items_test[u].append(i)


model = pickle.load(args.modelfile)


top_k = 5
print("@5")

item_ndcg, item_precision, item_recall, item_f1, item_ap = model.new_item_evaluate(top_k, test_tuple_list, user2items_test)
print("NDCG: {:.5f}\tPrecision: {:.5f}\tRecall: {:.5f}\tF1: {:.5f}\tMAP: {:.5f}".
      format(item_ndcg, item_precision, item_recall, item_f1, item_ap))

print("@10")
top_k = 10

item_ndcg, item_precision, item_recall, item_f1, item_ap = model.new_item_evaluate(top_k, test_tuple_list, user2items_test)
print("NDCG: {:.5f}\tPrecision: {:.5f}\tRecall: {:.5f}\tF1: {:.5f}\tMAP: {:.5f}".
      format(item_ndcg, item_precision, item_recall, item_f1, item_ap))