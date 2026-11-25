import argparse
import pickle
from utils import *


parser = argparse.ArgumentParser()
parser.add_argument("-o", "--modelfile", help="output model file", type=argparse.FileType('rb'), 
                    default='save_model/Beeradvocate_2024')
parser.add_argument('--data_path', type=str, help='path for loading raw data',
                    default='data/Beeradvocate/Beeradvocate.mat')
parser.add_argument('--mu_on_user', type=float,
                    help='ratio on user for score prediction, default=0.7 (-1 means from 0, 0.1, ..., 1)', default=0.7)
args = parser.parse_args()
train_tuple_list, test_tuple_list, user2items_test, user2index, item2index, exp2index = load_our_data(args.data_path)

# evaluating recommendation
rec = pickle.load(args.modelfile)
# print(get_now_time() + '@5')
# user2items_top = rec.get_prediction_item(5, list(user2items_test.keys()))
# ndcg, precision, recall, f1, ap = evaluate_item(user2items_test, user2items_top)
# print("NDCG: {:.5f}\tPrecision: {:.5f}\tRecall: {:.5f}\tF1: {:.5f}\tMAP: {:.5f}".
#       format(ndcg, precision, recall, f1, ap))
# item_r = [ndcg, precision, recall, f1, ap]

# test_tuple_predict = rec.get_prediction_exp(5, test_tuple_list, args.mu_on_user)
# ndcg, precision, recall, f1, ap = evaluate_exp(test_tuple_list, test_tuple_predict)
# print("NDCG: {:.5f}\tPrecision: {:.5f}\tRecall: {:.5f}\tF1: {:.5f}\tMAP: {:.5f}".
#       format(ndcg, precision, recall, f1, ap))
# exp_r = [ndcg, precision, recall, f1, ap]

# # evaluating recommendation
# print(get_now_time() + '@10')
# user2items_top = rec.get_prediction_item(10, list(user2items_test.keys()))
# ndcg, precision, recall, f1, ap = evaluate_item(user2items_test, user2items_top)
# print("NDCG: {:.5f}\tPrecision: {:.5f}\tRecall: {:.5f}\tF1: {:.5f}\tMAP: {:.5f}".
#       format(ndcg, precision, recall, f1, ap))
# item_r = [ndcg, precision, recall, f1, ap]

# test_tuple_predict = rec.get_prediction_exp(10, test_tuple_list, args.mu_on_user)
# ndcg, precision, recall, f1, ap = evaluate_exp(test_tuple_list, test_tuple_predict)
# print("NDCG: {:.5f}\tPrecision: {:.5f}\tRecall: {:.5f}\tF1: {:.5f}\tMAP: {:.5f}".
#       format(ndcg, precision, recall, f1, ap))
# exp_r = [ndcg, precision, recall, f1, ap]

# new item evaluating
top_k = 5
print(get_now_time() + '@5')

item_ndcg, item_precision, item_recall, item_f1, item_ap = rec.new_item_evaluate(top_k, test_tuple_list, user2items_test, args.mu_on_user)
print("NDCG: {:.5f}\tPrecision: {:.5f}\tRecall: {:.5f}\tF1: {:.5f}\tMAP: {:.5f}".
      format(item_ndcg, item_precision, item_recall, item_f1, item_ap))

print(get_now_time() + '@10')
top_k = 10

item_ndcg, item_precision, item_recall, item_f1, item_ap = rec.new_item_evaluate(top_k, test_tuple_list, user2items_test, args.mu_on_user)
print("NDCG: {:.5f}\tPrecision: {:.5f}\tRecall: {:.5f}\tF1: {:.5f}\tMAP: {:.5f}".
      format(item_ndcg, item_precision, item_recall, item_f1, item_ap))