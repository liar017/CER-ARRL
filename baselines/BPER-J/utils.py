import datetime
import random
import pickle
import math
import os
import re
import scipy.io as scio
import torch
import numpy as np


def get_now_time():
    """a string of current time"""
    return '[' + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f') + ']: '


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def split_data(data_path, save_dir, ratio_str):
    '''
    :param data_path: pickle file, a list of all instances
    :param save_dir: save the indexes
    :param ratio_str: in the format of train:test
    '''

    user2idx = {}
    item2idx = {}
    exp2idx = {}
    reviews = pickle.load(open(data_path, 'rb'))
    for idx, review in enumerate(reviews):
        u = review['user']
        i = review['item']
        exp_list = review['exp_idx']

        if u in user2idx:
            user2idx[u].append(idx)
        else:
            user2idx[u] = [idx]
        if i in item2idx:
            item2idx[i].append(idx)
        else:
            item2idx[i] = [idx]
        for e in exp_list:
            if e in exp2idx:
                exp2idx[e].append(idx)
            else:
                exp2idx[e] = [idx]

    # split data
    train_set = set()
    for (u, idxes) in user2idx.items():
        idx = random.choice(idxes)
        train_set.add(idx)
    for (i, idxes) in item2idx.items():
        idx = random.choice(idxes)
        train_set.add(idx)
    for (e, idxes) in exp2idx.items():
        idx = random.choice(idxes)
        train_set.add(idx)

    total_num = len(reviews)
    ratio = [float(r) for r in ratio_str.split(':')]
    train_num = int(ratio[0] / sum(ratio) * total_num)

    index_list = list(range(total_num))
    while len(train_set) < train_num:
        train_set.add(random.choice(index_list))
    test_set = set(index_list) - train_set

    def write_to_file(path, data_set):
        idx_list = [str(x) for x in data_set]
        with open(path, 'w', encoding='utf-8') as f:
            f.write(' '.join(idx_list))

    # save data
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    print(get_now_time() + 'writing index data to {}'.format(save_dir))
    write_to_file(save_dir + 'train.index', train_set)
    write_to_file(save_dir + 'test.index', test_set)

item_num = 0
def load_data(data_path, index_dir):
    # collect all users id and items id
    user_set = set()
    item_set = set()
    exp_set = set()

    reviews = pickle.load(open(data_path, 'rb'))
    for review in reviews:
        user_set.add(review['user'])
        item_set.add(review['item'])
        exp_set |= set(review['exp_idx'])

    # convert id to array index
    user_list = list(user_set)
    item_list = list(item_set)
    # 计算item_num
    item_num = len(item_list)
    exp_list = list(exp_set)
    user2index = {x: i for i, x in enumerate(user_list)}
    item2index = {x: i for i, x in enumerate(item_list)}
    exp2index = {x: i for i, x in enumerate(exp_list)}

    def format_data(data_type):
        with open(index_dir + data_type + '.index', 'r') as f:
            line = f.readline()
            indexes = [int(x) for x in line.split(' ')]

        tuple_list = []
        for idx in indexes:
            rev = reviews[idx]
            u = user2index[rev['user']]
            i = item2index[rev['item']]
            exp_list = rev['exp_idx']
            exps = set([exp2index[e] for e in exp_list])

            tuple_list.append([u, i, exps])
        return tuple_list

    train_tuple_list = format_data('train')
    test_tuple_list = format_data('test')
    user2items_test = {}
    for x in test_tuple_list:
        u = x[0]
        i = x[1]
        if u in user2items_test:
            user2items_test[u].add(i)
        else:
            user2items_test[u] = {i}

    return train_tuple_list, test_tuple_list, user2items_test, user2index, item2index, exp2index


def evaluate_exp(test_tuple_list, test_tuple_predict):
    top_k = len(test_tuple_predict[0])
    dcgs = [1 / math.log(i + 2) for i in range(top_k)]
    aps = [1 / i for i in range(1, top_k + 1)]
    ndcg_sum = 0
    precision_sum = 0
    recall_sum = 0  # it is also named hit ratio
    f1_sum = 0
    ap_sum = 0
    for x, rank_list in zip(test_tuple_list, test_tuple_predict):
        exps = x[2]
        hits = 0
        ap = 0
        ndcg = 0
        for idx, e in enumerate(rank_list):
            if e in exps:
                ndcg += dcgs[idx]
                hits += 1
                ap += hits * aps[idx]
        # NEW
        # m = min(top_k, len(exps))
        # old
        m = top_k
        idcg = 0
        for i in range(0, m):
            idcg += dcgs[i]
        pre = hits / top_k
        if len(exps) == 0:
            rec = 0
            ap_sum += 0
            ndcg_sum += 0
        else:
            rec = hits / len(exps)
            ap_sum += ap / m
            ndcg_sum += ndcg / idcg
        precision_sum += pre
        recall_sum += rec
        if (pre + rec) > 0:
            f1_sum += 2 * pre * rec / (pre + rec)

    ndcg = ndcg_sum / len(test_tuple_list)
    precision = precision_sum / len(test_tuple_list)
    recall = recall_sum / len(test_tuple_list)
    f1 = f1_sum / len(test_tuple_list)
    ap = ap_sum / len(test_tuple_list)
    return ndcg, precision, recall, f1, ap


def evaluate_item(user2items_test, user2items_top):
    top_k = len(list(user2items_top.values())[0])
    dcgs = [1 / math.log(i + 2) for i in range(top_k)]
    aps = [1 / i for i in range(1, top_k + 1)]
    ndcg_sum = 0
    precision_sum = 0
    recall_sum = 0  # it is also named hit ratio
    f1_sum = 0
    ap_sum = 0
    for u, test_items in user2items_test.items():
        rank_list = user2items_top[u]
        hits = 0
        ap = 0
        ndcg = 0
        for idx, item in enumerate(rank_list):
            if item in test_items:
                ndcg += dcgs[idx]
                hits += 1
                ap += hits * aps[idx]
        # NEW
        # m = min(top_k, len(test_items))
        # old
        m = top_k
        idcg = 0
        for i in range(0, m):
            idcg += dcgs[i]
        pre = hits / top_k
        rec = hits / len(test_items)
        precision_sum += pre
        recall_sum += rec
        if (pre + rec) > 0:
            f1_sum += 2 * pre * rec / (pre + rec)
        ap_sum += ap / m
        ndcg_sum += ndcg / idcg

    ndcg = ndcg_sum / len(user2items_test)
    precision = precision_sum / len(user2items_test)
    recall = recall_sum / len(user2items_test)
    f1 = f1_sum / len(user2items_test)
    ap = ap_sum / len(user2items_test)

    return ndcg, precision, recall, f1, ap


def exp_one_evaluate(target, result):
    hits = 0
    exp_num = len(target)
    k = len(result)
    dcgs = [1 / np.log(i + 2) for i in range(k)]
    ndcg = 0
    aps = [1 / i for i in range(1, k + 1)]
    ap = 0
    for idx, item in enumerate(result):
        # print(item)
        if item in target:
            hits += 1
            ndcg += dcgs[idx]
            ap += hits * aps[idx]
    return hits, exp_num, ndcg / sum(dcgs), ap


def load_our_data(file):
    data = scio.loadmat(file)
    data_data = {'E': data.get('E'), 'D': data.get('D'), 'is': data.get('is')[0], 'iu': data.get('iu')[0]}
    train_tuple_list = []
    test_tuple_list = []
    num = data_data['D'].shape[0]
    for i in range(0, num):
        if i % 5 == 4:
            if data_data['D'][i][0] >= 0:
                exps = [i.start() for i in re.finditer('1', ''.join('%s' % j for j in data_data['E'][i]))]
                test_tuple_list.append([data_data['iu'][i] - 1, data_data['is'][i] - 1, set(exps)])
        else:
            if data_data['D'][i][0] >= 0:
                exps = [i.start() for i in re.finditer('1', ''.join('%s' % j for j in data_data['E'][i]))]
                train_tuple_list.append([data_data['iu'][i] - 1, data_data['is'][i] - 1, set(exps)])
    user2items_test = {}
    for x in test_tuple_list:
        u = x[0]
        i = x[1]
        if u in user2items_test:
            user2items_test[u].add(i)
        else:
            user2items_test[u] = {i}
    m = max(data_data['is'])
    n = max(data_data['iu'])
    k = data_data['E'].shape[1]
    user2index = {x: i for i, x in enumerate([j for j in range(0, n)])}
    item2index = {x: i for i, x in enumerate([j for j in range(0, m)])}
    exp2index = {x: i for i, x in enumerate([j for j in range(0, k)])}
    # train_tuple_list: [[user, item, {exps}], [user, item, {exps}]...]
    # user2items_test: [[user, {items}], [user, {items}] ...]
    # user2index: [user_id: user_id_num, ...]
    return train_tuple_list, test_tuple_list, user2items_test, user2index, item2index, exp2index
