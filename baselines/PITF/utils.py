import math
import re
import scipy.io as scio


def load_data(file):
    data = scio.loadmat(file)
    data_data = {'E': data.get('E'), 'D': data.get('D'), 'is': data.get('is')[0], 'iu': data.get('iu')[0]}
    train_tuple_list = []
    test_tuple_list = []
    num = data_data['D'].shape[0]
    u_min = 1000
    for i in range(0, num):
        u_min = min(u_min, data_data['iu'][i] - 1)
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
    return [n, m, k], train_tuple_list, test_tuple_list, user2items_test, user2index, item2index, exp2index


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
