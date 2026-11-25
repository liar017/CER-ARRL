import heapq
import math

import numpy as np
import torch
import torch.nn.functional as F
from item_to_genre import item_to_genre
import pandas as pd
from sklearn.preprocessing import normalize
from topK import topK, hrK


class XEval(object):
    """
    evaluate the explainability
    """

    def __init__(self, dataset='100k'):
        """
        size: can be either '100k' or '1m', or 'ciao'
        """
        # if dataset == '100k' or dataset == '1m':
        #     path = "movielens/" + 'ml-' + dataset + '.ratings'
        #     path_i = "movielens/" + 'ml-' + dataset + '.iave'
        #     path_u = "movielens/" + 'ml-' + dataset + '.uave'
        # elif dataset == 'ymovie':
        #     path = 'ymovies/clean_data.txt'
        #     path_i = "ymovies/ymovie.iave"
        #     path_u = "ymovies/ymovie.uave"
        # else:
        #     raise Exception('not supported dataset!')

        self.dataset = dataset  # the dataset name
        path = 'data/' + dataset + '.ratings'
        path_u = 'data/' + dataset + '.uave'
        path_i = 'data/' + dataset + '.iave'
        # load rating data
        self.data_df = pd.read_csv(path, header=None)  # dataframe
        self.data = self.data_df.values[:, 0:3]  # numpy, note here still float type
        # load averages
        self.i_ave_df = pd.read_csv(path_i, index_col=0)  # item id as index
        self.u_ave_df = pd.read_csv(path_u, index_col=0)
        self.ave = np.mean(self.data[:, 2])  # the global average
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def get_general_pref(self, uid):
        """
        given uids, output the preference vectors
        uid is a tensor [batch_size, 1]
        """
        # u_rated_item = self.data[np.isin(self.data[:, 0], uid)]
        # u_rated_asp = item_to_genre(u_rated_item[:, 1], data_size=self.dataset)
        # u_concat = np.concatenate((u_rated_item, u_rated_asp), axis=1)
        # u_pref = np.sum(u_concat, axis=0, where=(u_concat[:,0]==0))
        asp_list = []
        uid = uid.cpu().numpy()
        for u in uid:
            u_rated_item = self.data[self.data[:, 0] == u]
            # aspects for movies rated by a user
            u_rated_asp = item_to_genre(u_rated_item[:, 1], data_size=self.dataset)
            u_rated_asp = np.nan_to_num(u_rated_asp)  # important, avoid nan
            u_rated_asp = u_rated_asp.astype(float)  # for later calculation
            # ratings given by a user
            u_rating = u_rated_item[:, 2].astype(float)

            ave_rating = u_rating.mean()
            weights = (u_rating - ave_rating) / 5.0  # minus global average, 5 scale
            # weights = weights / weights.sum()
            # weighted sum over movies
            u_pref = np.dot(u_rated_asp.T, weights).T
            # u_pref_norm = np.linalg.norm(u_pref, ord=2)
            # u_pref = u_pref / u_pref_norm
            u_pref = normalize(u_pref[:, np.newaxis], axis=0).ravel()
            u_pref = u_pref.reshape([1, -1])
            asp_list.append(u_pref)
        pref = np.concatenate(asp_list, axis=0)

        return pref

    def get_u_ave(self):
        user = self.data[:, 0].astype(int)
        user = np.unique(user)
        u_ave_list = []
        for u in user:
            u_rated_item = self.data[self.data[:, 0] == u]
            u_rating = u_rated_item[:, 2].astype(float)
            u_ave_rating = u_rating.mean()
            u_ave_list.append(u_ave_rating)
        u_ave = np.array(u_ave_list)
        # name = ['uid', 'ave']
        data = {'uid': user, 'ave': u_ave}
        df = pd.DataFrame(data)
        path = 'data/' + self.dataset + '.uave'
        df.to_csv(path, index=False, float_format='%.3f')

    def get_i_ave(self):
        item = self.data[:, 1].astype(int)
        item = np.unique(item)
        i_ave_list = []
        for i in item:
            i_rated_user = self.data[self.data[:, 1] == i]
            i_rating = i_rated_user[:, 2].astype(float)
            i_ave_rating = i_rating.mean()
            i_ave_list.append(i_ave_rating)
        i_ave = np.array(i_ave_list)
        # name = ['uid', 'ave']
        data = {'mid': item, 'ave': i_ave}
        df = pd.DataFrame(data)
        path = 'data/' + self.dataset + '.iave'
        df.to_csv(path, index=False, float_format='%.3f')

    def get_all_ave(self):
        # this function is for average calculation purpose
        ratings = self.data[:, 2].astype(float)
        ave = ratings.sum()
        print(ave)
        # '100k'： 3.530

    def get_u_pref(self, uid):
        ave = self.ave  # global average
        df = self.data_df
        uid = uid.cpu().numpy()

        # 筛选
        u_rated = df.loc[df['uid'].isin(uid)]
        item_rated = u_rated['mid']
        item_ave = self.i_ave_df.loc[item_rated.values].values
        item_bias = item_ave - self.ave
        user_ave = self.u_ave_df.loc[u_rated['uid'].values].values
        user_bias = user_ave - self.ave
        weight = u_rated[['rating']].values - (self.ave + item_bias + user_bias)
        weight = weight.flatten()
        # u_rated['weight'] = weight
        u_rated_asp = item_to_genre(item_rated, data_size=self.dataset).values
        # calculate the weighted rating
        u_pref = np.multiply(u_rated_asp.T, weight).T / 5.0
        u_pref_list = u_pref.tolist()
        u_rated['asp'] = u_pref_list
        u_rated['asp'] = u_rated['asp'].apply(lambda x: np.array(x))  # convert to array
        # u_rated['asp'] = u_rated['asp'].multiply(weight)
        u_rated = u_rated[['uid', 'asp']]
        pref = u_rated.groupby(['uid']).sum()
        # if self.dataset == '100k':
        #     path = 'movielens/ml-100k.upref'
        # pref.to_csv(path, index=False)
        pref_list = pref['asp'].tolist()
        pref_ary = np.array(pref_list)
        return pref_ary

    def get_cos_sim(self, uid, predicted):
        """
        predicted: a torch tensor [batch, num_asp]
        uid: a torch tensor [batch]
        """
        pref = self.get_u_pref(uid)
        # convert to tensor.cuda
        pref = torch.tensor(pref, dtype=torch.float).to(self.device)
        pref = F.normalize(pref, p=1, dim=-1)
        sim = F.cosine_similarity(pref, predicted, dim=-1)
        return sim

    def get_specific_cos_sim(self, uid, asp, predicted):
        """
        predicted: a torch tensor [batch, num_asp]
        uid: a torch tensor [batch]
        """
        pref = self.get_u_pref(uid)
        # convert to tensor.cuda
        pref = torch.tensor(pref, dtype=torch.float).to(self.device)
        pref = F.normalize(pref, p=1, dim=-1)
        spec_pref = pref * asp
        spec_pred = predicted * asp
        sim = F.cosine_similarity(spec_pred, spec_pref, dim=-1)
        return sim

    def get_rank(self, pref):  # pref is a (batch, num_asp) tensor
        pref = np.abs(pref)
        rank = np.argsort(pref, axis=-1)
        return rank

    def get_top_K_pos_(self, pref, pred, K=3):  # top K aspects thats are positive
        """
        pref: given preference
        pred: predicted preference
        """
        pref_rank = self.get_rank(pref)
        pred_rank = self.get_rank(pred)
        pref_top_K = (pref_rank < K).astype(float)
        pred_top_K = (pred_rank < K).astype(float)
        acc_top_K = np.multiply(pref_top_K, pred_top_K).sum(axis=-1) / float(K)
        return acc_top_K

    def get_top_K_pos(self, uid, pred, K=5, M=3):
        pref = self.get_u_pref(uid)
        pred = pred.cpu().data.numpy()
        # return self.get_top_K_pos_(pref, pred, K)
        if self.dataset == '100k':
            num_user = 643
        elif self.dataset == '1m':
            num_user = 6040
        elif self.dataset == 'ymovie':
            num_user = 7642
        return topK(pref, pred, K, M, num_user)

    def get_hr_K(self, uid, pred, K=3):
        pref = self.get_u_pref(uid)
        pred = pred.cpu().data.numpy()
        return hrK(pref, pred, K)

    def evaluate_exp(self, pred, real, top_k):
        pred_k = []
        for i in pred:
            exp2score = {}
            for e, s in enumerate(i):
                exp2score[e] = s
            top_list = heapq.nlargest(top_k, exp2score, key=exp2score.get)
            pred_k.append(top_list)
        # f = open('example/sample.txt', 'a')
        # for i in pred_k:
        #     f.write(' '.join([str(j) for j in i]) + '\n')
        dcgs = [1 / math.log(i + 2) for i in range(top_k)]
        aps = [1 / i for i in range(1, top_k + 1)]
        ndcg_sum = 0
        precision_sum = 0
        recall_sum = 0  # it is also named hit ratio
        f1_sum = 0
        ap_sum = 0
        for i in range(len(real)):
            hits = 0
            exps = 0
            ap = 0
            ndcg = 0
            for j in range(len(real[0])):
                if real[i][j] == 1:
                    exps += 1
            for k in range(len(pred_k[i])):
                e = pred_k[i][k]
                if real[i][e] == 1:
                    ndcg += dcgs[k]
                    hits += 1
                    ap += hits * aps[k]
            m = min(top_k, exps)
            idcg = 0
            for i in range(0, m):
                idcg += dcgs[i]
            pre = hits / top_k
            if exps == 0:
                rec = 0
                ap_sum += 0
                ndcg_sum += 0
            else:
                rec = hits / exps
                ap_sum += ap / m
                ndcg_sum += ndcg / idcg
            precision_sum += pre
            recall_sum += rec
            if (pre + rec) > 0:
                f1_sum += 2 * pre * rec / (pre + rec)
        return ndcg_sum, precision_sum, recall_sum, f1_sum, ap_sum
    

    def new_evaluate_exp(self, pred, real, top_k):
        pred_k = []
        for i in pred:
            exp2score = {}
            for e, s in enumerate(i):
                exp2score[e] = s
            top_list = heapq.nlargest(top_k, exp2score, key=exp2score.get)
            pred_k.append(top_list)
        # f = open('example/sample.txt', 'a')
        # for i in pred_k:
        #     f.write(' '.join([str(j) for j in i]) + '\n')
        dcgs = [1 / math.log(i + 2) for i in range(top_k)]
        aps = [1 / i for i in range(1, top_k + 1)]
        ndcg_sum_list = []
        precision_sum_list = []
        recall_sum_list = [] # it is also named hit ratio
        f1_sum_list = []
        ap_sum_list = []
        for i in range(len(real)):
            hits = 0
            exps = 0
            ap = 0
            ndcg = 0
            for j in range(len(real[0])):
                if real[i][j] == 1:
                    exps += 1
            for k in range(len(pred_k[i])):
                e = pred_k[i][k]
                if real[i][e] == 1:
                    ndcg += dcgs[k]
                    hits += 1
                    ap += hits * aps[k]
            m = min(top_k, exps)
            idcg = 0
            for i in range(0, m):
                idcg += dcgs[i]
            pre = hits / top_k
            if exps == 0:
                rec = 0
                ap_sum_list.append(0) 
                ndcg_sum_list.append(0) 
            else:
                rec = hits / exps
                ap_sum_list.append(ap / m)
                ndcg_sum_list.append(ndcg / idcg) 
            precision_sum_list.append(pre)
            recall_sum_list.append(rec) 
            if (pre + rec) > 0:
                f1_sum_list.append(2 * pre * rec / (pre + rec))

        return ndcg_sum_list, precision_sum_list, recall_sum_list, f1_sum_list, ap_sum_list

    def item_one_evaluate(self, target, result):
        k = len(result)
        dcgs = [1 / np.log(i + 2) for i in range(k)]
        ndcg = 0
        aps = [1 / i for i in range(1, k + 1)]
        ap = 0
        hits = 0
        for idx, item in enumerate(result):
            if item in target:
                hits += 1
                ndcg += dcgs[idx]
                ap += hits * aps[idx]
        m = min(k, len(target))
        idcg = 0
        for i in range(0, m):
            idcg += dcgs[i]
        return hits, ndcg / idcg, ap / m

    def item_evaluate(self, model, user2item_test, item_num, top_k):
        item_NDCG, item_pre, item_rec, item_F1, item_ap = [], [], [], [], []
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for user in user2item_test:
            users = torch.tensor([user for _ in range(item_num)])
            items = torch.tensor([i for i in range(item_num)])
            users = users.to(device)
            items = items.to(device)
            score = model.predict_spec_pref(users, items)
            i_, recommends = torch.topk(score, top_k)
            recommends = recommends.cpu()
            h, i_ndcg, i_ap = self.item_one_evaluate(user2item_test[user], recommends)
            pre = h / top_k
            item_pre.append(pre)
            rec = h / len(user2item_test[user])
            item_rec.append(rec)
            item_NDCG.append(i_ndcg)
            item_ap.append(i_ap)
            if (h / top_k + h) > 0:
                item_F1.append(2 * pre * rec / (pre + rec))
            else:
                item_F1.append(0)
        return np.mean(item_NDCG), np.mean(item_pre), np.mean(item_rec), np.mean(item_F1), np.mean(item_ap)
    
    def new_item_evaluate(self, model, user2item_test,user_num,item_num, top_k,testloader,evaluator):
        item_NDCG, item_pre, item_rec, item_F1, item_ap = [], [], [], [], []
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        exp_NDCG = torch.zeros(user_num, item_num).to(device)  # NDCG矩阵，保存（u, i）对解释的NDCG结果
        model.cpu()
        for i, data in enumerate(testloader, 0):
            inputs, labels, item_asp = data
            #inputs torch.Size([256, 2])
            #labels torch.Size([256])
            #item_asp torch.Size([256, 150])
            # get genre information from item id
            # item_asp = item_to_genre(inputs[:, 1], data_size=args.dataset).values
            # item_asp = torch.Tensor(item_asp).to(device)
            # inputs = inputs.to(device)
            # item_asp = item_asp.to(device)
            user_inputs = inputs[:, 0]
            item_inputs = inputs[:, 1]
            # item_inputs = inputs[:, 1]
            u_pred = model.predict_pref(user_inputs)
            #u_pred torch.Size([256, 150])
            ndcg, precision, recall, f1, ap = evaluator.new_evaluate_exp(u_pred, item_asp, top_k)
            for i in range(inputs.shape[0]):
                user = user_inputs[i]
                item = item_inputs[i]
                e_ndcg = ndcg[i]
                exp_NDCG[user,item] = e_ndcg
        print(">>>>>>>>>>>>>>>>")
        model.cuda(device)
        for user in user2item_test:
            users = torch.tensor([user for _ in range(item_num)])
            items = torch.tensor([i for i in range(item_num)])
            users = users.to(device)
            items = items.to(device)
            score = model.predict_spec_pref(users, items)
            score = score + exp_NDCG[user]  # 打分相加
            i_, recommends = torch.topk(score, top_k)
            recommends = recommends.cpu()
            h, i_ndcg, i_ap = self.item_one_evaluate(user2item_test[user], recommends)
            pre = h / top_k
            item_pre.append(pre)
            rec = h / len(user2item_test[user])
            item_rec.append(rec)
            item_NDCG.append(i_ndcg)
            item_ap.append(i_ap)
            if (h / top_k + h) > 0:
                item_F1.append(2 * pre * rec / (pre + rec))
            else:
                item_F1.append(0)
        return np.mean(item_NDCG), np.mean(item_pre), np.mean(item_rec), np.mean(item_F1), np.mean(item_ap)


if __name__ == '__main__':
    xeval = XEval(dataset='OpenRice')
    print(xeval.ave)
    xeval.get_u_ave()
    xeval.get_i_ave()
    # xeval.get_general_pref(torch.tensor([0, 1]))
    # c = xeval.get_top_K_pos(a, b)
