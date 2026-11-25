from utils import sigmoid
import numpy as np
import random
import heapq
from utils import evaluate_item

# 便于后续计算
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


class BPERJ:
    def __init__(self, train_tuple_list, user_num, item_num, exp_num, learning_rate=0.01, dimension=20, alpha=1.0, reg_rate=0.01):
        self.train_tuple_list = train_tuple_list
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.reg_rate = reg_rate
        # 便于后续引用
        # self.user_num = user_num
        # self.item_num = item_num

        self.user_matrix = np.random.uniform(size=(user_num, dimension), high=0.005, low=-0.005).astype('f')
        self.item_matrix = np.random.uniform(size=(item_num, dimension), high=0.005, low=-0.005).astype('f')
        self.exp_matrix_U = np.random.uniform(size=(exp_num, dimension), high=0.005, low=-0.005).astype('f')
        self.exp_matrix_I = np.random.uniform(size=(exp_num, dimension), high=0.005, low=-0.005).astype('f')
        self.item_bias = np.zeros(item_num, dtype=np.float32)
        self.exp_bias_U = np.zeros(exp_num, dtype=np.float32)
        self.exp_bias_I = np.zeros(exp_num, dtype=np.float32)

        exp2user_set = {}
        exp2item_set = {}
        self.user2exp_set = {}
        self.item2exp_set = {}
        item2user_set = {}
        self.user2item_set = {}
        for x in self.train_tuple_list:
            u = x[0]
            i = x[1]
            exps = x[2]
            for e in exps:
                if e in exp2user_set:
                    exp2user_set[e].add(u)
                else:
                    exp2user_set[e] = {u}
                if e in exp2item_set:
                    exp2item_set[e].add(i)
                else:
                    exp2item_set[e] = {i}
            if u in self.user2exp_set:
                self.user2exp_set[u] |= exps
            else:
                self.user2exp_set[u] = exps.copy()
            if i in self.item2exp_set:
                self.item2exp_set[i] |= exps
            else:
                self.item2exp_set[i] = exps.copy()
            if i in item2user_set:
                item2user_set[i].add(u)
            else:
                item2user_set[i] = {u}
            if u in self.user2item_set:
                self.user2item_set[u].add(i)
            else:
                self.user2item_set[u] = {i}

        global_avg = len(self.train_tuple_list) / (user_num * exp_num)
        for e in range(exp_num):
            if e in exp2user_set:
                self.exp_bias_U[e] = len(exp2user_set[e]) / user_num - global_avg
        global_avg = len(self.train_tuple_list) / (item_num * exp_num)
        for e in range(exp_num):
            if e in exp2item_set:
                self.exp_bias_I[e] = len(exp2item_set[e]) / item_num - global_avg
        global_avg = len(self.train_tuple_list) / (user_num * item_num)
        for i in range(item_num):
            if i in item2user_set:
                self.item_bias[i] = len(item2user_set[i]) / user_num - global_avg

        self.exp_list = list(range(exp_num))
        self.item_list = list(range(item_num))

    def __calculate_gradients_update(self, u, i, exps):
        if len(exps) == 0:
            return 0, 0
        e = random.choice(list(exps))
        u_item = self.user2item_set[u]
        u_exp = self.user2exp_set[u]
        i_exp = self.item2exp_set[i]
        if len(u_exp) == len(self.exp_list) or len(i_exp) == len(self.exp_list):
            return 0, 0
        else:
            i_ = i
            while i_ in u_item:
                i_ = random.choice(self.item_list)
            e_ = e
            while e_ in u_exp:
                e_ = random.choice(self.exp_list)
            e__ = e
            while e__ in i_exp:
                e__ = random.choice(self.exp_list)

            i_minus_i_ = self.item_matrix[i] - self.item_matrix[i_]
            e_minus_e_ = self.exp_matrix_U[e] - self.exp_matrix_U[e_]
            e_minus_e__ = self.exp_matrix_I[e] - self.exp_matrix_I[e__]

            z = -sigmoid(self.user_matrix[u].dot(-i_minus_i_) + self.item_bias[i_] - self.item_bias[i])
            s = -self.alpha * sigmoid(self.user_matrix[u].dot(-e_minus_e_) + self.exp_bias_U[e_] - self.exp_bias_U[e])
            t = -self.alpha * sigmoid(self.item_matrix[i].dot(-e_minus_e__) + self.exp_bias_I[e__] - self.exp_bias_I[e])

            # calculate gradients
            der_u_vec = z * i_minus_i_ + s * e_minus_e_ + self.reg_rate * self.user_matrix[u]
            der_i_vec = z * self.user_matrix[u] + t * e_minus_e__ + self.reg_rate * self.item_matrix[i]
            der_i_vec_ = -z * self.user_matrix[u] + self.reg_rate * self.item_matrix[i_]
            der_e_vec_U = s * self.user_matrix[u] + self.reg_rate * self.exp_matrix_U[e]
            der_e_vec_I = t * self.item_matrix[i] + self.reg_rate * self.exp_matrix_I[e]
            der_e_vec_ = -s * self.user_matrix[u] + self.reg_rate * self.exp_matrix_U[e_]
            der_e_vec__ = -t * self.item_matrix[i] + self.reg_rate * self.exp_matrix_I[e__]
            der_i_bias = z + self.reg_rate * self.item_bias[i]
            der_i_bias_ = -z + self.reg_rate * self.item_bias[i_]
            der_e_bias_U = s + self.reg_rate * self.exp_bias_U[e]
            der_e_bias_I = t + self.reg_rate * self.exp_bias_I[e]
            der_e_bias_ = -s + self.reg_rate * self.exp_bias_U[e_]
            der_e_bias__ = -t + self.reg_rate * self.exp_bias_I[e__]

            # update
            self.user_matrix[u] -= self.learning_rate * der_u_vec
            self.item_matrix[i] -= self.learning_rate * der_i_vec
            self.item_matrix[i_] -= self.learning_rate * der_i_vec_
            self.exp_matrix_U[e] -= self.learning_rate * der_e_vec_U
            self.exp_matrix_I[e] -= self.learning_rate * der_e_vec_I
            self.exp_matrix_U[e_] -= self.learning_rate * der_e_vec_
            self.exp_matrix_I[e__] -= self.learning_rate * der_e_vec__
            self.item_bias[i] -= self.learning_rate * der_i_bias
            self.item_bias[i_] -= self.learning_rate * der_i_bias_
            self.exp_bias_U[e] -= self.learning_rate * der_e_bias_U
            self.exp_bias_I[e] -= self.learning_rate * der_e_bias_I
            self.exp_bias_U[e_] -= self.learning_rate * der_e_bias_
            self.exp_bias_I[e__] -= self.learning_rate * der_e_bias__
            # 后加的损失计算
            i_minus_i_ = self.item_matrix[i] - self.item_matrix[i_]
            e_minus_e_ = self.exp_matrix_U[e] - self.exp_matrix_U[e_]
            e_minus_e__ = self.exp_matrix_I[e] - self.exp_matrix_I[e__]

            z = sigmoid(self.user_matrix[u].dot(i_minus_i_) + self.item_bias[i] - self.item_bias[i_])
            s = sigmoid(self.user_matrix[u].dot(e_minus_e_) + self.exp_bias_U[e] - self.exp_bias_U[e_])
            t = sigmoid(self.item_matrix[i].dot(e_minus_e__) + self.exp_bias_I[e] - self.exp_bias_I[e__])
            item_loss = -np.log(z)
            exp_loss = - np.log(s) - np.log(t)
            return item_loss, exp_loss

    def train_one_epoch(self):
        index_list = list(range(len(self.train_tuple_list)))
        random.shuffle(index_list)
        item_loss, exp_loss = 0, 0
        for idx in index_list:
            x = self.train_tuple_list[idx]
            u = x[0]
            i = x[1]
            exps = x[2]
            loss1, loss2 = self.__calculate_gradients_update(u, i, exps)
            item_loss += loss1
            exp_loss += loss2
        # print(item_loss, exp_loss)
        loss = item_loss + self.alpha * exp_loss
        return loss

    def __predict_item(self, u):
        # if u not in 
        score = self.user_matrix[u].dot(self.item_matrix.T) + self.item_bias
        return score

    def __predict_exp(self, u, i, mu):
        score_u = self.user_matrix[u].dot(self.exp_matrix_U.T) + self.exp_bias_U
        score_i = self.item_matrix[i].dot(self.exp_matrix_I.T) + self.exp_bias_I
        return score_u * mu + score_i * (1 - mu)

    def get_prediction_item(self, top_k, users_test):
        user2items_top = {}
        for u in users_test:
            rating_vec = self.__predict_item(u)
            item2rating = {}
            for i, s in enumerate(rating_vec):
                item2rating[i] = s
            top_list = heapq.nlargest(top_k, item2rating, key=item2rating.get)
            user2items_top[u] = top_list

        return user2items_top

    def get_prediction_exp(self, top_k, test_tuple_list, mu):
        test_tuple_predict = []
        for x in test_tuple_list:
            u = x[0]
            i = x[1]
            score_vec = self.__predict_exp(u, i, mu)
            exp2score = {}
            for e, s in enumerate(score_vec):
                exp2score[e] = s
            top_list = heapq.nlargest(top_k, exp2score, key=exp2score.get)
            test_tuple_predict.append(top_list)

        return test_tuple_predict

    def new_item_evaluate(self, top_k, test_tuple_list, user2items_test, mu):
        user2items_top = {}
        item_num = len(self.item_list)
        user_num,_ = self.user_matrix.shape
        exp_NDCG = np.zeros((user_num, item_num), dtype=float)
        test_tuple_predict = []
        for x in test_tuple_list:
            u = x[0]
            i = x[1]
            exps = x[2]
            score_vec = self.__predict_exp(u, i, mu)
            top_k_with_indices = heapq.nlargest(top_k, enumerate(score_vec), key=lambda x: x[1])
            recommends, _ = zip(*[(index, value) for index, value in top_k_with_indices])
            recommends = list(recommends)
            test_tuple_predict.append(recommends)
            h, e_num, e_ndcg, e_ap = exp_one_evaluate(exps, recommends)
            exp_NDCG[u][i] = e_ndcg
        
        item_NDCG, item_pre, item_rec, item_F1, item_ap = [], [], [], [], []
        for user in user2items_test:
            score = self.__predict_item(user) # [n_item]
            score = score + exp_NDCG[user]
            item2rating = {}
            for i, s in enumerate(score):
                item2rating[i] = s
            top_list = heapq.nlargest(top_k, item2rating, key=item2rating.get)
            user2items_top[user] = top_list
        item_ndcg, item_precision, item_recall, item_f1, item_ap = evaluate_item(user2items_test, user2items_top)
        return item_ndcg, item_precision, item_recall, item_f1, item_ap
