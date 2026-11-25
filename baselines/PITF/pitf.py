import heapq
import random
from visdom import Visdom
import numpy as np
from utils import evaluate_item



def get_user2item(train_tuple_list):
    user2item_set = {}
    for x in train_tuple_list:
        u = x[0]
        i = x[1]
        if u in user2item_set:
            user2item_set[u].add(i)
        else:
            user2item_set[u] = {i}
    return user2item_set

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


class PITF:
    def __init__(self, alpha=0.0001, lamb=0.1, k=30, max_iter=100, data_shape=None, verbose=0):
        self.alpha = alpha
        self.lamb = lamb
        self.k = k
        self.max_iter = max_iter
        self.data_shape = data_shape
        self.verbose = verbose

    def _init_latent_vectors(self, data_shape):
        latent_vector = {'u': np.random.normal(loc=0, scale=0.1, size=(data_shape[0], self.k)),
                         'i': np.random.normal(loc=0, scale=0.1, size=(data_shape[1], self.k)),
                         'tu': np.random.normal(loc=0, scale=0.1, size=(data_shape[2], self.k)),
                         'ti': np.random.normal(loc=0, scale=0.1, size=(data_shape[2], self.k))}
        return latent_vector

    def _calc_number_of_dimensions(self, data, validation):
        u_max = -1
        i_max = -1
        t_max = -1
        for u, i, t in data:
            if u > u_max: u_max = u
            if i > i_max: i_max = i
            if t > t_max: t_max = t
        if not validation is None:
            for u, i, t in validation:
                if u > u_max:
                    u_max = u
                if i > i_max:
                    i_max = i
                if t > t_max:
                    t_max = t
        return u_max + 1, i_max + 1, t_max + 1

    def _draw_negative_sample(self, user2item, u, i, t):
        u_item = user2item[u]
        i_ = i
        while i_ in u_item:
            i_ = np.random.randint(self.data_shape[1])
        e = random.choice(list(t))
        e_ = np.random.randint(self.data_shape[2])
        while e_ in t:
            e_ = np.random.randint(self.data_shape[2])
        return i_, e, e_

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def _score(self, data):
        if data is None: return "No validation data"
        correct = 0.
        for u, i, answer_t in data:
            predicted = self.predict(u, i)
            if predicted == answer_t:
                correct += 1
        return correct / data.shape[0]

    def fit(self, data, validation=None):
        wind = Visdom(port=8889)
        # coff between item and exp
        exp_item_coff = 0.1
        if self.data_shape is None:
            self.data_shape = self._calc_number_of_dimensions(data, validation)

        self.latent_vector_ = self._init_latent_vectors(self.data_shape)
        user2item_set = get_user2item(data)
        for remained_iter in range(self.max_iter):
            print(remained_iter)
            item_loss = 0
            exp_loss = 0
            np.random.shuffle(data)
            for u, i, t in data:
                if len(t) != 0:
                    ni, pt, nt = self._draw_negative_sample(user2item_set, u, i, t)
                    y_diff = self.y(u, i, pt) - self.y(u, i, nt)
                    y_diff_item = self.y1(u, i) - self.y1(u, ni)
                    item_loss += np.log(1e-10 + self._sigmoid(y_diff_item))
                    exp_loss += np.log(1e-10 + self._sigmoid(y_diff))
                    delta = exp_item_coff * (1 - self._sigmoid(y_diff))
                    delta_item = (1 - self._sigmoid(y_diff_item))
                    self.latent_vector_['u'][u] += self.alpha * (
                            delta * (self.latent_vector_['tu'][pt] - self.latent_vector_['tu'][nt]) + delta_item *
                            (self.latent_vector_['i'][i] - self.latent_vector_['i'][ni]) - self.lamb *
                            self.latent_vector_['u'][u])
                    self.latent_vector_['i'][i] += self.alpha * (
                            delta * (self.latent_vector_['ti'][pt] - self.latent_vector_['ti'][nt]) + delta_item *
                            self.latent_vector_['u'][u] - self.lamb * self.latent_vector_['i'][i])
                    self.latent_vector_['i'][ni] += self.alpha * (
                            delta_item * -self.latent_vector_['u'][u] - self.lamb * self.latent_vector_['i'][ni])
                    self.latent_vector_['tu'][pt] += self.alpha * (
                            delta * self.latent_vector_['u'][u] - self.lamb * self.latent_vector_['tu'][pt])
                    self.latent_vector_['tu'][nt] += self.alpha * (
                            delta * -self.latent_vector_['u'][u] - self.lamb * self.latent_vector_['tu'][nt])
                    self.latent_vector_['ti'][pt] += self.alpha * (
                            delta * self.latent_vector_['i'][i] - self.lamb * self.latent_vector_['ti'][pt])
                    self.latent_vector_['ti'][nt] += self.alpha * (
                            delta * -self.latent_vector_['i'][i] - self.lamb * self.latent_vector_['ti'][nt])
            print(exp_loss, item_loss)
            loss = item_loss + exp_item_coff * exp_loss
            if self.verbose == 1:
                print("%s\t%s" % (self.max_iter - remained_iter, self._score(validation)))
            if remained_iter == 60:
                self.alpha *= 0.1
            wind.line([loss], [remained_iter], win='train_loss', update='append')
        return self

    def y(self, i, j, k):
        return self.latent_vector_['tu'][k].dot(self.latent_vector_['u'][i]) + self.latent_vector_['ti'][k].dot(
            self.latent_vector_['i'][j])

    def y1(self, i, j):
        return self.latent_vector_['u'][i].dot(self.latent_vector_['i'][j])

    def predict(self, i, j):
        y = self.latent_vector_['tu'].dot(self.latent_vector_['u'][i]) + self.latent_vector_['ti'].dot(
            self.latent_vector_['i'][j])
        return y.argmax()

    def predict2(self, x):
        y = self.latent_vector_['u'][x[:, 0]].dot(self.latent_vector_['tu'].T) + self.latent_vector_['i'][x[:, 1]].dot(
            self.latent_vector_['ti'].T)
        return y.argmax(axis=1)

    def __predict_item(self, u):
        score = self.latent_vector_['u'][u].dot(self.latent_vector_['i'].T)
        return score

    def __predict_exp(self, u, i):
        score = self.latent_vector_['u'][u].dot(self.latent_vector_['tu'].T) + \
                self.latent_vector_['i'][i].dot(self.latent_vector_['ti'].T)
        return score

    def get_prediction_item(self, top_k, users_test):
        user2items_top = {}
        for u in users_test:
            rating_vec = self.__predict_item(u)
            # items = self.user2item_set[u]
            item2rating = {}
            for i, s in enumerate(rating_vec):
                item2rating[i] = s
            top_list = heapq.nlargest(top_k, item2rating, key=item2rating.get)
            user2items_top[u] = top_list
        return user2items_top

    def get_prediction_exp(self, top_k, test_tuple_list):
        test_tuple_predict = []
        for x in test_tuple_list:
            u = x[0]
            i = x[1]
            score_vec = self.__predict_exp(u, i)
            exp2score = {}
            for e, s in enumerate(score_vec):
                exp2score[e] = s
            top_list = heapq.nlargest(top_k, exp2score, key=exp2score.get)
            test_tuple_predict.append(top_list)
        return test_tuple_predict
    
    def new_item_evaluate(self, top_k, test_tuple_list, user2items_test):
        n_user = self.data_shape[0]
        n_item = self.data_shape[1]
        user2items_top = {}
        exp_NDCG = np.zeros((n_user, n_item), dtype=float)
        test_tuple_predict = []
        for x in test_tuple_list:
            u = x[0]
            i = x[1]
            exps = x[2]
            score_vec = self.__predict_exp(u, i)
            top_k_with_indices = heapq.nlargest(top_k, enumerate(score_vec), key=lambda x: x[1])
            recommends, _ = zip(*[(index, value) for index, value in top_k_with_indices])
            recommends = list(recommends)
            test_tuple_predict.append(recommends)
            h, e_num, e_ndcg, e_ap = exp_one_evaluate(exps, recommends)
            exp_NDCG[u][i] = e_ndcg
        
        item_ndcg, item_precision, item_recall, item_f1, item_ap = [], [], [], [], []
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
    
    