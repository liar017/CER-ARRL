import time
import sys
import heapq
import numpy as np
import pandas as pd
import logging
import pickle
from tqdm import tqdm
import json

'''
SentimentUtilityLogisticModel class implements
    Sentiment Utility Logistic Model (SULM).
This model estimates users' and items' profiles based on information extracted
from user reviews.
The data should contain:
    - userID
    - itemID
    - overallRating - {0,1}
    - list of aspects sentiments - {0,1,nan}
'''


class SentimentUtilityLogisticModel:
    '''
    ratings - the list of data points 
    num_aspects - number of aspects in the dataset
    num_factors - number of latent factors in the model
    lambda_b, lambda_pq - regularization parameter for profile coefficients (default=0.6)
    lambda_z, lambda_w  - regularization parameter for regression weight (default=0.6)
    gamma - the coefficient for the initial gradient descent step (default=1.0)
    iterations - number of iterations for training the model (default=30)
    alpha - the relative importance between rating and sentiment estimation parts (default=0.5)
    l1 - L1 normalization
    l2 - L2 normalization
    mult - multiplication of general-user-item coefficients
    '''

    def __init__(self, logger, ratings, num_aspects, num_users, num_items,
                 num_factors=3,
                 lambda_b=0.5,
                 lambda_pq=0.5,
                 lambda_z=0.5,
                 lambda_w=0.5,
                 lambda_su=0.05,
                 gamma=1.0,
                 iterations=30,
                 alpha=0.5,
                 l1=False,
                 l2=True,
                 mult=False):

        self.logger = logger
        self.ratings = ratings
        self.num_ratings = len(ratings)
        self.num_aspects = num_aspects
        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = num_factors
        self.iterations = iterations
        self.alpha = alpha
        self.lambda_b = lambda_b
        self.lambda_pq = lambda_pq
        self.lambda_z = lambda_z
        self.lambda_w = lambda_w
        self.lambda_su = lambda_su
        self.gamma = gamma
        self.mu = None
        self.l1 = l1
        self.l2 = l2
        self.mult = mult
        self.average_sentiments()
        self.init_weight()

    def init_weight(self):
        self.profile_users_bu = np.random.normal(loc=0.0, scale=0.1, size=(self.num_users, self.num_aspects + 1))
        self.profile_users_p = np.random.normal(loc=0.0, scale=0.1, size=(self.num_users, self.num_aspects + 1,
                                                                          self.num_factors))
        self.profile_users_w = np.random.normal(loc=0.0, scale=0.1, size=(self.num_users, self.num_aspects + 1))
        self.profile_items_bi = np.random.normal(loc=0.0, scale=0.1, size=(self.num_items, self.num_aspects + 1))
        self.profile_items_q = np.random.normal(loc=0.0, scale=0.1, size=(self.num_items, self.num_aspects + 1,
                                                                          self.num_factors))
        self.profile_items_v = np.random.normal(loc=0.0, scale=0.1, size=(self.num_items, self.num_aspects + 1))
        return

    '''Calculate average \mu for parameter initialization'''

    def mu_initialization(self):
        if self.mu:
            return

        'iterate over aspects'
        aspects = zip(*self.ratings)
        self.mu = list()
        for i, aspect in enumerate(aspects):
            'rating'
            if i == 2:
                mean_rating = self.logistic_inverse(np.nanmean(aspect))
            'dismiss user_id, item_id, rating'
            if i < 3:
                continue
            self.mu.append(self.logistic_inverse(np.nanmean(aspect)))
        'last mu is for constant'
        self.mu.append(mean_rating)
        self.mu = np.array(self.mu)

    '''Calculate average sentiments'''

    def average_sentiments(self):
        aspects = zip(*self.ratings)
        self.avg_sentiments = list()
        for i, aspect in enumerate(aspects):
            'dismiss user_id, item_id, rating'
            if i < 3:
                continue
            self.avg_sentiments.append(np.nanmean(aspect))

    '''Calculate correlation between sentiments'''

    def sentiments_correlation(self):
        df = pd.DataFrame(self.ratings)
        corr = df.corr()
        df_len = len(df)
        for column in df.columns:
            frequency = len(df[df[column].notnull()])
            print('aspect: %d,\tfrequency: %d,\tpercent: %.2f' % (column, frequency, frequency / df_len * 100))
            if column in corr:
                aspect_corr = corr[(corr[column] > 0.5) | (corr[column] < -0.5)][column]
                for aspect2 in aspect_corr.index:
                    if aspect2 != column:
                        aspect2_corr = aspect_corr.ix[aspect2, column]
                        collective_frequency = len(df[(df[column].notnull()) & (df[aspect2].notnull())])
                        aspect2_frequency = len(df[df[aspect2].notnull()])
                        print('(%d,%d)\t%.3f\t%.2f\t(%.2f)' % (column, aspect2,
                                                               aspect2_corr,
                                                               100 * collective_frequency / df_len,
                                                               100 * aspect2_frequency / df_len))

    '''Train the model to fit the rating data set'''

    def train_model(self, l1=False, l2=True):
        # initialize coefficients
        self.mu_initialization()  # initialize with average values
        #         self.mu = np.random.normal(size=(self.num_aspects+1)) #random initialization
        self.logger.info('Initial mu: %s' % str(self.mu))
        self.z = np.random.normal(loc=(1.0 / self.num_aspects), scale=0.1,
                                  size=(self.num_aspects + 1))  # random initialization
        Q_old = 100000000000000000000.0
        conv_num = 0
        # make the specified number of iterations
        for i in range(self.iterations):
            t0 = time.time()
            # self.ratings - the list of arrays
            # shuffle the list of ratings on each iteration
            np.random.shuffle(self.ratings)
            flag = 0
            for num, element in enumerate(self.ratings):
                user = element[0]
                item = element[1]

                rating = element[2]
                aspect_ratings = np.append(element[3:], np.nan)
                assert len(aspect_ratings) == self.num_aspects + 1

                # identify which aspects are specified
                indicator = np.invert(np.isnan(aspect_ratings))

                # calculate aspect sentiment predictions
                sentiment_utility_prediction = self.calculate_sentiment_utility_prediction(user, item)
                # sentiment_utility_prediction_initial = sentiment_utility_prediction
                sentiment_prediction = self.logistic(sentiment_utility_prediction)
                # calculate rating predictions
                if self.mult:
                    rating_utility_prediction = sum(sentiment_utility_prediction * (
                            self.z * self.profile_users_w[user] * self.profile_items_v[item]))
                else:
                    rating_utility_prediction = sum(sentiment_utility_prediction * (
                            self.z + self.profile_users_w[user] + self.profile_items_v[item]))
                rating_prediction = self.logistic(rating_utility_prediction)

                # calculate deltas
                delta_s = aspect_ratings - sentiment_prediction
                delta_s = delta_s - (np.abs(delta_s) < 0.001) * delta_s
                delta_s = delta_s - 0.001 * (np.abs(delta_s) > 0.999) * delta_s

                delta_r = rating - rating_prediction
                delta_r = delta_r - (np.abs(delta_r) < 0.001) * delta_r
                delta_r = delta_r - 0.001 * (np.abs(delta_r) > 0.999) * delta_r

                # update vector mu
                if self.mult:
                    mu_step = self.alpha * delta_r * (
                            self.z * self.profile_users_w[user] * self.profile_items_v[item])
                else:
                    mu_step = self.alpha * delta_r * (
                            self.z + self.profile_users_w[user] + self.profile_items_v[item])
                mu_step += (1 - self.alpha) * np.nan_to_num(indicator.astype(int) * delta_s)
                if any(np.abs(mu_step) > 10000):
                    print(mu_step)
                    print('mu_step', delta_r, self.z, self.profile_users_w[user], self.profile_items_v[item])
                    print()
                    exit()

                # Fix items and update users' profiles
                # take step towards negative gradient of log likelihood
                # we take a step in negative direction because we are minimizing functional Q
                bu_step = mu_step
                if self.l2:
                    self.profile_users_bu[user] -= self.gamma * self.lambda_b * self.profile_users_bu[user]
                if self.l1:
                    self.profile_users_bu[user] -= self.gamma * self.lambda_b * np.sign(
                        self.profile_users_bu[user])
                if self.lambda_su:
                    self.profile_users_bu[user] -= self.gamma * self.lambda_su * indicator.astype(int) * \
                                                      self.profile_users_bu[user]

                self.profile_users_bu[user] += self.gamma * bu_step

                p_step = np.matrix(
                    [np.dot(self.profile_items_q[item][i], mu_step[i]) for i in range(self.num_aspects + 1)])
                if self.l2:
                    self.profile_users_p[user] -= self.gamma * self.lambda_pq * self.profile_users_p[user]
                if self.l1:
                    self.profile_users_p[user] -= self.gamma * np.sign(self.profile_users_p[user])
                if self.lambda_su:
                    self.profile_users_p[user] -= self.gamma * self.lambda_su * np.matrix(
                        [np.dot(self.profile_items_q[item][i], indicator.astype(int)[i]) for i in
                         range(self.num_aspects + 1)])

                self.profile_users_p[user] += self.gamma * p_step

                # Fix users and update items' profiles
                # take step towards negative gradient of log likelihood
                # we take a step in negative direction because we are minimizing functional Q
                # calculate aspect sentiment predictions
                sentiment_utility_prediction = self.calculate_sentiment_utility_prediction(user, item)
                sentiment_prediction = self.logistic(sentiment_utility_prediction)
                if self.mult:
                    rating_utility_prediction = sum(sentiment_utility_prediction * (
                            self.z * self.profile_users_w[user] * self.profile_items_v[item]))
                else:
                    rating_utility_prediction = sum(sentiment_utility_prediction * (
                            self.z + self.profile_users_w[user] + self.profile_items_v[item]))
                rating_prediction = self.logistic(rating_utility_prediction)

                delta_s = aspect_ratings - sentiment_prediction
                delta_s = delta_s - (np.abs(delta_s) < 0.001) * delta_s
                delta_s = delta_s - 0.001 * (np.abs(delta_s) > 0.999) * delta_s

                delta_r = rating - rating_prediction
                delta_r = delta_r - (np.abs(delta_r) < 0.001) * delta_r
                delta_r = delta_r - 0.001 * (np.abs(delta_r) > 0.999) * delta_r

                # calculate mu_step
                if self.mult:
                    mu_step = self.alpha * delta_r * (
                            self.z * self.profile_users_w[user] * self.profile_items_v[item])
                else:
                    mu_step = self.alpha * delta_r * (
                            self.z + self.profile_users_w[user] + self.profile_items_v[item])
                mu_step += (1 - self.alpha) * np.nan_to_num(indicator.astype(int) * delta_s)

                bi_step = mu_step
                if self.l2:
                    self.profile_items_bi[item] -= self.gamma * self.lambda_b * self.profile_items_bi[item]
                if self.l1:
                    self.profile_items_bi[item] -= self.gamma * self.lambda_b * np.sign(
                        self.profile_items_bi[item])
                if self.lambda_su:
                    self.profile_items_bi[item] -= self.gamma * self.lambda_su * indicator.astype(int) * \
                                                      self.profile_items_bi[item]

                self.profile_items_bi[item] += self.gamma * bi_step

                q_step = np.matrix(
                    [np.dot(self.profile_users_p[user][i], mu_step[i]) for i in range(self.num_aspects + 1)])
                if self.l2:
                    self.profile_items_q[item] -= self.gamma * self.lambda_pq * self.profile_items_q[item]
                if self.l1:
                    self.profile_items_q[item] -= self.gamma * self.lambda_pq * np.sign(
                        self.profile_items_q[item])
                if self.lambda_su:
                    self.profile_items_q[item] -= self.gamma * self.lambda_su * np.matrix(
                        [np.dot(self.profile_users_p[user][i], indicator.astype(int)[i]) for i in
                         range(self.num_aspects + 1)])

                self.profile_items_q[item] += self.gamma * q_step

                # Fix users, items profiles and solve for weights
                # take step towards negative gradient of log likelihood
                # we take a step in negative direction because we are minimizing functional Q
                # calculate aspect sentiment predictions
                sentiment_utility_prediction = self.calculate_sentiment_utility_prediction(user, item)
                sentiment_prediction = self.logistic(sentiment_utility_prediction)
                # calculate rating predictions
                if self.mult:
                    rating_utility_prediction = sum(sentiment_utility_prediction * (
                            self.z * self.profile_users_w[user] * self.profile_items_v[item]))
                else:
                    rating_utility_prediction = sum(sentiment_utility_prediction * (
                            self.z + self.profile_users_w[user] + self.profile_items_v[item]))
                rating_prediction = self.logistic(rating_utility_prediction)

                delta_r = rating - rating_prediction
                delta_r = delta_r - (np.abs(delta_r) < 0.001) * delta_r
                delta_r = delta_r - 0.001 * (np.abs(delta_r) > 0.999) * delta_r

                z_step = self.alpha * sentiment_utility_prediction * delta_r
                if self.mult:
                    z_step *= self.profile_users_w[user] * self.profile_items_v[item]

                if self.l2:
                    z_step -= self.lambda_z * self.z
                if self.l1:
                    z_step -= self.lambda_z * np.sign(self.z)

                w_step = self.alpha * sentiment_utility_prediction * delta_r
                if self.mult:
                    w_step *= self.z * self.profile_items_v[item]

                if self.l2:
                    w_step -= self.lambda_w * self.profile_users_w[user]
                if self.l1:
                    w_step -= self.lambda_w * np.sign(self.profile_users_w[user])

                v_step = self.alpha * sentiment_utility_prediction * delta_r
                if self.mult:
                    v_step *= self.z * self.profile_users_w[user]

                if self.l2:
                    v_step -= self.lambda_w * self.profile_items_v[item]
                if self.l1:
                    v_step -= self.lambda_w * np.sign(self.profile_items_v[item])

                self.z += self.gamma * z_step

                self.profile_users_w[user] += self.gamma * w_step

                self.profile_items_v[item] += self.gamma * v_step

                if num % 10000 == 0 and num > 0:
                    self.logger.debug('%d elements processed' % num)
            # update the length of gradient descent step
            if i % 20 == 0:
                self.gamma *= 0.1
            t1 = time.time()
            Q_new = self.calculate_Q() / data_length
            Q_dif = np.abs((Q_old - Q_new) / Q_old)
            Q_old = Q_new
            self.logger.info(
                'Iteration %.2i finished in %.2f seconds with Q = %.7f (diff = %.7f)' % (i + 1, t1 - t0, Q_old, Q_dif))
            if 1e-6 > Q_dif > 0:
                conv_num += 1
                if conv_num > 2:
                    self.logger.info('Model converged on iteration %.2i' % (i + 1))
                    break
            else:
                conv_num = 0

    '''Calculate the aspect sentiments predictions based on user and item profile'''

    def calculate_sentiment_utility_prediction(self, user, item):
        sentiment_utility_predictions = self.mu.copy()
        sentiment_utility_predictions += self.profile_users_bu[user]
        sentiment_utility_predictions += self.profile_items_bi[item]
        product = [np.dot(self.profile_users_p[user][i], self.profile_items_q[item][i]) for i in
                   range(self.num_aspects + 1)]
        sentiment_utility_predictions += product
        return sentiment_utility_predictions

    def calculate_sentiment_utility_prediction_batch(self, user, item):
        r = []
        sentiment_utility_predictions = self.mu.copy()
        sentiment_utility_predictions += self.profile_users_bu[user]
        for j in item:
            sentiment_utility_prediction = sentiment_utility_predictions + self.profile_items_bi[j]
            product = [np.dot(self.profile_users_p[user][i], self.profile_items_q[j][i]) for i in
                       range(self.num_aspects + 1)]
            sentiment_utility_prediction += product
            r.append(sentiment_utility_prediction)
        return np.array(r)

    '''Calculate the logistic function and its inverse'''

    def logistic(self, t):
        return 1.0 / (1 + np.exp(-t))

    def logistic_inverse(self, t):
        if t < 0.00001:
            return -40
        return -np.log(1 / t - 1)

    '''Calculate the value of the functional Q to be optimized'''

    def calculate_Q(self):
        rating_part = - self.log_likelihood_rating()
        sentiment_part = - self.log_likelihood_sentiment()
        rerularization_part = self.regularization()

        Q = self.alpha * rating_part + (1 - self.alpha) * sentiment_part + rerularization_part
        if np.isnan(Q):
            print(rating_part, self.alpha, sentiment_part, rerularization_part)
        return Q

    '''Calculate log-likelihood for the sentiment part of the model'''

    def log_likelihood_sentiment(self):
        log_likelihood = 0
        for element in self.ratings:
            user = element[0]
            item = element[1]
            aspect_ratings = element[3:]

            indicator = np.invert(np.isnan(aspect_ratings))
            'calculate aspect sentiment predictions'
            s_logistic_predictions = self.logistic(self.calculate_sentiment_utility_prediction(user, item))
            #             print(s_logistic_predictions,aspect_ratings)
            for i in range(len(indicator)):
                if indicator[i]:
                    log_likelihood += aspect_ratings[i] * np.log(s_logistic_predictions[i])
                    log_likelihood += (1 - aspect_ratings[i]) * np.log(1 - s_logistic_predictions[i])
        #             print('log_likelihood_sentiment',log_likelihood)
        return log_likelihood

    '''Calculate log-likelihood for the rating part of the model'''

    def log_likelihood_rating(self):
        log_likelihood = 0
        for element in self.ratings:
            user = element[0]
            item = element[1]
            rating = element[2]
            'calculate aspect sentiment predictions'
            s_predictions = self.calculate_sentiment_utility_prediction(user, item)
            if self.mult:
                r_prediction = sum(
                    s_predictions * (self.z * self.profile_users_w[user] * self.profile_items_v[item]))
            else:
                r_prediction = sum(
                    s_predictions * (self.z + self.profile_users_w[user] + self.profile_items_v[item]))
            if np.isnan(r_prediction):
                print(s_predictions, self.z, self.profile_users_w[user], self.profile_items_v[item])
                exit()
            r_logistic_prediction = self.logistic(r_prediction)
            if r_logistic_prediction > 0.999:
                r_logistic_prediction = 0.999
            elif r_logistic_prediction < 0.001:
                r_logistic_prediction = 0.001
            log_likelihood += rating * np.log(r_logistic_prediction)
            log_likelihood += (1 - rating) * np.log(1 - r_logistic_prediction)
            if np.isnan(log_likelihood):
                print(r_prediction, r_logistic_prediction)
                print(rating, np.log(r_logistic_prediction), np.log(1 - r_logistic_prediction))
                exit()
                break
        return log_likelihood

    ''' Calculate the L2 regularization part of the model'''

    def regularization(self):
        user_norm = dict()
        item_norm = dict()

        if self.l2:
            norm_function = np.square
        elif self.l1:
            norm_function = np.abs

        norm_z = np.sum(norm_function(self.z))

        for user in range(self.num_users):
            norm_b = np.sum(norm_function(self.profile_users_bu[user]))
            norm_pq = np.sum(norm_function(self.profile_users_p[user]))
            norm_w = np.sum(norm_function(self.profile_users_w[user]))
            user_norm[user] = self.lambda_b * norm_b + self.lambda_pq * norm_pq + self.lambda_w * norm_w

        for item in range(self.num_items):
            norm_b = np.sum(norm_function(self.profile_items_bi[item]))
            norm_pq = np.sum(norm_function(self.profile_items_q[item]))
            norm_w = np.sum(norm_function(self.profile_items_v[item]))
            item_norm[item] = self.lambda_b * norm_b + self.lambda_pq * norm_pq + self.lambda_w * norm_w

        total_norm = 0
        for element in self.ratings:
            user = element[0]
            item = element[1]
            total_norm += user_norm[user] + item_norm[item] + self.lambda_z * norm_z

            aspect_ratings = np.append(element[3:], np.nan)
            indicator = np.invert(np.isnan(aspect_ratings))
            sentiment_utility_predictions = indicator.astype(int) * self.calculate_sentiment_utility_prediction(user,
                                                                                                                item)
            total_norm += self.lambda_su * np.sum(np.square(sentiment_utility_predictions))
        return total_norm

    # Print train output ONLY for testing purposes 
    def predict_train(self):
        for i, element in enumerate(self.ratings):
            user = element[0]
            item = element[1]
            #             rating = element[2]
            # calculate aspect sentiment predictions
            s_predictions = self.calculate_sentiment_utility_prediction(user, item)
            s_logistic_predictions = self.logistic(s_predictions)
            if self.mult:
                r_prediction = sum(
                    s_predictions * (self.z * self.profile_users_w[user] * self.profile_items_v[item]))
            else:
                r_prediction = sum(
                    s_predictions * (self.z + self.profile_users_w[user] + self.profile_items_v[item]))
            r_logistic_prediction = self.logistic(r_prediction)
            message = str(element) + '\nRating: %d\tPrediction: %.3f' % (element[2], r_logistic_prediction)
            message += '\nReal sentiments: %s\nPredicted sentiments: %s' % (element[3:], s_logistic_predictions[:-1])
            self.logger.info(message)
            if i > 15:
                break

    def predict(self, user, item):
        '''
        Predict ratings and sentiments for a pair of user and item
        Input:  user_id, item_id
        Output: rating_prediction, list of sentiment_predictions
        '''

        'calculate aspect sentiment predictions'
        sentiment_utility_prediction = self.calculate_sentiment_utility_prediction(user, item)
        sentiment_predictions = self.logistic(sentiment_utility_prediction)
        r = []
        if self.mult:
            rating_utility_prediction = sum(
                sentiment_utility_prediction * (self.z * self.profile_users_w[user] * self.profile_items_v[item]))
        else:
            rating_utility_prediction = sum(
                sentiment_utility_prediction * (self.z + self.profile_users_w[user] + self.profile_items_v[item]))
        rating_prediction = self.logistic(rating_utility_prediction)
        return rating_prediction, sentiment_predictions

    def predict_batch(self, user, item):
        '''
        Predict ratings and sentiments for a pair of user and item
        Input:  user_id, item_id
        Output: rating_prediction, list of sentiment_predictions
        '''

        'calculate aspect sentiment predictions'
        sentiment_utility_prediction = self.calculate_sentiment_utility_prediction_batch(user, item)
        if self.mult:
            rating_utility_prediction = np.sum(
                sentiment_utility_prediction * (self.z * self.profile_users_w[user] * self.profile_items_v[item]), 1)
        else:
            rating_utility_prediction = np.sum(
                sentiment_utility_prediction * (self.z + self.profile_users_w[user] + self.profile_items_v[item]), 1)
        rating_prediction = self.logistic(rating_utility_prediction)
        return rating_prediction

    def calculate_aspect_impacts(self, user, item, average=False, absolute=True):
        '''Calculate aspect impacts for a given pair of user_id, item_id'''

        'calculate aspect sentiment predictions'
        sentiment_prediction = self.logistic(self.calculate_sentiment_utility_prediction(user, item)[:-1])

        if self.mult:
            aspect_impacts = (self.z * self.profile_users_w[user] * self.profile_items_v[item])[:-1]
        else:
            aspect_impacts = (self.z + self.profile_users_w[user] + self.profile_items_v[item])[:-1]

        if average:
            sentiment_difference = sentiment_prediction - self.avg_sentiments
            aspect_impacts = aspect_impacts * sentiment_difference
        else:
            aspect_impacts = aspect_impacts * sentiment_prediction

        if absolute:
            aspect_impacts = np.abs(aspect_impacts)
        return list(aspect_impacts)

    '''Predict ratings and sentiments for a given dataset'''

    def predict_test(self, testset, filename):
        result = []
        for element in testset:
            user = element[0]
            item = element[1]

            r_logistic_prediction, s_logistic_predictions = self.predict(user, item)
            result.append([user, item, r_logistic_prediction] + s_logistic_predictions.tolist())

            self.logger.info('User: %s, Item %s' % (user, item))
            self.logger.info('Aspect_impacts: ' +
                             str(self.calculate_aspect_impacts(user, item, average=True, absolute=False)))
            self.logger.info('ABSOLUTE_Aspect_impacts: ' +
                             str(self.calculate_aspect_impacts(user, item, average=True, absolute=True)))

        json.dump(result, open(filename, 'w'))
        return result

    def hit(self, target, result, k):
        hits = 0
        result = result[:k]
        dcgs = [1 / np.log(i + 2) for i in range(k)]
        ndcg = 0
        """ whether gt_item is in pred_items or not """
        for idx, item in enumerate(result):
            # print(item)
            if item in target:
                hits += 1
                ndcg += dcgs[idx]
        m = min(k, len(target))
        idcg = 0
        for i in range(0, m):
            idcg += dcgs[i]
        if m == 0:
            return hits, 0, 0
        return hits, ndcg / idcg

    def item_eval(self, user2item_test):
        item_NDCG_10, item_pre_10, item_rec10, item_F1_10, item_ap_10 = [], [], [], [], []
        item_NDCG_5, item_pre_5, item_rec5, item_F1_5, item_ap_5 = [], [], [], [], []
        for user in user2item_test:
            its = np.array([i for i in range(self.num_items)])
            score, _ = self.predict_batch(user, its)
            item2rating = {}
            for i, s in enumerate(score):
                item2rating[i] = s
            recommends = heapq.nlargest(10, item2rating, key=item2rating.get)
            # recommends = recommends.cpu()
            # -----------------k = 10---------------------------
            k = 10
            h, i_ndcg, i_ap = self.hit(user2item_test[user], recommends, k)
            pre = h / k
            item_pre_10.append(pre)
            rec = h / len(user2item_test[user])
            item_rec10.append(rec)
            item_NDCG_10.append(i_ndcg)
            item_ap_10.append(i_ap)
            if (h / k + h) > 0:
                item_F1_10.append(2 * pre * rec / (pre + rec))
            else:
                item_F1_10.append(0)
            # -----------------k = 5---------------------------
            k = 5
            h, i_ndcg, i_ap = self.hit(user2item_test[user], recommends, k)
            pre = h / k
            item_pre_5.append(pre)
            rec = h / len(user2item_test[user])
            item_rec5.append(rec)
            item_NDCG_5.append(i_ndcg)
            item_ap_5.append(i_ap)
            if (h / k + h) > 0:
                item_F1_5.append(2 * pre * rec / (pre + rec))
            else:
                item_F1_5.append(0)
        return [np.mean(item_NDCG_5), np.mean(item_pre_5), np.mean(item_rec5), np.mean(item_F1_5), np.mean(item_ap_5)], \
               [np.mean(item_NDCG_10), np.mean(item_pre_10), np.mean(item_rec10), np.mean(item_F1_10), np.mean(item_ap_10)] 


    def exp_hit(self, target, result):
        hits = 0
        exp_num = 0
        k = len(result)
        dcgs = [1 / np.log(i + 2) for i in range(k)]
        ndcg = 0
        aps = [1 / i for i in range(1, k + 1)]
        ap = 0
        for i in target:
            if i == 1:
                exp_num += 1
        """ whether gt_item is in pred_items or not """
        for idx, item in enumerate(result):
            if target[item] == 1:
                hits += 1
                ndcg += dcgs[idx]
                ap += hits * aps[idx]
        m = min(k, exp_num)
        idcg = 0
        for i in range(0, m):
            idcg += dcgs[i]
        if m == 0:
            return hits, exp_num, 0, 0
        return hits, exp_num, ndcg / idcg, ap / m

    def exp_evaluate(self, test_set, k):
        exp_NDCG, exp_pre, exp_rec, exp_F1, exp_ap = [], [], [], [], []
        for element in test_set:
            user = element[0]
            item = element[1]
            label = element[3: ]
            _, exps_score = self.predict(user, item)
            exp2rating = {}
            for e, s in enumerate(exps_score[:-1]):
                exp2rating[e] = s
            recommends = heapq.nlargest(k, exp2rating, key=exp2rating.get)
            h, e_num, e_NDCG, e_ap = self.exp_hit(label, recommends)
            pre = h / k
            exp_pre.append(pre)
            if e_num != 0:
                rec = h / e_num
            else:
                rec = 0
            exp_rec.append(rec)
            exp_NDCG.append(e_NDCG)
            exp_ap.append(e_ap)
            if (h / k + h) > 0:
                exp_F1.append(2 * pre * rec / (pre + rec))
            else:
                exp_F1.append(0)
        print("NDCG: {:.5f}\tPrecision: {:.5f}\tRecall: {:.5f}\tF1: {:.5f}\tMAP: {:.5f}".
              format(np.mean(exp_NDCG), np.mean(exp_pre), np.mean(exp_rec), np.mean(exp_F1), np.mean(exp_ap)))
        # logger.info("NDCG: {:.5f}\tPrecision: {:.5f}\tRecall: {:.5f}\tF1: {:.5f}\tMAP: {:.5f}".
        #       format(np.mean(exp_NDCG), np.mean(exp_pre), np.mean(exp_rec), np.mean(exp_F1), np.mean(exp_ap)))
        return np.mean(exp_NDCG), np.mean(exp_pre), np.mean(exp_rec), np.mean(exp_F1), np.mean(exp_ap)

    '''Print the model to file in the readable format'''

    def pretty_save(self, filename):
        model_file = open(filename, 'w')
        model_file.write('\mu = ' + np.array_str(self.mu) + '\n')
        for user in range(self.num_users):
            model_file.write('\n***********\n')
            model_file.write(str(user) + '\nbu = ' + np.array_str(self.profile_users_bu[user]) + '\n')
            model_file.write('p = ' + np.array_str(self.profile_users_p[user]) + '\n')
            model_file.write('w = ' + np.array_str(self.profile_users_w[user]) + '\n')
        model_file.write('\n===========================\n===========================\n\n')
        for item in range(self.num_items):
            model_file.write('\n***********\n')
            model_file.write(str(item) + '\nbi = ' + np.array_str(self.profile_items_bi[item]) + '\n')
            model_file.write('q = ' + np.array_str(self.profile_items_q[item]) + '\n')
            model_file.write('v = ' + np.array_str(self.profile_items_v[item]) + '\n')
        model_file.write('\n===========================\n===========================\n\n')
        model_file.write('z = ' + np.array_str(self.z) + '\n')
        model_file.close()

    '''Save the model'''

    def new_item_eval(self, user2item_test, test_set):
        exp_NDCG_10 = np.zeros((self.num_users, self.num_items), dtype=float)
        exp_NDCG_5 = np.zeros((self.num_users, self.num_items), dtype=float)

        for element in tqdm(test_set, desc="Processing elements"):
            user = element[0]
            item = element[1]
            label = element[3: ]
            _, exps_score = self.predict(user, item)
            exp2rating = {}
            for e, s in enumerate(exps_score[:-1]):
                exp2rating[e] = s
            recommends_10 = heapq.nlargest(10, exp2rating, key=exp2rating.get)
            recommends_5 = heapq.nlargest(5, exp2rating, key=exp2rating.get)
            _, _, e_NDCG_10, _ = self.exp_hit(label, recommends_10)
            _, _, e_NDCG_5, _ = self.exp_hit(label, recommends_5)
            exp_NDCG_10[user][item] = e_NDCG_10
            exp_NDCG_5[user][item] = e_NDCG_5
        
        item_NDCG_10, item_pre_10, item_rec10, item_F1_10 = [], [], [], []
        item_NDCG_5, item_pre_5, item_rec5, item_F1_5 = [], [], [], []
        
        its = np.array([i for i in range(self.num_items)])
        for user in tqdm(user2item_test, desc="Processing"):
            
            score = self.predict_batch(user, its)
            
            score_10 = score + exp_NDCG_10[user]
            score_5 = score + exp_NDCG_5[user]
                        
            recommends_10 = heapq.nlargest(10, enumerate(score_10), key=lambda x: x[1])
            recommends_5 = heapq.nlargest(5, enumerate(score_5), key=lambda x: x[1])

            # -----------------k = 10---------------------------
            k = 10
            h, i_ndcg = self.hit(user2item_test[user], recommends_10, k)
            pre = h / k
            item_pre_10.append(pre)
            rec = h / len(user2item_test[user])
            item_rec10.append(rec)
            item_NDCG_10.append(i_ndcg)
            if (h / k + h) > 0:
                item_F1_10.append(2 * pre * rec / (pre + rec))
            else:
                item_F1_10.append(0)
            # -----------------k = 5---------------------------
            k = 5
            h, i_ndcg = self.hit(user2item_test[user], recommends_5, k)
            pre = h / k
            item_pre_5.append(pre)
            rec = h / len(user2item_test[user])
            item_rec5.append(rec)
            item_NDCG_5.append(i_ndcg)
            if (h / k + h) > 0:
                item_F1_5.append(2 * pre * rec / (pre + rec))
            else:
                item_F1_5.append(0)
        return [np.mean(item_NDCG_5), np.mean(item_pre_5), np.mean(item_rec5), np.mean(item_F1_5)], \
               [np.mean(item_NDCG_10), np.mean(item_pre_10), np.mean(item_rec10), np.mean(item_F1_10)]

    
    
    def save(self, filename):
        pickle.dump(self.mu, open(filename + 'mu', 'wb'))
        pickle.dump(self.avg_sentiments, open(filename + 'av_sent', 'wb'))
        pickle.dump(self.z, open(filename + 'z', 'wb'))
        pickle.dump(self.profile_users, open(filename + 'user_profiles', 'wb'))
        pickle.dump(self.profile_items, open(filename + 'item_profiles', 'wb'))

    '''Load the model'''

    def load(self, filename):
        self.mu = pickle.load(open(filename + 'mu', 'rb'))
        self.avg_sentiments = pickle.load(open(filename + 'av_sent', 'rb'))
        self.z = pickle.load(open(filename + 'z', 'rb'))
        self.profile_users = pickle.load(open(filename + 'user_profiles', 'rb'))
        self.profile_items = pickle.load(open(filename + 'item_profiles', 'rb'))


def read_data(file):
    r = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for i in lines:
            line = i.strip().split(' ')
            line = [int(j) for j in line]
            r.append(line)
    return r


def get_u_i(test):
    u_i = {}
    for i in range(len(test)):
        if test[i][0] not in u_i:
            u_i[test[i][0]] = [test[i][1]]
        else:
            u_i[test[i][0]].append(test[i][1])
    return u_i


# fix random seed
def same_seeds(seed):
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数


if __name__ == '__main__':
    for seed in [2010,2017,2020, 2023, 2024]:
        logger = logging.getLogger('signature')
        logging.basicConfig(format='%(asctime)s : %(name)-12s: %(levelname)s : %(message)s')
        logging.root.setLevel(level=logging.DEBUG)
        logger.info("running %s" % ' '.join(sys.argv))
        # seed = 2010
        same_seeds(seed)
        # ratings = read_data('data/OpenRiceText/OpenRiceText_train.txt')
        ratings = read_data('data/HotelRec/HotelRec_train.txt')
        r = np.array(ratings)
        data_length = len(r)
        users = r[:, 0]
        items = r[:, 1]
        user_num = max(users) + 1
        item_num = max(items) + 1
        aspects = ratings[0][3:]
        model = SentimentUtilityLogisticModel(logger, ratings, num_aspects=len(aspects), num_users=user_num, num_items=item_num,
                                              num_factors=32, lambda_b=0.05, lambda_pq=0.05, lambda_z=0.05, lambda_w=0.05, 
                                              gamma=0.0001, iterations=200, alpha=0.1, l1=False, l2=True, mult=False)

            # model.sentiments_correlation()
        model.train_model()

        logger.info('Average Sentiments:\n%s' % str(list(zip(aspects, model.avg_sentiments))))
        # model.pretty_save('readable_model.txt')
        # model.predict_train()
        path = 'save_model/new_HotelRec_' + str(seed)
        # path = 'save_model/new_OpenRiceText_' + str(seed)
        with open(path, 'wb') as f:
            pickle.dump(model, f)
