import argparse
import pickle
from sulm import *
import numpy as np

def read_data(file):
    r = []
    with open(file, 'r') as f:
        lines = f.readlines()
        for i in lines:
            line = i.strip().split(' ')
            line = [int(j) for j in line]
            r.append(line)
    return r

if __name__=='__main__':
    dataset = "OpenRiceEmoji"
    logger = logging.getLogger('signature')
    logging.basicConfig(format='%(asctime)s : %(name)-12s: %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.DEBUG)
    logger.info("running %s" % ' '.join(sys.argv))
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--modelfile", help="output model file", type=argparse.FileType('rb'), 
                        default=f'save_model/{dataset}_2023')
    args = parser.parse_args()
    ratings = read_data(f'data/{dataset}/{dataset}_train.txt')
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
    print(args.modelfile)
    model = pickle.load(args.modelfile)
    testset = read_data(f'data/{dataset}/{dataset}_test.txt')
    user2item_test = get_u_i(testset)
    item_m_5, item_m_10 = model.new_item_eval(user2item_test, testset)

    print(f"Dataset: {dataset}")
    top_k = 5
    print("@5")
    print("NDCG: {:.5f}\tPrecision: {:.5f}\tRecall: {:.5f}\tF1: {:.5f}\t".
        format(item_m_5[0], item_m_5[1], item_m_5[2], item_m_5[3]))

    top_k = 10
    print("@10")
    print("NDCG: {:.5f}\tPrecision: {:.5f}\tRecall: {:.5f}\tF1: {:.5f}".
        format(item_m_10[0], item_m_10[1], item_m_10[2], item_m_10[3]))
