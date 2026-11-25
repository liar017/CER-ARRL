import random
from visdom import Visdom
from amcf import AMCF
import torch.nn as nn
import torch
from torch.optim import Adam, SGD
from dataset import get_data
import numpy as np
from evaluate import XEval


class Args(object):
    """Used to generate different sets of arguments"""

    def __init__(self):
        self.path = 'Data/'
        self.dataset = 'OpenRiceEmoji'  # '100k', '1m'， 'ymovie'
        self.epochs = 150
        self.batch_size = 256
        self.seed = 2010
        self.model_path = 'save_model/' + self.dataset + '_' + str(self.seed)
        self.e_dim = 30
        # self.mlp_dim = [64, 32, 16]
        self.reg = 1e-1
        self.bias_reg = 3e-3
        self.asp_reg = 5e-3  # 5e-3
        # self.num_neg = 4
        self.lr = 1e-4  # 7e-3
        self.bias_lr = 7e-3  # 7e-3
        self.asp_lr = 7e-2  # 7e-2
        # self.lambda0 = 0
        self.lambda1 = 100  # sim_loss coff
        # self.lambda2 = 1  # neg_loss coff
        # self.loss_weights = [1, 1, 1]


def item_negative(user, user2item_train, item_num):
    item_neg = []
    item_list = [i for i in range(item_num)]
    for i in range(0, len(user)):
        j_ = random.choice(item_list)
        while j_ in user2item_train[user[i].item()]:
            j_ = random.choice(item_list)
        item_neg.append(j_)
    return np.array(item_neg)


def train(model, trainloader, user2item_train, item_num, optimizer, criterion, device, args):
    wind = Visdom(port=7999)
    model.train()
    for epoch in range(args.epochs):
        running_loss = 0.0
        running_mse_loss = 0.0
        running_sim_loss = 0.0
        running_neg_loss = 0.0
        epoch_size = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels, item_asp = data
            # inputs and labels
            neg = item_negative(inputs[:, 0], user2item_train, item_num)
            inputs = inputs.to(device)
            labels = labels.to(device)
            item_asp = item_asp.to(device)
            user_inputs = inputs[:, 0]
            item_inputs = inputs[:, 1]
            neg = torch.tensor(np.array(neg)).to(device)
            epoch_size += len(user_inputs)  # calculate the total number of samples in an epoch

            optimizer.zero_grad()
            outputs, neg_outputs, cos_sim, pref = model(user_inputs, item_inputs, neg, item_asp)
            # cos_sim stands for a distance measure.
            outputs = outputs.flatten()
            neg_outputs = neg_outputs.flatten()
            cos_sim = cos_sim.flatten()

            mse_loss = criterion(outputs, labels.to(torch.float))  # to float
            sim_loss = cos_sim.sum()
            neg_loss = -torch.log(1e-10 + torch.sigmoid(outputs - neg_outputs)).sum()
            # combined loss
            # loss = mse_loss + (args.lambda1 * sim_loss)
            loss = neg_loss + (args.lambda1 * sim_loss)
            # loss = args.lambda0 * mse_loss + (args.lambda1 * sim_loss) + args.lambda2 * neg_loss
            loss.backward()
            optimizer.step()
            # collect running losses
            running_loss += loss.data
            running_mse_loss += mse_loss.data
            running_sim_loss += sim_loss.data
            running_neg_loss += neg_loss.item()
        # total loss
        epoch_loss = running_loss / epoch_size
        wind.line([epoch_loss.cpu().detach().numpy()], [epoch], win='train_loss', update='append')
        # rmse loss
        epoch_mse_loss = running_mse_loss / epoch_size
        rmse = np.sqrt(epoch_mse_loss.cpu().numpy())
        # sim loss
        epoch_sim_loss = running_sim_loss / epoch_size
        epoch_neg_loss = running_neg_loss / epoch_size
        print("Epoch {:d}: the training RMSE loss: {:.4f}, item embedding similarity: {:.4f}, neg loss: {:.4f}".
              format(epoch, rmse, epoch_sim_loss, epoch_neg_loss))
        print("Total loss: {:.4f}".format(epoch_loss))
        # val_rmse = test(model, testloader, evaluator, criterion, device, args)
        model.train()
    print(30 * '+' + 'training completed!' + 30 * '+')

    return model


def test(model, testloader, user_item_test, num_item, evaluator, device):
    model.eval()
    for epoch in range(1):
        i5_NDCG, i5_pre, i5_rec, i5_F1, i5_ap = evaluator.item_evaluate(model, user_item_test, num_item, 5)
        item_NDCG, item_pre, item_rec, item_F1, item_ap = evaluator.item_evaluate(model, user_item_test, num_item, 10)
        ndcg_sum, precision_sum, recall_sum, f1_sum, ap_sum = 0, 0, 0, 0, 0
        ndcg_5_sum, precision_5_sum, recall_5_sum, f1_5_sum, ap_5_sum = 0, 0, 0, 0, 0
        epoch_size = 0
        model.cpu()
        for i, data in enumerate(testloader, 0):
            inputs, labels, item_asp = data
            # get genre information from item id
            # item_asp = item_to_genre(inputs[:, 1], data_size=args.dataset).values
            # item_asp = torch.Tensor(item_asp).to(device)
            # inputs = inputs.to(device)
            # item_asp = item_asp.to(device)
            user_inputs = inputs[:, 0]
            # item_inputs = inputs[:, 1]
            u_pred = model.predict_pref(user_inputs)
            ndcg, precision, recall, f1, ap = evaluator.evaluate_exp(u_pred, item_asp, 10)
            ndcg_5, precision_5, recall_5, f1_5, ap_5 = evaluator.evaluate_exp(u_pred, item_asp, 5)
            ndcg_sum += ndcg
            precision_sum += precision
            recall_sum += recall
            f1_sum += f1
            ap_sum += ap
            epoch_size += len(user_inputs)
            ndcg_5_sum += ndcg_5
            precision_5_sum += precision_5
            recall_5_sum += recall_5
            f1_5_sum += f1_5
            ap_5_sum += ap_5
            # print(u_pred.shape)
        print("@5")
        print("NDCG: {:.5f}\tPrecision: {:.5f}\tRecall: {:.5f}\tF1: {:.5f}\tMAP: {:.5f}".
              format(i5_NDCG, i5_pre, i5_rec, i5_F1, i5_ap))
        print("NDCG: {:.5f}\tPrecision: {:.5f}\tRecall: {:.5f}\tF1: {:.5f}\tMAP: {:.5f}".
              format(ndcg_5_sum / epoch_size, precision_5_sum / epoch_size, recall_5_sum / epoch_size,
                     f1_5_sum / epoch_size, ap_5_sum / epoch_size))
        # item_result = [i5_NDCG, i5_pre, i5_rec, i5_F1, i5_ap]
        # exp_result = [ndcg_5_sum / epoch_size, precision_5_sum / epoch_size, recall_5_sum / epoch_size,
        #               f1_5_sum / epoch_size, ap_5_sum / epoch_size]
        # geo_mean = np.sqrt(np.array(item_result) * np.array(exp_result))
        # print("NDCG: {:.5f}\tPrecision: {:.5f}\tRecall: {:.5f}\tF1: {:.5f}\tMAP: {:.5f}".
        #       format(geo_mean[0], geo_mean[1], geo_mean[2], geo_mean[3], geo_mean[4]))
        print("@10")
        print("NDCG: {:.5f}\tPrecision: {:.5f}\tRecall: {:.5f}\tF1: {:.5f}\tMAP: {:.5f}".
              format(item_NDCG, item_pre, item_rec, item_F1, item_ap))
        print("NDCG: {:.5f}\tPrecision: {:.5f}\tRecall: {:.5f}\tF1: {:.5f}\tMAP: {:.5f}".
              format(ndcg_sum / epoch_size, precision_sum / epoch_size, recall_sum / epoch_size,
                     f1_sum / epoch_size, ap_sum / epoch_size))
        item_result = [item_NDCG, item_pre, item_rec, item_F1, item_ap]
        exp_result = [ndcg_sum / epoch_size, precision_sum / epoch_size, recall_sum / epoch_size,
                      f1_sum / epoch_size, ap_sum / epoch_size]
        # geo_mean = np.sqrt(np.array(item_result) * np.array(exp_result))
        # print("NDCG: {:.5f}\tPrecision: {:.5f}\tRecall: {:.5f}\tF1: {:.5f}\tMAP: {:.5f}".
        #       format(geo_mean[0], geo_mean[1], geo_mean[2], geo_mean[3], geo_mean[4]))
    return

def new_test(model, testloader, user_item_test,num_user,num_item, evaluator, device):
    model.eval()
    for epoch in range(1):
        i5_NDCG, i5_pre, i5_rec, i5_F1, i5_ap = evaluator.new_item_evaluate(model, user_item_test,num_user,num_item, 5,testloader,evaluator)
        i10_NDCG, i10_pre, i10_rec, i10_F1, i10_ap = evaluator.new_item_evaluate(model, user_item_test,num_user,num_item, 10,testloader,evaluator)

        print("@5")
        print("NDCG: {:.5f}\tPrecision: {:.5f}\tRecall: {:.5f}\tF1: {:.5f}\tMAP: {:.5f}".
              format(i5_NDCG, i5_pre, i5_rec, i5_F1, i5_ap))
        # print("NDCG: {:.5f}\tPrecision: {:.5f}\tRecall: {:.5f}\tF1: {:.5f}\tMAP: {:.5f}".
        #       format(ndcg_5_sum / epoch_size, precision_5_sum / epoch_size, recall_5_sum / epoch_size,
        #              f1_5_sum / epoch_size, ap_5_sum / epoch_size))

        print("@10")
        print("NDCG: {:.5f}\tPrecision: {:.5f}\tRecall: {:.5f}\tF1: {:.5f}\tMAP: {:.5f}".
              format(i10_NDCG, i10_pre, i10_rec, i10_F1, i10_ap))
        # print("NDCG: {:.5f}\tPrecision: {:.5f}\tRecall: {:.5f}\tF1: {:.5f}\tMAP: {:.5f}".
        #       format(ndcg_sum / epoch_size, precision_sum / epoch_size, recall_sum / epoch_size,
        #              f1_sum / epoch_size, ap_sum / epoch_size))
    return

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = Args()
    print(args.seed)
    same_seeds(args.seed)
    data_loaders, _, user_item_train = get_data(size=args.dataset, batch_size=args.batch_size)
    trainloader = data_loaders[0][0]
    num_users, num_items, num_asp = 0, 0, 0
    # determine data size
    if args.dataset == 'Yelp':
        num_users = 22123
        num_items = 16417
        num_asp = 63
    elif args.dataset == 'Beeradvocate':
        num_users = 6538
        num_items = 9032
        num_asp = 95
    elif args.dataset == 'HotelRec':
        num_users = 72612
        num_items = 38905
        num_asp = 81
    elif args.dataset == 'OpenRiceText':
        num_users = 11463
        num_items = 12915
        num_asp = 100
    elif args.dataset == 'OpenRiceEmoji':
        num_users = 11463
        num_items = 12915
        num_asp = 50
    elif args.dataset == 'OpenRice':
        num_users = 11463
        num_items = 12915
        num_asp = 150
    # load datasets to perform K-Fold cross validation
    model = AMCF(num_user=num_users, num_item=num_items, num_asp=num_asp, e_dim=args.e_dim)
    model = model.to(device)
    # optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.reg)

    # set parameter learning rates and regularizations
    # model.named_parameters(): 返回每一层的参数的名称和参数内容（权重和偏置）
    params_dict = dict(model.named_parameters())
    # for index, param in enumerate(model.parameters()):
    #     print(index, param.shape)
    params = []
    for key, value in params_dict.items():
        if key == 'i_bias' or key == 'u_bias':
            params += [{'params': [value], 'lr': args.bias_lr, 'weight_decay': args.bias_reg}]
        elif key == 'asp_emb.W':
            params += [{'params': [value], 'lr': args.asp_lr, 'weight_decay': args.asp_reg}]
        else:
            params += [{'params': [value], 'lr': args.lr, 'weight_decay': args.reg}]
    optimizer = SGD(params, lr=args.lr, weight_decay=args.reg)

    criterion = nn.MSELoss(reduction='sum')

    # calculate validation losses
    fitted_model = train(model, trainloader, user_item_train, num_items, optimizer, criterion, device, args)
    torch.save(fitted_model, args.model_path)
    # test(fitted_model, testloader, user_item_test, num_items, evaluator, device)
    # running_rmse += val_rmse
    # running_mae += val_mae
    # print('The overall {:d}-fold cross validation RMSE: {:.4f}'.format(K, running_rmse / K))
    # print('The overall {:d}-fold cross validation MAE: {:.4f}'.format(K, running_mae / K))


def eval():
    args = Args()
    print("args")
    fitted_model = torch.load(args.model_path)
    print("load_model")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fitted_model.cuda(device)
    data_loaders, user_item_test, _ = get_data(size=args.dataset, batch_size=args.batch_size)
    print("data_loade")
    testloader = data_loaders[0][1]
    if args.dataset == 'Yelp':
        num_items = 16417
    elif args.dataset == 'Beeradvocate':
        num_items = 9032
    elif args.dataset == 'HotelRec':
        num_items = 38905
    elif args.dataset == 'OpenRiceText':
        num_items = 12915
    elif args.dataset == 'OpenRiceEmoji':
        num_items = 12915
    elif args.dataset == 'OpenRice':
        num_items = 12915
    print("eval")
    evaluator = XEval(dataset=args.dataset)
    print("test")
    test(fitted_model, testloader, user_item_test, num_items, evaluator, device)


def new_eval():
    args = Args()
    print("args")
    fitted_model = torch.load(args.model_path)
    print("load_model")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    fitted_model.cuda(device)
    data_loaders, user_item_test, _ = get_data(size=args.dataset, batch_size=args.batch_size)
    print("data_loade")
    testloader = data_loaders[0][1]
    if args.dataset == 'Yelp':
        num_users = 22123
        num_items = 16417
        num_asp = 63
    elif args.dataset == 'Beeradvocate':
        num_users = 6538
        num_items = 9032
        num_asp = 95
    elif args.dataset == 'HotelRec':
        num_users = 72612
        num_items = 38905
        num_asp = 81
    elif args.dataset == 'OpenRiceText':
        num_users = 11463
        num_items = 12915
        num_asp = 100
    elif args.dataset == 'OpenRiceEmoji':
        num_users = 11463
        num_items = 12915
        num_asp = 50
    elif args.dataset == 'OpenRice':
        num_users = 11463
        num_items = 12915
        num_asp = 150
    print("eval")
    evaluator = XEval(dataset=args.dataset)
    print("test")
    new_test(fitted_model, testloader, user_item_test, num_users ,num_items, evaluator, device)

# fix random seed
def same_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数


if __name__ == '__main__':
    #main()
    eval()
    new_eval()
    
