import numpy as np
import random
import torch
import scipy.io as scio
from torch.utils.data import DataLoader, TensorDataset


def get_data(size, batch_size=256):
    """
    size: can be either '100k' or '1m', or 'ciao'
    """
    path = 'data/' + size + '.mat'
    data = scio.loadmat(path)
    data = {'E': data.get('E'), 'D': data.get('D'), 'is': data.get('is')[0], 'iu': data.get('iu')[0]}
    train_index, test_index = [], []
    user2item_test = {}
    user2item_train = {}
    for i in range(len(data['E'])):
        if i % 5 == 4:
            test_index.append(i)
            u = data['iu'][i] - 1
            i = data['is'][i] - 1
            if u not in user2item_test:
                user2item_test[u] = [i]
            else:
                user2item_test[u].append(i)
        else:
            train_index.append(i)
            u = data['iu'][i] - 1
            i = data['is'][i] - 1
            if u not in user2item_train:
                user2item_train[u] = [i]
            else:
                user2item_train[u].append(i)
    inps = np.array([data['iu'] - 1, data['is'] - 1]).T.astype(int)  # get user, item inputs
    tgts = (data['D'][:, 0].astype(int) * 2 + 3)  # get rating targets
    features = data['E']
    with open('data/' + size + '.ratings', 'w') as f:
        for i in train_index:
            f.write(','.join(str(i) for i in [data['iu'][i] - 1, data['is'][i] - 1, 2 * data['D'][:, 0][i] + 3]) + '\n')
    # train_item_genre
    item_genre = {}
    exp_num = len(data['E'][0])
    for i in train_index:
        item_id = data['is'][i] - 1
        if item_id not in item_genre:
            item_genre[item_id] = data['E'][i]
        else:
            for j in range(exp_num):
                if data['E'][i][j] == 1:
                    item_genre[item_id][j] = 1
    with open('data/' + size + '.genre', 'w') as f:
        for i in range(max(data['is'])):
            if i not in item_genre:
                f.write(str(i) + ',' + ','.join([str(0) for _ in range(exp_num)]) + '\n')
            else:
                f.write(str(i) + ',' + ','.join([str(i) for i in item_genre[i]]) + '\n')

    # print("TRAIN:", train_index, "TEST:", test_index)
    # split and convert to tensors
    inps_train = torch.tensor(inps[train_index], dtype=torch.long)
    inps_test = torch.tensor(inps[test_index], dtype=torch.long)
    features_train = torch.tensor(features[train_index], dtype=torch.float32)
    tgts_train = torch.tensor(tgts[train_index], dtype=torch.long)
    tgts_test = torch.tensor(tgts[test_index], dtype=torch.long)
    features_test = torch.tensor(features[test_index], dtype=torch.float32)
    # convert to TensorDataset type
    train_set = TensorDataset(inps_train, tgts_train, features_train)
    test_set = TensorDataset(inps_test, tgts_test, features_test)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    # return all loaders for cross validation
    return [[train_loader, test_loader]], user2item_test, user2item_train


if __name__ == '__main__':
    dataloader = get_data('OpenRice')
    # for trainloader, testloader in dataloader:
    #     for i, data in enumerate(trainloader, 0):
    #         inputs, labels, exps = data
    #         print(inputs, labels, exps)
    print()
