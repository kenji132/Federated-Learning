import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
import numpy as np
from torch.utils.data.dataset import Subset
from torch.utils.data import DataLoader

class Net2nn(nn.Module):
    def __init__(self):
        super(Net2nn, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

def create_model(opt, num_clients, device):
    models, optimizers, criterions = [], [], []
    for i in range(num_clients):
        model = Net2nn().to(device)
        if opt == 'SGD':
            lr = 0.2
            optimizer = optim.SGD(params=model.parameters(), lr=lr)
        elif opt == 'Adam':
            lr = 0.001
            optimizer = optim.Adam(params=model.parameters(), lr=lr)
        elif opt == 'Adagrad':
            lr = 0.01
            optimizer = optim.Adagrad(params=model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        models.append(model)
        optimizers.append(optimizer)
        criterions.append(criterion)
    return models, optimizers, criterions

def dirichlet_split_noniid(train_labels, alpha, n_clients):
    n_classes = train_labels.max() + 1
    label_distribution = np.random.dirichlet([alpha] * n_clients, n_classes)  # ディレクレ分布10(client)*10(label) labelごとに各々のclientにそのラベルが出る確率
    class_idcs = [np.argwhere(train_labels == y).flatten() for y in range(n_classes)]  # データ全部をラベル(0,1,2,3,4,5,6,7,8,9)ごとに分割
    client_idcs = [[] for _ in range(n_clients)]

    for c, fracs in zip(class_idcs, label_distribution):
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i] += [idcs]
    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]
    return client_idcs

def create_dataset_and_dataloader(client_idcs, train_data, n_clients):
    datasets = []
    dataloaders = []
    for i in range(n_clients):
        indice = list(client_idcs[i])
        if indice == []:
            datasets.append([])
            dataloaders.append([])
        else:
            sub_dataset = Subset(train_data, indice)
            dataloader = DataLoader(sub_dataset, batch_size=64, shuffle=True)
            datasets.append(sub_dataset)
            dataloaders.append(dataloader)
    return datasets, dataloaders