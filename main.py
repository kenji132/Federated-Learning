import torch
import numpy as np
import random
from torchvision import datasets
import torchvision.transforms as transforms
import math
import copy

from utils.attacks import untargeted_attack
from utils.aggregation import fedavg, trimmed_mean
from utils.test import test
from utils.update import train
from utils.setting import Net2nn, create_model, dirichlet_split_noniid, create_dataset_and_dataloader
from utils.options import args_parser

if __name__ == '__main__':
    args = args_parser()
    print(args)
    N_CLIENTS = 20
    numEpoch = 100
    local_ep = 1
    DIRICHLET_ALPHA = 1
    ATTACK = 'mpaf'
    mp_lambda = 10**6
    METHOD = 'trimmed_mean'
    trim_factor = 0.15
    batch_size = 64
    opt = 'Adam'
    np.random.seed(42)
    random.seed(42)
    use_cuda = True
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

    # MNISTデータセットのダウンロード
    train_data = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    test_data = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size)

    # ディレクレ分布に従ったデータセットの作成
    x_train = np.array(train_data.data)
    y_train = np.array(train_data.targets)
    x_train, y_train = map(torch.tensor, (x_train, y_train))
    y_train_client_idcs = dirichlet_split_noniid(y_train, DIRICHLET_ALPHA, N_CLIENTS)
    client_datasets, client_dataloaders = create_dataset_and_dataloader(y_train_client_idcs, train_data, N_CLIENTS)

    all_result = []
    # epoch数
    for i in range(1):
        models, optimizers, criterions = create_model(opt, N_CLIENTS, device)
        net_glob = Net2nn().to(device)

        w_glob = net_glob.state_dict()

        # fake client
        fake_client_rate = 0.4
        fake_clients = math.floor(N_CLIENTS * fake_client_rate)
        client_num_list = list(range(0, N_CLIENTS))
        fake_client_num_list = random.sample(client_num_list, fake_clients)
        print("fake_client")
        print(fake_client_num_list)
        benign_clisent_num_list = [i for i in range(N_CLIENTS)]
        for f in fake_client_num_list:
            if f in benign_clisent_num_list:
                benign_clisent_num_list.remove(f)
        print("benign client")
        print(benign_clisent_num_list)

        # sampling client
        sampling_rate = 1
        selected_clients = max(int(sampling_rate * N_CLIENTS), 1)
        compromised_num = int(selected_clients * fake_client_rate)
        test_result = []

        for epoch in range(numEpoch):
            print(epoch)
            w_locals = []

            idxs_users=[i for i in range(0,20,1)]
            #local training
            for idx in idxs_users:
                if idx in fake_client_num_list:
                    # fake client
                    w_locals.append(copy.deepcopy(untargeted_attack(w_glob, models[idx].state_dict(), device, mp_lambda)))
                else:
                    # benign client
                    print("-----")
                    train(idx, models, local_ep, client_dataloaders, client_datasets, optimizers, criterions, device)
                    w_locals.append(copy.deepcopy(models[idx].state_dict()))
            # aggregation
            if METHOD == 'fedavg':
                w_glob = fedavg(w_locals, device)
            elif METHOD == 'trimmed_mean':
                w_glob = trimmed_mean(w_locals, trim_factor, device)
            else:
                print('error')

            net_glob.load_state_dict(w_glob)
            net_glob.to(device)

            # global model disemination
            with torch.no_grad():
                for i in range(N_CLIENTS):
                    model = models[i]
                    model.fc1.weight.data = net_glob.fc1.weight.data.clone()
                    model.fc1.bias.data = net_glob.fc1.bias.data.clone()
                    model.fc2.weight.data = net_glob.fc2.weight.data.clone()
                    model.fc2.bias.data = net_glob.fc2.bias.data.clone()
                    model.fc3.weight.data = net_glob.fc3.weight.data.clone()
                    model.fc3.bias.data = net_glob.fc3.bias.data.clone()
                    models[i] = model
            # test
            correct, test_loss = test(idx, net_glob, test_dataloader,criterions, device)
            test_acc = 100. * correct / len(test_dataloader.dataset)
            test_loss /= len(test_dataloader.dataset)
            test_result.append(test_acc)
            print(test_result)
        print(i)
        all_result.append(test_result)
        print(all_result)
