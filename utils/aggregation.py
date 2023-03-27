import torch
import copy

def fedavg(w_locals):
    w_avg = copy.deepcopy(w_locals[0])
    with torch.no_grad():
        for k in w_avg.keys():
            for i in range(1, len(w_locals)):
                w_avg[k] += w_locals[i][k]
            w_avg[k] = torch.true_divide(w_avg[k], len(w_locals))
    return w_avg

def euclid(v1, v2):
    diff = v1 - v2
    return torch.matmul(diff, diff.T)

def multi_vectorization(w_locals, device):
    vectors = copy.deepcopy(w_locals)
    for i, v in enumerate(vectors):
        for name in v:
            v[name] = v[name].reshape([-1]).to(device)
        vectors[i] = torch.cat(list(v.values()))
    return vectors

def pairwise_distance(w_locals, device):
    vectors = multi_vectorization(w_locals, device)
    distance = torch.zeros([len(vectors), len(vectors)]).to(device)
    for i, v_i in enumerate(vectors):
        for j, v_j in enumerate(vectors[i:]):
            distance[i][j + i] = distance[j + i][i] = euclid(v_i, v_j)
    return distance

def trimmed_mean(w_locals, trim_factor, device):
    n = len(w_locals) - int(len(w_locals)*trim_factor*2)
    distance = pairwise_distance(w_locals, device)
    distance = distance.sum(dim=1)
    med = distance.median()
    _, chosen = torch.sort(abs(distance - med))
    chosen = chosen[: n]
    return fedavg([copy.deepcopy(w_locals[int(i)]) for i in chosen])
