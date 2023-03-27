import torch
import copy

def untargeted_attack(first_net_glob, w, device, mp_lambda):
    mpaf = copy.deepcopy(w)
    for k in w.keys():
        tmp = torch.zeros_like(mpaf[k], dtype=torch.float32).to(device)
        tmp += (first_net_glob[k] - w[k].to(device)) * mp_lambda
        mpaf[k].copy_(tmp)
    return mpaf

