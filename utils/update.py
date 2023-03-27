# benign clients
def train(id, models, local_ep, dataloaders, datasets, optimizers, criterions, device):
    models[id].train()
    for iter in range(local_ep):
        for batch_idx, (data, target) in enumerate(dataloaders[id]):
            size = len(datasets[id])
            optimizers[id].zero_grad()
            data = data.view(-1, 28 * 28).to(device)
            target = target.view(len(target)).to(device)
            log_probs = models[id](data)
            loss = criterions[id](log_probs, target)
            loss.backward()
            optimizers[id].step()
            if batch_idx % 100 == 0:
                loss, current = loss, batch_idx * len(data)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
