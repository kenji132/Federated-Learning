import torch

def test(id, net_g, test_loader, criterions, device):
    net_g.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(-1, 28 * 28).to(device)
            target = target.to(device)
            output = net_g(data)
            loss = criterions[id](output, target)
            test_loss += loss.item()
            prediction = torch.argmax(output, dim=1)
            correct += prediction.eq(target.view_as(prediction)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))
    return correct,test_loss
