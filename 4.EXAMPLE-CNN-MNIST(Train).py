'''
    CIFAR10
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=20, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1)
        self.fc1 = nn.Linear(5 * 5 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x): ## x.shape -> batch-size, channel, height, weight 28 28
        x = F.relu(self.conv1(x)) ## 24x24x20 28 28 20
        x = F.max_pool2d(x, kernel_size=2, stride=2) ## 12x12x20 14 14 20
        x = F.relu(self.conv2(x)) ## 8x8x50 10 10 50
        x = F.max_pool2d(x, kernel_size=2, stride=2) ## 4x4x50 5 5 50

        ## batch-size x 800 으로 펴겠다
        print(x.shape)
        x = x.view(-1, 5 * 5 * 50) # [batch_size, 50, 4, 4] -> 풀리커넥트 만들기, batch-size 모르기때문에 일단 -1로
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x ## batch-size x 10

def train(model, device, train_loader, epoch, optimizer, criterion):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 20 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss, target은 batch-size 수
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

if __name__ == '__main__':

    batch_size = 100
    epochs = 50

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)

    test_data = datasets.CIFAR10('./data', train=False, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, **kwargs)

    model = Model().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, epoch, optimizer, criterion)
        test(model, device, test_loader, criterion)

    torch.save(model.state_dict(),"mnist_cnn.pt")
