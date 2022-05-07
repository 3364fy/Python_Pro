import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from build_dataset import get_dataloader, ws, MAX_LEN


class IMDBModel(nn.Module):
    def __init__(self,max_len):
        super(IMDBModel,self).__init__()
        self.embedding = nn.Embedding(len(ws),300,padding_idx=ws.PAD) #[N,300]
        self.fc = nn.Linear(max_len*300,10)  #[max_len*300,10]

    def forward(self, x):
        embed = self.embedding(x) #[batch_size,max_len,300]
        embed = embed.view(x.size(0),-1)
        out = self.fc(embed)
        return F.log_softmax(out,dim=-1)


train_batch_size = 128
test_batch_size = 1000
imdb_model = IMDBModel(MAX_LEN)
optimizer = optim.Adam(imdb_model.parameters())
criterion = nn.CrossEntropyLoss()


def train(epoch):
    mode = True
    imdb_model.train(mode)
    train_dataloader = get_dataloader(mode)
    for idx, (target, input, input_lenght) in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = imdb_model.forward(input)
        loss = F.nll_loss(output, target)  # traget需要是[0,9]，不能是[1-10]
        loss.backward()
        optimizer.step()
        if idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, idx * len(input), len(train_dataloader.dataset),
                       100. * idx / len(train_dataloader), loss.item()))

            torch.save(imdb_model.state_dict(), "model/mnist_net.pkl")
            torch.save(optimizer.state_dict(), 'model/mnist_optimizer.pkl')


def test():
    test_loss = 0
    correct = 0
    mode = False
    imdb_model.eval()
    test_dataloader = get_dataloader(mode)
    with torch.no_grad():
        for target, input, input_lenght in test_dataloader:
            output = imdb_model.forward(input)
            test_loss += F.nll_loss(output, target, reduction="sum")
            pred = torch.max(output, dim=-1, keepdim=False)[-1]
            correct = pred.eq(target.data).sum()
        test_loss = test_loss / len(test_dataloader.dataset)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_dataloader.dataset),
            100. * correct / len(test_dataloader.dataset)))


if __name__ == '__main__':
    test()
    for i in range(3):
        train(i)
        #test()
