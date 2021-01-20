import numpy as np
import torch
from torch.utils import data
from torch import nn
train_data = np.load('train.npy', allow_pickle=True)
train_labels = np.load('train_labels.npy', allow_pickle=True)
dev_data = np.load('dev.npy', allow_pickle=True)
dev_labels = np.load('dev_labels.npy', allow_pickle=True)
test = np.load('test.npy', allow_pickle=True)

cuda = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MyDataset(data.Dataset):
    def __init__(self, X, Y, context):

        self.context = context
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        X = self.X[idx:idx+(2*self.context+1)].float().reshape(-1)
        Y = self.Y[idx].long()
        return (X, Y)


num_workers = 8 if cuda else 0


def pre_processing(input, context, labels=None):

    input = np.pad(np.concatenate((input[:]), axis=0), pad_width=(
        (context, context), (0, 0)), mode='constant', constant_values=0)

    if labels is not None:
        labels = np.concatenate((labels[:]))
    return input, labels


def train(train_data, train_labels, context=0, eval=False, dev_data=None, dev_labels=None, num_epochs=10, batch_size=512, lr=5e-4):
    train_data, train_labels = pre_processing(train_data, context, labels=train_labels)
    train_data, train_labels = torch.from_numpy(train_data), torch.from_numpy(train_labels)
    train_dataset = MyDataset(train_data, train_labels, context)
    train_loader_args = dict(shuffle=True, batch_size=batch_size, num_workers=num_workers,
                             pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)

    train_loader = data.DataLoader(train_dataset, **train_loader_args, drop_last=True)

    model = nn.Sequential(nn.Linear((((2*context)+1)*13), 1024, bias=True), nn.ReLU(), nn.BatchNorm1d(1024),
                          nn.Linear(1024, 1024, bias=True), nn.ReLU(), nn.BatchNorm1d(1024),
                          nn.Linear(1024, 1024, bias=True), nn.ReLU(), nn.BatchNorm1d(1024),
                          nn.Linear(1024, 1024, bias=True), nn.ReLU(), nn.BatchNorm1d(1024),
                          nn.Linear(1024, 1024, bias=True), nn.ReLU(), nn.BatchNorm1d(1024),
                          nn.Linear(1024, 1024, bias=True), nn.ReLU(), nn.BatchNorm1d(1024),
                          nn.Linear(1024, 1024, bias=True), nn.ReLU(), nn.BatchNorm1d(1024),
                          nn.Linear(1024, 1024, bias=True), nn.ReLU(), nn.BatchNorm1d(1024),
                          nn.Linear(1024, 346, bias=True)).to(device)

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    for epoch in range(num_epochs):

        model.train()
        size = torch.tensor(len(train_dataset)).float()
        train_count = torch.tensor(0).float()
        num = 0
        print("####### EPOCH = {} #######".format(epoch))
        print(len(train_loader))

        for (x, y) in train_loader:
            if (num % 1000 == 0):
                print(num)
            optimizer.zero_grad()
            x = x.to(device)
            y = y.to(device)
            output = model(x)
            loss = criterion(output, y)
            num += 1
            score = torch.eq(torch.argmax(output, dim=1), y).sum().float()
            train_count = train_count + score

            loss.backward()

            optimizer.step()

        # scheduler.step(running_loss)
        print("EPOCH", epoch, "TRAIN ACCURACY", train_count/size)

        if eval:
            model.eval()
            if (epoch == 0):
                dev_data, dev_labels = pre_processing(dev_data, context, labels=dev_labels)
                dev_data, dev_labels = torch.from_numpy(dev_data), torch.from_numpy(dev_labels)
            dev_dataset = MyDataset(dev_data, dev_labels, context)
            dev_loader_args = dict(shuffle=False, batch_size=batch_size, num_workers=num_workers,
                                   pin_memory=True) if cuda else dict(shuffle=False, batch_size=64)

            dev_loader = data.DataLoader(dev_dataset, **dev_loader_args, drop_last=False)
            accuracies = []
            n_correct = torch.tensor(0).float()
            total = torch.tensor(len(dev_dataset)).float()
            for (x, y) in dev_loader:
                x = x.to(device)
                y = y.to(device)
                print(x.shape)

                output = torch.argmax(model(x), dim=1)

                score = torch.eq(output, y).sum().float()
                n_correct = n_correct + score

            print(n_correct)
            print(total)
            print("VALID ACC", n_correct/total)
            accuracies.append(n_correct/total)
            model.train()

    return (model, accuracies)


mlp, acc = train(train_data, train_labels, context=15, eval=True,
                 num_epochs=5, dev_data=dev_data, dev_labels=dev_labels)
test_loader_args = dict(shuffle=False, batch_size=512, num_workers=num_workers,
                        pin_memory=True) if cuda else dict(shuffle=True, batch_size=64)
test_data = pre_processing(test, 15)[0]


test_data = torch.from_numpy(test_data)


class TestDataset(data.Dataset):
    def __init__(self, X, context):
        self.context = context
        self.X = X

    def __getitem__(self, index):
        x = self.X[index:index+(2*self.context+1)].float().reshape(-1)
        return x

    def __len__(self):
        return len(self.X) - (2*self.context)


test_data = TestDataset(test_data, 15)

test_loader_args = dict(shuffle=False, batch_size=778, num_workers=num_workers,
                        pin_memory=True) if cuda else dict(shuffle=False, batch_size=64)


test_loader = data.DataLoader(test_data, **test_loader_args, drop_last=False)
out_val = []
mlp.eval()
count = 0

print(len(test_loader))
for x in test_loader:

    x = x.to(device)
    # print(x.shape)
    print(count)
    output = mlp(x)
    output = torch.argmax(output, dim=1)
    out_val.append(list(output.cpu().numpy()))

    count += 1

outfile = open("submission.csv", 'w+')
outfile.write("ID,Label\n")

for i in range(len(out_val)):
    for j in range(len(out_val[i])):
        outfile.write(str(i*512 + j) + "," + str(out_val[i][j]) + str('\n'))


outfile.close()


print("DONE")
