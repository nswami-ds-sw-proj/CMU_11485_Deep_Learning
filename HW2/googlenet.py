import os
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader

class Inception_Aux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Inception_Aux, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((4,4))
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=1, stride=1, bias=False)
        self.linear_1 = nn.Linear(2048, 1024)
        self.dropout = nn.Dropout(0.7)
        self.linear_2 = nn.Linear(1024, num_classes)
    def forward(self, x):
        out = self.avg_pool(x)
        out = F.relu(self.conv(out))

        out = out.flatten(1)
        out = F.relu(self.linear_1(out))
        if self.training:
            out = self.dropout(out)
        out = self.linear_2(out)
        return out


class InceptionBlock(nn.Module):

    def __init__(self, in_channels, layer_1x1, layer_reduce_3x3, layer_3x3, layer_5x5_reduce, layer_5x5, pooling_output):
        super(InceptionBlock, self).__init__()

        self.one_by_one = nn.Conv2d(in_channels=in_channels, out_channels=layer_1x1, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(layer_1x1)
        self.three_by_three_reduce = nn.Conv2d(in_channels=in_channels, out_channels=layer_reduce_3x3, kernel_size=1, bias=False)
        self.bn_3_1 = nn.BatchNorm2d(layer_reduce_3x3)
        self.layer_3x3 = nn.Conv2d(in_channels=layer_reduce_3x3, out_channels=layer_3x3, kernel_size=3, padding=1, bias=False)
        self.bn_3_2 = nn.BatchNorm2d(layer_3x3)
        self.five_by_five_reduce = nn.Conv2d(in_channels=in_channels, out_channels=layer_5x5_reduce, kernel_size=1, bias=False)
        self.bn_5_1 = nn.BatchNorm2d(layer_5x5_reduce)
        self.five_by_five = nn.Conv2d(in_channels=layer_5x5_reduce, out_channels=layer_5x5, kernel_size=5, padding=2, bias=False)
        self.bn_5_2 = nn.BatchNorm2d(layer_5x5)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=1)
        self.bn_pool = nn.BatchNorm2d(pooling_output)
        self.pool_conv = nn.Conv2d(in_channels=in_channels, out_channels=pooling_output, kernel_size=1, padding=1, bias=False)
    def forward(self, x):

        layer_one = self.bn1(F.relu(self.one_by_one(x)))
        layer_3x3 = self.bn_3_1(F.relu(self.three_by_three_reduce(x)))
        layer_3x3 = self.bn_3_2(F.relu(self.layer_3x3(layer_3x3)))
        layer_5x5 = self.bn_5_1(F.relu(self.five_by_five_reduce(x)))
        layer_5x5 = self.bn_5_2(F.relu(self.five_by_five(layer_5x5)))
        pool = self.pool(x)
        pool = self.bn_pool(F.relu(self.pool_conv(pool)))
        output = torch.cat([layer_one, layer_3x3, layer_5x5, pool], dim=1)
        return output
class Network(nn.Module):
    def __init__(self, num_classes):
        super(Network, self).__init__()
        self.layers = []
        self.input_reduction = nn.Conv2d(in_channels=3, out_channels=64,
                                kernel_size=1, stride=1, bias=False)
        self.input_layer = nn.Conv2d(in_channels=64, out_channels=192, 
                                        kernel_size=3, stride=1, padding=1, bias=False)
        self.input_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.input = nn.Sequential(self.input_reduction, self.input_layer, self.input_pool)

        self.aux1 = Inception_Aux(512, num_classes)
        self.aux2 = Inception_Aux(528, num_classes)
        self.layer_3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        #Total output = 256
        self.layer_3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer_4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.layer_4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.layer_4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.layer_4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.layer_4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer_5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.layer_5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)
        self.dropout = nn.Dropout(0.4)
        self.output_layer = nn.Linear(1024, num_classes, bias=False)
        self.layers = [self.layer_3a, self.layer_3b, self.pool1, self.layer_4a, self.layer_4b, self.layer_4c, self.layer_4d, self.layer_4e, self.pool2, self.layer_5a, self.layer_5b]
        self.layers = nn.Sequential(*self.layers)
    def forward(self, x):
        x = self.input(x)
        output = self.layer_3a(x)
        output = self.layer_3b(output)
        output = self.pool1(output)
        output = self.layer_4a(output)
        aux1 = self.aux1(output)
        output = self.layer_4b(output)
        output = self.layer_4c(output)
        output = self.layer_4d(output)
        aux2 = self.aux2(output)
        output = self.layer_5b(self.layer_5a(self.pool2(self.layer_4e(output))))
        output = F.avg_pool2d(output, [output.size(2), output.size(3)], stride=1)
        
        output = output.reshape(output.shape[0], output.shape[1])
        if self.training:
            output = self.dropout(output)
        label_output = self.output_layer(output)

        return label_output, aux1, aux2
def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight.data)

class ImageDataset(Dataset):
    def __init__(self, file_list, target_list):
        self.file_list = file_list
        self.target_list = target_list
        self.n_class = len(list(set(target_list)))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img = Image.open(self.file_list[index])
        img = torchvision.transforms.ToTensor()(img)
        label = self.target_list[index]
        return img, label



def parse_data(datadir):
    img_list = []
    ID_list = []
    for root, directories, filenames in os.walk(datadir):  # root: median/1
        for filename in filenames:
            if filename.endswith('.jpg'):
                filei = os.path.join(root, filename)
                img_list.append(filei)
                ID_list.append(root.split('/')[-1])

    # construct a dictionary, where key and value correspond to ID and target
    uniqueID_list = list(set(ID_list))
    class_n = len(uniqueID_list)
    target_dict = dict(zip(uniqueID_list, range(class_n)))
    label_list = [target_dict[ID_key] for ID_key in ID_list]

    print('{}\t\t{}\n{}\t\t{}'.format('#Images', '#Labels', len(img_list), len(set(label_list))))
    return img_list, label_list, class_n


img_list, label_list, class_n = parse_data('classification_data/train_data')

def train(model, train_loader, optimizer, criterion, dev_loader, device, test_loader, scheduler=None):
    model.train()

    for epoch in range(25):
        
        avg_loss = 0.0
        print(len(train_loader))
        for batch_num, (feats, labels) in enumerate(train_loader):
            optimizer.zero_grad()

            feats =  feats.to(device)
            labels = labels.to(device)

            outputs, aux1, aux2 = model(feats)

            loss1 = criterion(outputs, labels.long())
            loss2 = criterion(aux1, labels.long())
            loss3 = criterion(aux2, labels.long())
            loss = loss1 + 0.3*(loss2 + loss3)
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

            if batch_num % 100 == 99:
                print('Epoch: {}\tBatch: {}\tAvg-Loss: {:.4f}'.format(epoch+1, batch_num+1, avg_loss/100))
                avg_loss = 0.0

            torch.cuda.empty_cache()
            del feats
            del labels
            del loss
            
            
        if ((epoch > 5) and ((epoch + 1) % 5 == 0)) or epoch==29:
            checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
            torch.save(checkpoint, 'checkpoint_epoch%d.pth'%(epoch))

        val_loss, val_acc = test_classify(model, dev_loader, device, criterion)
        train_loss, train_acc = test_classify(model, train_loader, device, criterion)
        if (epoch > 9) or (epoch % 2==0):
            test_loss, test_acc = test_classify(model, test_loader, device, criterion)
            print('Train Loss: {:.4f}\tTrain Accuracy: {:.4f}\tVal Loss: {:.4f}\tVal Accuracy: {:.4f}\tTest Loss: {:.4f}\tTest Accuracy: {:.4f}'.format(train_loss, train_acc, val_loss, val_acc, test_loss, test_acc))
        else:
            print('Train Loss: {:.4f}\tTrain Accuracy: {:.4f}\tVal Loss: {:.4f}\tVal Accuracy: {:.4f}'.format(train_loss, train_acc, val_loss, val_acc))

        scheduler.step(val_loss)
            


def test_classify(model, test_loader, device, criterion):
    model.eval()
    test_loss = []
    accuracy = 0
    total = 0
    
    with torch.no_grad():
        for batch_num, (feats, labels) in enumerate(test_loader):
            feats, labels = feats.to(device), labels.to(device)
            outputs, aux1, aux2 = model(feats)

            loss1 = criterion(outputs, labels.long())
            loss2 = criterion(aux1, labels.long())
            loss3 = criterion(aux2, labels.long())
            loss = loss1 + 0.3*(loss2 + loss3)

            _, pred_labels = torch.max(F.softmax(outputs, dim=1), 1)
            pred_labels = pred_labels.view(-1)


            accuracy += torch.sum(torch.eq(pred_labels, labels)).item()
            total += len(labels)
            test_loss.extend([loss.item()]*feats.size()[0])
            del feats
            del labels
            
        
    model.train()
    return np.mean(test_loss), accuracy/total








train_data = ImageFolder(root='classification_data/train_data', transform=torchvision.transforms.ToTensor())
val_data = ImageFolder(root='classification_data/val_data', transform=torchvision.transforms.ToTensor())
test_data = ImageFolder(root='classification_data/test_data', transform=torchvision.transforms.ToTensor())

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
train_loader = DataLoader(train_data, batch_size = 64, shuffle=True, num_workers=8) if DEVICE=='cuda' else DataLoader(train_data, shuffle=True, batch_size=64)
dev_loader = DataLoader(val_data, batch_size = 64, shuffle=True, num_workers=8) if DEVICE=='cuda' else DataLoader(val_data, shuffle=True, batch_size=64)
test_loader = DataLoader(test_data, batch_size = 64, shuffle=True, num_workers=8) if DEVICE=='cuda' else DataLoader(test_data, shuffle=True, batch_size=64)


# learningRate = .15
# weightDecay = 5e-5
num_classes = len(train_data.classes)

# criterion = CrossEntropyLoss()




model = Network(num_classes)
# model.apply(init_weights)
model.to(DEVICE)

# optimizer = SGD(model.parameters(), lr=learningRate, weight_decay=weightDecay, momentum=0.9, nesterov=True)

# schedu ler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.15, min_lr=1e-6, patience=1, verbose=True)

# train(model, train_loader, optimizer, criterion, dev_loader, DEVICE, test_loader, scheduler)

img_list, label_list, class_n = parse_data('verification_data')

checkpoint = torch.load('checkpoint_epoch14.pth')
model.load_state_dict(checkpoint['model'])
model.to('cuda')
a_list = []
b_list = []
true_label = []
cos = nn.CosineSimilarity(dim=1, eps=1e-6)
file_to_index = dict()
for i in range(len(img_list)):
    file_to_index[img_list[i]] = i
val_data = ImageDataset(img_list, label_list)
with torch.no_grad():
    with open('submission1.csv','w+') as f:
        f.write('Id,Category\n')
        with open('verification_pairs_test.txt','r') as g:
            count = 0
            model.eval()
            for line in g.readlines():
                # print('HERE')
                a,b = line.split()
                # print(a,b)
                
                new_a = model(torch.unsqueeze(val_data[file_to_index[a]][0],0).to('cuda'))
                new_b = model(torch.unsqueeze(val_data[file_to_index[b]][0],0).to('cuda'))
                new_a = new_a[0].to('cuda')
                new_b = new_b[0].to('cuda')

                c = float(cos(new_a,new_b))
                # print(c)
                f.write(str(a)+ ' ' + str(b) + ',' + str(c)+ '\n')
                count += 1

                if count % 100 ==0:
                    print(count)
            new_a.detach().cpu()
            new_b.detach().cpu()



    f.close()