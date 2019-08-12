import pandas as pd 
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import numpy as np
from torch import optim
from torch.nn import functional as F
import matplotlib.pyplot as plt

def create_datasets(data, valid_pct=0.1, seed=None):
    """Converts NumPy arrays into PyTorch datsets.
    
    Three datasets are created in total:
        * training dataset
        * validation dataset
        * testing (un-labelled) dataset

    """
    data_train, data_test = data
    sz = data_train.shape[0]
    idx = np.arange(sz)
    trn_idx, val_idx = train_test_split(idx, test_size=valid_pct)
    input_train = data_train.values[trn_idx, :-1]
    output_train = data_train.values[trn_idx, -1]
    trn_ds = TensorDataset(
        torch.FloatTensor(data_train.values[trn_idx, :-1]).unsqueeze(1),
        torch.LongTensor(data_train.values[trn_idx, -1]))
    val_ds = TensorDataset(
        torch.FloatTensor(data_train.values[val_idx, :-1]).unsqueeze(1),
        torch.LongTensor(data_train.values[val_idx, -1]))
    tst_ds = TensorDataset(
        torch.FloatTensor(data_test.values[:, :-1]).unsqueeze(1),
        torch.LongTensor(data_test.values[:, -1]))
    return trn_ds, val_ds, tst_ds


def create_loaders(data, bs=128, jobs=0):
    """Wraps the datasets returned by create_datasets function with data loaders."""    
    trn_ds, val_ds, tst_ds = data
    trn_dl = DataLoader(trn_ds, batch_size=bs, shuffle=True, num_workers=jobs)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=jobs)
    tst_dl = DataLoader(tst_ds, batch_size=bs, shuffle=False, num_workers=jobs)
    return trn_dl, val_dl, tst_dl



class _SepConv1d(nn.Module):
    """A simple separable convolution implementation.
    
    The separable convlution is a method to reduce number of the parameters 
    in the deep learning network for slight decrease in predictions quality.
    """
    def __init__(self, ni, no, kernel, stride, pad):
        super().__init__()
        self.depthwise = nn.Conv1d(ni, ni, kernel, stride, padding=pad, groups=ni)
        self.pointwise = nn.Conv1d(ni, no, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))



class SepConv1d(nn.Module):
    """Implementes a 1-d convolution with 'batteries included'.
    
    The module adds (optionally) activation function and dropout layers right after
    a separable convolution layer.
    """
    def __init__(self, ni, no, kernel, stride, pad, drop=None,
                 activ=lambda: nn.ReLU(inplace=True)):
    
        super().__init__()
        assert drop is None or (0.0 < drop < 1.0)
        layers = [_SepConv1d(ni, no, kernel, stride, pad)]
        if activ:
            layers.append(activ())
        if drop is not None:
            layers.append(nn.Dropout(drop))
        self.layers = nn.Sequential(*layers)
        
    def forward(self, x): 
        return self.layers(x)
        

class Flatten(nn.Module):
    """Converts N-dimensional tensor into 'flat' one."""

    def __init__(self, keep_batch_dim=True):
        super().__init__()
        self.keep_batch_dim = keep_batch_dim

    def forward(self, x):
        if self.keep_batch_dim:
            return x.view(x.size(0), -1)
        return x.view(-1)


class Classifier(nn.Module):
    def __init__(self, raw_ni, no, drop=.5):
        super().__init__()
        
        self.raw = nn.Sequential(
            SepConv1d(raw_ni,  32, 8, 1, 3, drop=drop),
            SepConv1d(    32,  64, 8, 1, 2, drop=drop),
            SepConv1d(    64, 128, 8, 1, 2, drop=drop),
            SepConv1d(   128, 256, 8, 1, 2),
            Flatten(),
            #nn.Dropout(drop), nn.Linear(256, 64), nn.ReLU(inplace=True),
            nn.Dropout(drop), nn.Linear(5120, 64), nn.ReLU(inplace=True),
            nn.Dropout(drop), nn.Linear( 64, 64), nn.ReLU(inplace=True))

        self.out = nn.Sequential(
            nn.Linear(64, 64), nn.ReLU(inplace=True), nn.Linear(64, no))
        
    def forward(self, t_raw):
        raw_out = self.raw(t_raw)
        out = self.out(raw_out)
        return out


data_train = pd.read_excel("data/HIF training data.xlsx", index_col=None, header = None)
num_in = data_train.values.shape[1] - 1
data_train = data_train.sample(frac=1).reset_index(drop=True)                           #to shuffle
data_test = pd.read_excel("data/HIF test data.xlsx", index_col=None, header = None)

valid_ratio = 0.2
datasets = create_datasets((data_train, data_test), valid_ratio)
trn_sz = len(datasets[0])
trn_dl, val_dl, tst_dl = create_loaders(datasets, bs=16)



lr = 0.001
n_epochs = 3000
iterations_per_epoch = len(trn_dl)
num_classes = 2
best_acc = 0
patience, trials = 500, 0
base = 1
step = 2
loss_history = []
acc_history = []

model = Classifier(1, num_classes).to('cuda:1')
criterion = nn.CrossEntropyLoss(reduction='sum')
opt = optim.Adam(model.parameters(), lr=lr)

print('Start model training')


for epoch in range(1, n_epochs + 1):
    
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(trn_dl):
        x_raw, y_batch = [t.to('cuda:1') for t in batch]
        opt.zero_grad()
        out = model(x_raw)
        loss = criterion(out, y_batch)
        epoch_loss += loss.item()
        loss.backward()
        opt.step()
        
    epoch_loss /= trn_sz
    loss_history.append(epoch_loss)
    
    model.eval()
    correct, total = 0, 0
    for batch in val_dl:
        x_raw, y_batch = [t.to('cuda:1') for t in batch]
        out = model(x_raw)
        preds = F.log_softmax(out, dim=1).argmax(dim=1)
        total += y_batch.size(0)
        correct += (preds == y_batch).sum().item()
    
    acc = correct / total
    acc_history.append(acc)

    if epoch % base == 0:
        print('Epoch: {:3d}. Loss: {:.4f}. Acc.: {:2.2%}'.format(epoch, epoch_loss, acc))
        base *= step

    if acc > best_acc:
        trials = 0
        best_acc = acc
        torch.save(model.state_dict(), 'best.pth')
        print('Epoch {:3d} best model saved with accuracy: {:2.2%}'.format(epoch, best_acc))
    else:
        trials += 1
        if trials >= patience:
            print('Early stopping on epoch {:3d}'.format(epoch))
            break
            
print('Done!')

fig = plt.figure(figsize=(12, 10))
ax1 = fig.add_subplot(2,1,1)
ax1.plot(loss_history, label='loss')
ax1.set_title('Validation Loss History')
ax1.set_xlabel('Epoch no.')
ax1.set_ylabel('Loss')

ax2 = fig.add_subplot(2,1,2)
ax2.plot(acc_history, label='acc')
ax2.set_title('Validation Accuracy History')
ax2.set_xlabel('Epoch no.')
ax2.set_ylabel('Accuracy');

# Save the full figure...
fig.savefig('training.png')

model.load_state_dict(torch.load('best.pth'))
model.eval()
correct, total = 0, 0
for batch in tst_dl:
    x_raw, y_batch = [t.to('cuda:1') for t in batch]
    out = model(x_raw)
    preds = F.log_softmax(out, dim=1).argmax(dim=1)
    total += y_batch.size(0)
    correct += (preds == y_batch).sum().item()
    
tst_acc = correct / total
print('Accuracy on test data: {:2.2%}'.format(tst_acc))
