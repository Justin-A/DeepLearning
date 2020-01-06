import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext import data, datasets

# HyperParameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")

# Data Loading
TEXT = data.Field(sequential = True, batch_first = True, lower = True)
LABEL = data.Field(sequential = False, batch_first = True)

trainset, testset = datasets.IMDB.splits(TEXT, LABEL)
TEXT.build_vocab(trainset, min_freq = 5)
LABEL.build_vocab(trainset)

# Split dataset
trainset, valset = trainset.split(split_ratio = 0.8)
train_iter, val_iter, test_iter = data.BucketIterator.splits((trainset, valset, testset), batch_size = BATCH_SIZE, shuffle = True, repeat = False)

vocab_size = len(TEXT.vocab)
n_classes = 2

print("[Train_Set]: %d [Validation_Set]: %d [Test_Set]: %d [NB_of_Vocab]: %d [Class] %d" % (len(trainset), len(valset), len(testset), vocab_size, n_classes))

class BasicGRU(nn.Module):
    def __init__(self, n_layers, hidden_dim, n_vocab, embed_dim, n_classes, dropout_p = 0.2):
        super(BasicGRU, self).__init__()
        print("Building Basic GRU Model...")
        self.n_layers = n_layers
        self.embed = nn.Embedding(n_vocab, embed_dim)
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout_p)
        self.gru = nn.GRU(embed_dim, self.hidden_dim, num_layers = self.n_layers, batch_first = True)
        self.out = nn.Linear(self.hidden_dim, n_classes)
    
    def forward(self, x):
        x = self.embed(x)
        h_0 = self._init_state(batch_size = x.size(0))
        x, _ = self.gru(x, h_0)
        h_t = x[:, -1, :]
        self.dropout(h_t)
        logit = self.out(h_t)
        return logit

    def _init_state(self, batch_size = 1):
        weight = next(self.parameters()).data
        return weight.new(self.n_layers, batch_size, self.hidden_dim).zero_()
    
def train(model, optimizer, train_iter):
    '''Training Model'''
    model.train()
    for b, batch in enumerate(train_iter):
        x, y = batch.text.to(DEVICE), batch.label.to(DEVICE)
        y.data.sub_(1)
        optimizer.zero_grad()

        logit = model(x)
        loss = F.cross_entropy(logit, y)
        loss.backward()
        optimizer.step()
    
def evaluate(model, val_iter):
    '''Evaluate Model'''
    model.eval()
    corrects, total_loss = 0, 0
    for batch in val_iter:
        x, y = batch.text.to(DEVICE), batch.label.to(DEVICE)
        y.data.sub_(1)
        logit = model(x)
        loss = F.cross_entropy(logit, y, reduction = 'sum')
        total_loss += loss.item()
        corrects += (logit.max(1)[1].view(y.size()).data == y.data).sum()
    size = len(val_iter.dataset)
    avg_loss = total_loss / size
    avg_accuracy = 100.0 * corrects / size
    return avg_loss, avg_accuracy

model = BasicGRU(1, 256, vocab_size, 128, n_classes, 0.5).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr = LEARNING_RATE)

best_val_loss = None
for e in range(1, EPOCHS + 1):
    train(model, optimizer, train_iter)
    val_loss, val_accuracy = evaluate(model, val_iter)

    print("[EPOCH: %d] Val_Error:%5.2f | Val_Accuracy: %5.2f" % (e, val_loss, val_accuracy))

    if not best_val_loss or val_loss < best_val_loss:
        if not os.path.isdir("snapshot"):
            os.makedirs("snapshot")
        torch.save(model.state_dict(), './snapshot/txtclassification.pt')
        best_val_loss = val_loss

model.load_state_dict(torch.load('./snapshot/txtclassification.pt'))
test_loss, test_acc = evaluate(model, test_iter)
print("Test_Error: %5.2f | Test_Accuray: %5.2f" % (test_loss, test_acc))
