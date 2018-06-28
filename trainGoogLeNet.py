""""
Training and evaluation routine for our GoogLeNet
@author: Dominique Cheray
"""

# necessary imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from mpii_datasets import get_train_and_validation_loader
from googLeNet import GoogLeNet

# define some usefull globals
USE_CUDA = torch.cuda.is_available()
DEVICE = 'cuda' if USE_CUDA else 'cpu'

BEST_ACC = 0 # best validation accuracy
START_EPOCH = 0 # start from 0 or last checkpoint epoch

MOMENTUM = 0.9
LR = 0.01
GAMMA = 0.96

def train(epoch, model, train_loader, optimizer, criterion):
    print('Trainining in Epoch: {}'.format(epoch))

    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        optimizer.zero_grad()
        prediction = model(data)
        loss = criterion(prediction, target)
        loss.backward()
        optimizer.step()

        train_loss = train_loss + loss.item()
        _, pred = prediction.max(1)
        total = total + target.size(0)
        correct = correct + pred.eq(target).sum().item()

        print("Batch: {},| Loss: {:.2f} | Acc: {:.2f}".format(idx, (train_loss/idx),
                                                             (correct/total*100)))


def validation(epoch, model, valid_loader, criterion):
    print('Validation in Epoch {}'.format(epoch))
    global BEST_ACC
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for idx, (data, target) in enumerate(valid_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            prediction = model(data)
            loss = criterion(prediction, target)

            test_loss = test_loss + loss.item()
            _, pred = prediction.max(1)
            total = total + target.size(0)
            correct = correct + pred.eq(target).sum().item()

            print("Batch: {},| Loss: {:.2f} | Acc: {:.2f}".format(idx, (test_loss/idx),
                                                             (correct/total*100)))

    # if the model performs well, save it
    accuracy = correct/total*100
    if accuracy > BEST_ACC:
        print('Saving model')
        state = {
            'model': model.state_dict(),
            'accuracy': accuracy,
            'epoch': epoch,
        }
    torch.save(state, 'ckpt.t7')
    BEST_ACC = accuracy


def resume_from_checkpoint(model):
    global BEST_ACC
    global START_EPOCH
    checkpoint = torch.load('ckpt.t7')
    model.load_state_dict(checkpoint['model'])
    BEST_ACC = checkpoint['accuracy']
    START_EPOCH = checkpoint['epoch']


def train_dat_net(start_epoch):
    train_loader, valid_loader = get_train_and_validation_loader(8, False, USE_CUDA)
    model = GoogLeNet(10)
    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    # scheduler to decrease learning rate by 4% every 8 epochs as described in the paper
    scheduler = StepLR(optimizer, step_size=8, gamma=GAMMA)

    # now start training and validation
    for epoch in range(start_epoch, start_epoch +1):
        scheduler.step()
        train(epoch, model, train_loader, optimizer, criterion)
        validation(epoch, model, valid_loader, criterion)


train_dat_net(START_EPOCH)
