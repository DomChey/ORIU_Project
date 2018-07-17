""""
Training, evaluation and testing routines for our GoogLeNet
@author: Dominique Cheray
"""

# necessary imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision.models import resnet
from mpii_datasets import get_train_and_validation_loader, get_test_loader
from googLeNet import GoogLeNet

# define some usefull globals
USE_CUDA = torch.cuda.is_available()
DEVICE = 'cuda' if USE_CUDA else 'cpu'
# best validation accuracy
BEST_ACC = 0
# start from 0 or last checkpoint epoch
START_EPOCH = 0
MOMENTUM = 0.9
LR = 0.01
GAMMA = 0.96


def train(epoch, model, train_loader, optimizer, criterion, augmented, log):
    """
    Training method for our googLeNet

    Args:
        epoch: Idx of trainig epoch
        model: The Network to train
        train_loader: The dataloader for the training images
        optimizer: The optimizer for the network
        criterion: The criterion for the network
        augmented: Wheter training data was augmentd
        log: the logfile to log the Accuracy and the Loss
    """

    print('Training in Epoch: {}'.format(epoch))
    log.write('Training in Epoch: {}\n'.format(epoch))

    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(DEVICE), target.to(DEVICE)
        bs, ncrops, c, h, w = data.size()
        optimizer.zero_grad()
        # if training data was augmented we have to average over the prediction for the
        # five image crops
        if augmented:
            result = model(data.view(-1, c, h, w))
            prediction = result.view(bs, ncrops, -1).mean(1)
        # if training data was not augmented just do as usual
        else:
            prediction = model(data)
        loss = criterion(prediction.float(), target)
        loss.backward()
        optimizer.step()

        train_loss = train_loss + loss.item()
        _, pred = prediction.max(1)
        total = total + target.size(0)
        correct = correct + pred.eq(target).sum().item()

    print("Loss: {:.2f} | Acc: {:.2f}".format((train_loss/len(train_loader)),
                                                             (correct/total*100)))
    log.write("Loss: {:.2f} | Acc: {:.2f}\n".format((train_loss/len(train_loader)),
                                                             (correct/total*100)))


def validation(epoch, model, valid_loader, criterion, log):
    """
    Validation method for our googLeNet

    Args:
        epoch: Idx of training epoch
        model: The network to validate
        valid_loader: The data loader for the validation images
        criterion: The criterion for the network
        log: the logfile to log the Accuracy and the Loss 
    """

    print('Validation in Epoch {}'.format(epoch))
    log.write('Validation in Epoch {}\n'.format(epoch))
    global BEST_ACC
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for idx, (data, target) in enumerate(valid_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            prediction = model(data)
            loss = criterion(prediction.float(), target)

            test_loss = test_loss + loss.item()
            _, pred = prediction.max(1)
            total = total + target.size(0)
            correct = correct + pred.eq(target).sum().item()

        print("Loss: {:.2f} | Acc: {:.2f}".format((test_loss/len(valid_loader)),
                                                             (correct/total*100)))
        log.write("Loss: {:.2f} | Acc: {:.2f}\n".format((test_loss/len(valid_loader)),
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


def resume_from_checkpoint(model, checkpoint):
    """
    Method to load a saved model

    Args:
        model: the model to add the saved parameters to
        checkpoint: which checkpoint file to load
    """

    global BEST_ACC
    global START_EPOCH
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['model'])
    BEST_ACC = checkpoint['accuracy']
    START_EPOCH = checkpoint['epoch']
    return model


def train_dat_net(start_epoch, model):
    """
    Training and validation routine for our GoogLeNet

    Args:
        epoch: Idx of epoch to start
        model: Networkt to train and validate
    """

    model = model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM)
    # scheduler to decrease learning rate by 4% every 8 epochs as described in the paper
    scheduler = StepLR(optimizer, step_size=8, gamma=GAMMA)
    # whether trainig data should be augmented or not
    augment = True
    train_loader, valid_loader = get_train_and_validation_loader(5, augment, USE_CUDA)
    # open logfile
    log = open("logfile.txt", "w")
    # now start training and validation
    for epoch in range(start_epoch, start_epoch + 200):
        scheduler.step()
        train(epoch, model, train_loader, optimizer, criterion, augment, log)
        validation(epoch, model, valid_loader, criterion, log)
    log.close()


def test_dat_net(model):
    """
    Test method for our GoogLeNet

    Args:
        model: Network to test
    """

    print("Testing")
    model.to(DEVICE)
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    test_loader = get_test_loader(5, USE_CUDA)
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(DEVICE), target.to(DEVICE)
            prediction = model(data)
            loss = criterion(prediction.float(), target)

            test_loss = test_loss + loss.item()
            _, pred = prediction.max(1)
            total = total + target.size(0)
            correct = correct + pred.eq(target).sum().item()
        print("Loss: {:.2f} | Acc: {:.2f}".format((test_loss/len(test_loader)),
                                                             (correct/total*100)))


#model = GoogLeNet(10)
#model = resume_from_checkpoint(model, GoogLeNet.t7)
model = resnet.resnet34(pretrained=False, num_classes=10)
train_dat_net(START_EPOCH, model)
test_dat_net(model)
