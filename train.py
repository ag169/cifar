import argparse
import os
import sys
import yaml
import datetime
import torch
import torch.nn as nn
import torch.optim
import time
import numpy as np
import seaborn as sns

from dataset import get_cifar10_loaders
from models import get_model
from tqdm import tqdm
from matplotlib import pyplot as plt
from utils import print_and_log


# Set seed for repeatable experiments
torch.random.manual_seed(1947)


def train(model_, loss_, optimizer_, data_loader, epoch_):
    time_ = time.time()

    model_.train()

    total = 0
    correct = 0

    total_loss = 0

    for data in tqdm(data_loader):
        optimizer_.zero_grad()

        images = data[0]
        class_labels = data[1]

        if args.cuda:
            images = images.cuda()
            class_labels = class_labels.cuda()

        with torch.cuda.amp.autocast(enabled=args.amp):
            model_op = model_(images)

            loss_op = loss_(model_op, class_labels)

        if args.amp:
            amp_scaler.scale(loss_op).backward()
            amp_scaler.step(optimizer_)
            amp_scaler.update()
        else:
            loss_op.backward()
            optimizer_.step()

        total_loss += loss_op.item() * images.size(0)

        pred_labels = torch.argmax(model_op, dim=1)

        correct += (pred_labels == class_labels).sum().item()
        total += images.size(0)

    tqdm._instances.clear()

    accuracy_avg = correct / total
    loss_avg = total_loss / total

    time_ = (time.time() - time_) / 60

    print(f"\nTrain | Epoch: {epoch_} \t Avg Loss: {loss_avg:.5f}\tAvg Acc: {accuracy_avg:.5f}\tTime: {time_:.3f} mins")

    return {
        'loss': loss_avg,
        'acc': accuracy_avg,
        'time': time_
    }


def evaluate(model_, data_loader, epoch_, loss_=None, num_classes_=10):
    time_ = time.time()

    model_.eval()

    total = 0
    correct = 0

    confusion_matrix = np.zeros((num_classes_, num_classes_), dtype=int)

    if loss_ is None:
        total_loss = None
    else:
        total_loss = 0

    for data in tqdm(data_loader):
        images = data[0]
        class_labels = data[1]

        if args.cuda:
            images = images.cuda()
            class_labels = class_labels.cuda()

        with torch.cuda.amp.autocast(enabled=args.amp):
            model_op = model_(images)

            if loss_ is not None:
                loss_op = loss_(model_op, class_labels)
                total_loss += loss_op.item()

        pred_labels = torch.argmax(model_op, dim=1)

        for pred_label, class_label in zip(pred_labels, class_labels):
            confusion_matrix[class_label, pred_label] += 1

        correct += (pred_labels == class_labels).sum().item()
        total += images.size(0)

    tqdm._instances.clear()

    accuracy_avg = correct / total

    time_ = (time.time() - time_) / 60

    if loss_ is not None:
        loss_avg = total_loss / total
        print(f"\nTest | Epoch: {epoch_} \t Avg Loss: {loss_avg:.5f}\tAvg Acc: {accuracy_avg:.5f}\tTime: {time_:.3f} mins")
    else:
        loss_avg = None
        print(f"\nTest | Epoch: {epoch_} \t Avg Acc: {accuracy_avg:.5f}\tTime: {time_:.3f} mins")

    return {
        'loss': loss_avg,
        'acc': accuracy_avg,
        'time': time_,
        'confusion_matrix': confusion_matrix
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cdr', type=str, default='checkpoints/c1')

    parser.add_argument('--amp', action='store_true', default=False,
                        help='Enable Mixed-Precision Training')

    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()

    tqdm._instances.clear()

    if not os.path.isdir(args.cdr):
        print(f'Directory {args.cdr} not found!')
        sys.exit()

    cfg_path = os.path.join(args.cdr, 'config.yml')

    if not os.path.isfile(cfg_path):
        print(f'Cfg file not found at {cfg_path}!')
        sys.exit()

    with open(cfg_path) as fp:
        cfg = yaml.safe_load(fp)

    print('-' * 50)
    print('Config is as follows:', end='\n\n')
    for k, v in cfg.items():
        print(f'{k}: {v}')

    print('-' * 50)

    # TODO: Add print and logger
    log_file = args.cdr + '/log_' + str(datetime.datetime.now()).replace(':', '-').replace(' ', '_') + '.txt'
    print_and_log(log_file)

    print('Getting dataloaders!')
    data_loaders = get_cifar10_loaders(cfg.get('batchsize', 32))

    print('-' * 50)

    print('Getting model!')
    model_arch = cfg['model']['arch']
    model = get_model(model_arch)

    if args.cuda:
        model.cuda()

    print('-' * 50)

    print('Getting loss!')
    loss_fn = cfg.get('loss', 'ce')

    if loss_fn == 'ce':
        loss = nn.CrossEntropyLoss()
    else:
        raise NotImplementedError(f"Loss function: {loss_fn} not implemented yet!")

    if args.cuda:
        loss.cuda()

    print('-' * 50)

    print('Getting optimizer!')
    opt_name = cfg.get('optimizer', 'adam').lower()
    if opt_name == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.get('lr', 1.e-3),
                                    weight_decay=cfg.get('weight_decay', 0))
    elif opt_name == 'adam':
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.get('lr', 1.e-3),
                                      weight_decay=cfg.get('weight_decay', 0))
    else:
        raise NotImplementedError(f"Optimizer function: {opt_name} not implemented yet!")

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    print('-' * 50)

    num_epochs = cfg.get('nepochs', 25)

    if args.amp:
        print('Using Mixed-Precision Training!')
        amp_scaler = torch.cuda.amp.GradScaler()

    print(f'Training the model for {num_epochs} epochs!')

    train_losses = []
    test_losses = []

    train_accs = []
    test_accs = []

    print('-' * 80)

    total_train_time = 0
    num_classes = len(data_loaders['metadata']['label_names'])

    for epoch in range(num_epochs):
        train_res = train(model, loss, optimizer, data_loaders['train_loader'], epoch + 1)

        train_losses.append(train_res['loss'])
        train_accs.append(train_res['acc'])

        total_train_time += train_res['time']

        scheduler.step()

        print()

        with torch.no_grad():
            test_res = evaluate(model, data_loaders['test_loader'], epoch+1, loss_=loss, num_classes_=num_classes)

        test_accs.append(test_res['acc'])
        test_losses.append(test_res['loss'])

        print('-' * 80)

    print('Training complete!')
    print(f'Total train time: {total_train_time} mins')

    torch.save(model, os.path.join(args.cdr, 'trained_model.pth'))

    print('-' * 50)
    print('Final test results:')
    for k, v in test_res.items():
        if k in ['confusion_matrix']:
            print(f'{k}:\n{v}')
        else:
            print(f'{k}: {v}')

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    epoch_arr = np.arange(num_epochs) + 1
    axes[0].plot(epoch_arr, train_losses, label='Train Loss')
    axes[0].plot(epoch_arr, test_losses, label='Test Loss')
    axes[0].set_ylabel('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend(loc="upper right")

    axes[1].plot(epoch_arr, train_accs, label='Train Accuracy')
    axes[1].plot(epoch_arr, test_accs, label='Test Accuracy')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].legend(loc="upper right")

    fig.suptitle(f'Train-Plot for Model: {model_arch}')

    plt.savefig(os.path.join(args.cdr, 'train_plot.png'), bbox_inches='tight')

    plt.figure(figsize=(15, 15))
    conf_mat = test_res['confusion_matrix'].copy()
    sns.heatmap(conf_mat, vmin=0, vmax=conf_mat.sum()/num_classes, annot=True, fmt='d')
    plt.savefig(os.path.join(args.cdr, 'confusion_matrix.png'), bbox_inches='tight')

    print('Done')
