import argparse
import torch
import yaml
import os
import sys
import torchvision
import numpy as np
import shutil
import seaborn as sns

from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm


CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cdr', type=str, default='checkpoints/c5')
    parser.add_argument('--imgsize', type=int, default=32)

    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()

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

    print('-' * 50)

    print('Getting model!')
    model = torch.load(os.path.join(args.cdr, 'trained_model.pth'))

    if args.cuda:
        model.cuda()

    model.eval()

    print('-' * 50)

    print('Loading test images')
    tst_dir = os.path.join('cifar-10-batches-py', 'test_ds')

    if not os.path.isdir(tst_dir):
        print('Test directory not found! Please run make_test.py first!')
        sys.exit()

    imgs = list()
    for img_name in sorted(os.listdir(tst_dir)):
        img = Image.open(os.path.join(tst_dir, img_name)).convert('RGB')
        imgs.append((img, img_name))

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((args.imgsize, args.imgsize),
                                      interpolation=torchvision.transforms.InterpolationMode.LANCZOS),
    ])

    to_tensor = torchvision.transforms.ToTensor()

    print('Processing...')

    confusion_matrix = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)

    tgt_dir = os.path.join(args.cdr, 'test_op')
    if os.path.isdir(tgt_dir):
        shutil.rmtree(tgt_dir)
    os.makedirs(tgt_dir)

    acc = 0

    pred_list = list()

    with torch.no_grad():
        for img, img_name in tqdm(imgs):
            tx_img = transforms(img)

            ip_tensor = to_tensor(tx_img)[None, ...]

            if args.cuda:
                ip_tensor = ip_tensor.cuda()

            op_tensor = model(ip_tensor)
            op_tensor = torch.softmax(op_tensor, dim=1)

            op_tensor_np = op_tensor.cpu().numpy()[0]

            max_pred = int(np.argmax(op_tensor_np))

            img_label = int(img_name.split('_')[-1][:-4])

            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(25, 10))
            axes[0].imshow(tx_img)
            axes[1].bar(x=CLASS_NAMES, height=op_tensor_np)
            axes[1].set(ylim=(0, 1.))
            fig.suptitle(f'Label: {CLASS_NAMES[img_label]} | Predicted: {CLASS_NAMES[max_pred]} '
                         f'with probability: {op_tensor_np[max_pred]}')
            plt.savefig(os.path.join(tgt_dir, 'op_' + img_name), bbox_inches='tight')
            plt.close()

            # Accuracy and confusion matrix
            confusion_matrix[img_label, max_pred] += 1

            acc += img_label == max_pred

            pred_list.append((img_name, max_pred, img_label, op_tensor_np[max_pred]))

    acc /= len(imgs)

    plt.figure(figsize=(15, 15))
    sns.heatmap(confusion_matrix, vmin=0, vmax=confusion_matrix.sum() / len(CLASS_NAMES), annot=True, fmt='d',
                xticklabels=[x + '_pred' for x in CLASS_NAMES],
                yticklabels=[x + '_actual' for x in CLASS_NAMES])
    plt.title(f'Average accuracy: {100 * acc:.3f}%')
    plt.savefig(os.path.join(args.cdr, 'confusion_matrix_test_op.png'), bbox_inches='tight')
    plt.close()

    with open(os.path.join(args.cdr, 'test_op_list.csv'), 'w') as fp:
        fp.write('Img_name,Prediction,Label,Confidence\n')
        for img_name, pred, label, prob in pred_list:
            fp.write(f'{img_name},{pred},{label},{prob}\n')

    #
    # dir_path = os.path.join(args.cdr, f'outlier_images_{args.imgsize}')
    # os.makedirs(dir_path, exist_ok=True)
    #
    # for ii, img_tx in enumerate(imgs_tx):
    #     max_pred = int(np.argmax(op_tensor_np[ii]))
    #     fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(25, 10))
    #     axes[0].imshow(img_tx)
    #     axes[1].bar(x=CLASS_NAMES, height=op_tensor_np[ii])
    #     axes[1].set(ylim=(0, 1.))
    #     fig.suptitle(f'Predicted: {CLASS_NAMES[max_pred]} with probability: {op_tensor_np[ii, max_pred]}')
    #     plt.savefig(os.path.join(dir_path, f'{str(ii).zfill(3)}.png'), bbox_inches='tight')

    print('Done!')

