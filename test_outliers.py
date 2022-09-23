import argparse
import torch
import yaml
import os
import sys
import torchvision
import numpy as np

from matplotlib import pyplot as plt
from PIL import Image


CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--cdr', type=str, default='checkpoints/c1')
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

    print('Loading outlier images')

    imgs = list()
    for img_name in os.listdir('outlier_images'):
        img = Image.open(os.path.join('outlier_images', img_name)).convert('RGB')
        imgs.append(img)

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((args.imgsize, args.imgsize),
                                      interpolation=torchvision.transforms.InterpolationMode.LANCZOS),
    ])

    to_tensor = torchvision.transforms.ToTensor()

    print('Processing...')

    imgs_tx = list()
    imgs_t = list()
    for img in imgs:
        tx_img = transforms(img)
        imgs_tx.append(tx_img)
        imgs_t.append(to_tensor(tx_img))

    ip_tensor = torch.stack(imgs_t, dim=0)

    if args.cuda:
        ip_tensor = ip_tensor.cuda()

    print('-' * 50)

    print('Saving output images...')

    with torch.no_grad():
        op_tensor = model(ip_tensor)
        op_tensor = torch.softmax(op_tensor, dim=1)

        op_tensor_np = op_tensor.cpu().numpy()

    dir_path = os.path.join(args.cdr, f'outlier_images_{args.imgsize}')
    os.makedirs(dir_path, exist_ok=True)

    for ii, img_tx in enumerate(imgs_tx):
        max_pred = int(np.argmax(op_tensor_np[ii]))
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(25, 10))
        axes[0].imshow(img_tx)
        axes[1].bar(x=CLASS_NAMES, height=op_tensor_np[ii])
        axes[1].set(ylim=(0, 1.))
        fig.suptitle(f'Predicted: {CLASS_NAMES[max_pred]} with probability: {op_tensor_np[ii, max_pred]}')
        plt.savefig(os.path.join(dir_path, f'{str(ii).zfill(3)}.png'), bbox_inches='tight')

    print('Done!')

