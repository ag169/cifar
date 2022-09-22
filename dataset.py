import torchvision
from torch.utils.data.dataloader import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import pickle


def get_cifar10_loaders(batchsize=32):
    train_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
        ]
    )

    test_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
        ]
    )

    train_dataset = torchvision.datasets.CIFAR10(root='./', train=True, transform=train_transforms, download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='./', train=False, transform=test_transforms, download=True)

    train_loader_ = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=4, pin_memory=True,
                               drop_last=False)
    test_loader_ = DataLoader(test_dataset, batch_size=batchsize, shuffle=False, num_workers=4, pin_memory=False,
                              drop_last=False)

    # Load metadata from downloaded dataset
    with open('cifar-10-batches-py/batches.meta', 'rb') as fp:
        cifar_metadata_ = pickle.load(fp)

    return {
        'train_loader': train_loader_,
        'test_loader': test_loader_,
        'metadata': cifar_metadata_,
    }


# Main function to test dataset-load and visualize images
if __name__ == '__main__':
    NUM_IMAGES_PER_LABEL = 8

    loader_dict = get_cifar10_loaders()

    train_loader = loader_dict['train_loader']
    train_ds = train_loader.dataset

    train_data = train_ds.data
    train_labels = np.array(train_ds.targets, dtype=np.int32)
    inds_arr = np.arange(train_labels.shape[0])

    label_set = np.sort(np.unique(train_labels))

    labelwise_inds = [inds_arr[train_labels == x] for x in label_set]

    fig, axes = plt.subplots(nrows=len(label_set), ncols=NUM_IMAGES_PER_LABEL, sharey=True, sharex=True,
                             figsize=(1.2 * len(label_set), 1.2 * NUM_IMAGES_PER_LABEL))

    label_names = loader_dict['metadata']['label_names']

    for ii, (label, data_inds) in enumerate(zip(label_set, labelwise_inds)):
        np.random.shuffle(data_inds)
        data_inds = data_inds[:NUM_IMAGES_PER_LABEL]
        data = [train_data[x] for x in data_inds]

        for ij, img in enumerate(data):
            axes[ii, ij].imshow(img)
            axes[ii, ij].get_xaxis().set_visible(False)
            axes[ii, ij].get_yaxis().set_visible(False)

        axes[ii, 0].set_ylabel(label_names[ii])
        axes[ii, 0].get_yaxis().set_visible(True)
        for key, spine in axes[ii, 0].spines.items():
            spine.set_visible(False)

    plt.yticks([])
    plt.xticks([])
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig('./train_ds_samples.png', bbox_inches='tight')

    print('Done')

