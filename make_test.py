import torchvision
import os
import shutil
import random
from PIL import Image
from tqdm import tqdm

if __name__ == '__main__':
    NUM_IMGS_PER_CLASS = 100

    tgt_dir = os.path.join('cifar-10-batches-py', 'test_ds')

    if os.path.isdir(tgt_dir):
        shutil.rmtree(tgt_dir)

    os.makedirs(tgt_dir)

    test_dataset = torchvision.datasets.CIFAR10(root='./', train=False, transform=None, download=True)

    test_data = test_dataset.data

    class_names = test_dataset.classes
    test_labels = list(enumerate(test_dataset.targets))

    classes = list(test_dataset.class_to_idx.values())

    test_idxs_selected = list()

    for class_idx in classes:
        idxs_for_class = [x for x in test_labels if x[1] == class_idx]
        random.shuffle(idxs_for_class)

        test_idxs_selected.extend(idxs_for_class[:NUM_IMGS_PER_CLASS])

    test_idxs_selected.sort()

    for ii, label_idx in tqdm(test_idxs_selected):
        img = Image.fromarray(test_data[ii])

        img.save(os.path.join(tgt_dir, f'{str(ii).zfill(6)}_{label_idx}.png'))

        # print(class_names[label_idx])

    print('Done')
