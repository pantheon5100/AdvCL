import torch
from dataset import CIFAR10IndexPseudoLabelEnsemble
import pickle
import torchvision.transforms as transforms
from utils import progress_bar, TwoCropTransformAdv
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets


def get_data_loader(args):
    print('=====> Preparing data...')
    # Multi-cuda
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        batch_size = args.batch_size

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
        ])
    train_transform_org = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
    transform_train = TwoCropTransformAdv(transform_train, train_transform_org)

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
    ])

    val_dataset = datasets.CIFAR10(root='data',
                                   train=False,
                                   transform=val_transform,
                                   download=True)
    
    # import ipdb; ipdb.set_trace()

    val_dataset.data = val_dataset.data[:256*8]
    val_dataset.targets = val_dataset.targets[:256*8]

    val_loader = torch.utils.data.DataLoader(
                val_dataset, batch_size=256, shuffle=False,
                num_workers=8, pin_memory=True)

    label_pseudo_train_list = []
    num_classes_list = [2, 10, 50, 100, 500]

    dict_name = 'data/{}_pseudo_labels.pkl'.format(args.cname)
    f = open(dict_name, 'rb')  # Pickle file is newly created where foo1.py is
    feat_label_dict = pickle.load(f)  # dump data to f
    f.close()
    for i in range(5):
        class_num = num_classes_list[i]
        key_train = 'pseudo_train_{}'.format(class_num)
        label_pseudo_train = feat_label_dict[key_train]
        label_pseudo_train_list.append(label_pseudo_train)

    train_dataset = CIFAR10IndexPseudoLabelEnsemble(root='data',
                                                    transform=transform_train,
                                                    pseudoLabel_002=label_pseudo_train_list[0],
                                                    pseudoLabel_010=label_pseudo_train_list[1],
                                                    pseudoLabel_050=label_pseudo_train_list[2],
                                                    pseudoLabel_100=label_pseudo_train_list[3],
                                                    pseudoLabel_500=label_pseudo_train_list[4],
                                                    download=True)
    # Data Loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=n_gpu*4)
    return train_loader, val_loader