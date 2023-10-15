from os.path import join, dirname
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from data.JigsawLoader import get_split_dataset_info, _dataset_info
from data.concat_dataset import ConcatDataset
from data.JigsawLoader import JigsawIADataset, JigsawTestIADataset, DistillCLIPDataset, DistillCLIPTestDataset

class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, limit):
        indices = torch.randperm(len(dataset))[:limit]
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def get_train_dataloader(args):

    dataset_list = args.source
    assert isinstance(dataset_list, list)
    datasets = []
    val_datasets = []
    img_transformer, tile_transformer = get_train_transformers(args)
    for dname in dataset_list:
        if args.dataset == 'PACS':
            name_train, labels_train = _dataset_info(join(dirname(__file__), 'data_path_txt_lists', '%s_train_kfold.txt' % dname))
            name_val, labels_val = _dataset_info(join(dirname(__file__), 'data_path_txt_lists', '%s_crossval_kfold.txt' % dname))
        else:
            name_train, name_val, labels_train, labels_val = get_split_dataset_info(
                join(dirname(__file__), 'data_path_txt_lists', '%s_train.txt' % dname), args.val_size)

        train_dataset = JigsawIADataset(name_train, labels_train, args.data_path, img_transformer=img_transformer)

        datasets.append(train_dataset)
        val_datasets.append(
            JigsawTestIADataset(name_val, labels_val, args.data_path, img_transformer=get_val_transformer(args)))
    dataset = ConcatDataset(datasets)
    val_dataset = ConcatDataset(val_datasets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    return loader, val_loader


def get_val_dataloader(args):
    if args.dataset == 'PACS':
        names, labels = _dataset_info(join(dirname(__file__), 'data_path_txt_lists', '%s_test_kfold.txt' % args.target))
    else:
        names, labels = _dataset_info(join(dirname(__file__), 'data_path_txt_lists', '%s_test.txt' % args.target))
    img_tr = get_val_transformer(args)
    val_dataset = JigsawTestIADataset(names, labels, args.data_path, img_transformer=img_tr)

    dataset = ConcatDataset([val_dataset])
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    return loader


def get_train_ems_dataloader(args, clip_transform):

    dataset_list = args.source
    assert isinstance(dataset_list, list)
    datasets = []
    val_datasets = []
    img_transformer, tile_transformer = get_train_transformers(args)
    for dname in dataset_list:
        if args.dataset == 'PACS':
            name_train, labels_train = _dataset_info(join(dirname(__file__), 'data_path_txt_lists', '%s_train_kfold.txt' % dname))
            name_val, labels_val = _dataset_info(join(dirname(__file__), 'data_path_txt_lists', '%s_crossval_kfold.txt' % dname))
        else:
            name_train, name_val, labels_train, labels_val = get_split_dataset_info(
                join(dirname(__file__), 'data_path_txt_lists', '%s_train.txt' % dname), args.val_size)

        train_dataset = DistillCLIPDataset(name_train, labels_train, args.data_path, img_transformer=img_transformer,
                                      clip_transformer=clip_transform)

        datasets.append(train_dataset)
        val_datasets.append(
            DistillCLIPTestDataset(name_val, labels_val, args.data_path, img_transformer=get_val_transformer(args),
                              clip_transformer=clip_transform))
    dataset = ConcatDataset(datasets)
    val_dataset = ConcatDataset(val_datasets)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    return loader, val_loader


def get_val_ems_dataloader(args, clip_transform):
    if args.dataset == 'PACS':
        names, labels = _dataset_info(join(dirname(__file__), 'data_path_txt_lists', '%s_test_kfold.txt' % args.target))
    else:
        names, labels = _dataset_info(join(dirname(__file__), 'data_path_txt_lists', '%s_test.txt' % args.target))
    img_tr = get_val_transformer(args)
    val_dataset = DistillCLIPTestDataset(names, labels, args.data_path, img_transformer=img_tr, clip_transformer=clip_transform)
    dataset = ConcatDataset([val_dataset])
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    return loader


def get_train_transformers(args):
    img_tr = [transforms.RandomResizedCrop((int(args.image_size), int(args.image_size)), (args.min_scale, args.max_scale))]
    if args.random_horiz_flip > 0.0:
        img_tr.append(transforms.RandomHorizontalFlip(args.random_horiz_flip))
    if args.jitter > 0.0:
        img_tr.append(transforms.ColorJitter(brightness=args.jitter, contrast=args.jitter, saturation=args.jitter, hue=min(0.5, args.jitter)))
    img_tr.append(transforms.RandomGrayscale(args.tile_random_grayscale))
    img_tr.append(transforms.ToTensor())
    img_tr.append(transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    tile_tr = []
    if args.tile_random_grayscale:
        tile_tr.append(transforms.RandomGrayscale(args.tile_random_grayscale))
    tile_tr = tile_tr + [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]

    return transforms.Compose(img_tr), transforms.Compose(tile_tr)


def get_val_transformer(args):
    img_tr = [transforms.Resize((args.image_size, args.image_size)), transforms.ToTensor(),
              transforms.Normalize([0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
    return transforms.Compose(img_tr)
