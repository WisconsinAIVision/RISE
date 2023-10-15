import torch.utils.data as data
from PIL import Image
from random import sample
from PIL import ImageFile


def get_random_subset(names, labels, percent):
    """

    :param names: list of names
    :param labels:  list of labels
    :param percent: 0 < float < 1
    :return:
    """
    samples = len(names)
    amount = int(samples * percent)
    random_index = sample(range(samples), amount)
    name_val = [names[k] for k in random_index]
    name_train = [v for k, v in enumerate(names) if k not in random_index]
    labels_val = [labels[k] for k in random_index]
    labels_train = [v for k, v in enumerate(labels) if k not in random_index]
    return name_train, name_val, labels_train, labels_val


def _dataset_info(txt_labels):
    with open(txt_labels, 'r') as f:
        images_list = f.readlines()

    file_names = []
    labels = []
    for row in images_list:
        row = row.split(' ')
        file_names.append(row[0])
        labels.append(int(row[1]))

    return file_names, labels


def get_split_dataset_info(txt_list, val_percentage):
    names, labels = _dataset_info(txt_list)
    return get_random_subset(names, labels, val_percentage)


class JigsawIADataset(data.Dataset):
    def __init__(self, names, labels, data_path, img_transformer=None):
        self.data_path = data_path
        self.names = names
        self.labels = labels
        self._image_transformer = img_transformer

    def get_image(self, index):
        framename = self.data_path + '/' + self.names[index]
        img = Image.open(framename).convert('RGB')
        return self._image_transformer(img)

    def __getitem__(self, index):
        framename = self.data_path + '/' + self.names[index]
        img = Image.open(framename).convert('RGB')
        return self._image_transformer(img), int(self.labels[index] - 1)

    def __len__(self):
        return len(self.names)

class JigsawTestIADataset(JigsawIADataset):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)

    def __getitem__(self, index):
        framename = self.data_path + '/' + self.names[index]
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img = Image.open(framename).convert('RGB')
        return self._image_transformer(img), int(self.labels[index] - 1)


class DistillCLIPDataset(data.Dataset):
    def __init__(self, names, labels, data_path, img_transformer=None, clip_transformer=None):
        self.data_path = data_path
        self.names = names
        self.labels = labels
        self._image_transformer = img_transformer
        self._CLIP_transformer = clip_transformer

    def get_image(self, index):
        framename = self.data_path + '/' + self.names[index]
        img = Image.open(framename).convert('RGB')
        return self._image_transformer(img)

    def __getitem__(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        framename = self.data_path + '/' + self.names[index]
        img = Image.open(framename).convert('RGB')
        return self._image_transformer(img), self._CLIP_transformer(img), int(self.labels[index] - 1)


    def __len__(self):
        return len(self.names)

class DistillCLIPTestDataset(DistillCLIPDataset):
    def __init__(self, *args, **xargs):
        super().__init__(*args, **xargs)

    def __getitem__(self, index):
        framename = self.data_path + '/' + self.names[index]
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        img = Image.open(framename).convert('RGB')
        return self._image_transformer(img), self._CLIP_transformer(img), int(self.labels[index] - 1)
