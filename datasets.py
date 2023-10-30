import itertools
import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler


class RandomTranslateWithReflect:
    """Translate image randomly
    Translate vertically and horizontally by n pixels where
    n is integer drawn uniformly independently for each axis
    from [-max_translation, max_translation].
    Fill the uncovered blank area with reflect padding.
    """

    def __init__(self, max_translation):
        self.max_translation = max_translation

    def __call__(self, old_image):
        xtranslation, ytranslation = np.random.randint(-self.max_translation,
                                                       self.max_translation + 1,
                                                       size=2)
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = old_image.size

        flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = old_image.transpose(Image.ROTATE_180)

        new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad))

        new_image.paste(old_image, (xpad, ypad))

        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))

        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))

        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_image = new_image.crop((xpad - xtranslation,
                                    ypad - ytranslation,
                                    xpad + xsize - xtranslation,
                                    ypad + ysize - ytranslation))

        return new_image


class LocalImageDataset(Dataset):

    def __init__(self, img_labels, img_dir, transform, label_dict, p_labels=None):
        self.img_labels = img_labels
        self.img_dir = img_dir
        self.transform = transform
        self.label_dict = label_dict

        if p_labels is None:
            self.p_labels = -1 * np.ones(len(self.img_labels)).astype(int)
        else:
            self.p_labels = np.array(p_labels).astype(int)
        self.p_weights = np.ones(len(self.img_labels))
        self.class_weights = np.ones(len(label_dict))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_filename, label = self.img_labels[idx]
        img_path = os.path.join(self.img_dir, label, img_filename)

        with open(img_path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        if self.transform:
            img = self.transform(img)

        img_plabel = self.p_labels[idx]

        return img, self.label_dict[label], img_plabel, self.p_weights[idx], self.class_weights[int(img_plabel)]

    def update_plabels(self, p_labels, p_weights, class_weights):
        assert len(p_labels) == len(self.p_labels)
        assert len(p_weights) == len(self.p_weights)
        assert len(class_weights) == len(self.class_weights)
        self.p_labels = p_labels
        self.p_weights = p_weights
        self.class_weights = class_weights


class DoubleDataset(Dataset):

    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.n1 = len(dataset1)
        self.n2 = len(dataset2)

    def __len__(self):
        return self.n1 + self.n2

    def __getitem__(self, idx):
        if idx < self.n1:
            return self.dataset1.__getitem__(idx)
        else:
            return self.dataset2.__getitem__(idx - self.n1)

    def update_plabels(self, p_labels, p_weights, class_weights):
        self.dataset1.update_plabels(p_labels[:self.n1], p_weights[:self.n1], class_weights)
        self.dataset2.update_plabels(p_labels[self.n1:], p_weights[self.n1:], class_weights)


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in  zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
