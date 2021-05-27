import numbers
import random

import math
import numpy as np
import torch
from PIL import Image, ImageOps, ImageEnhance


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size  # h, w
        self.padding = padding

    def __call__(self, sample):
        image, mask = sample['image'], sample['label']

        if self.padding > 0:
            if isinstance(image, list):
                new_image = []
                for item in image:
                    item = ImageOps.expand(item, border=self.padding, fill=0)
                    new_image.append(item)
                image = new_image
            else:
                image = ImageOps.expand(image, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)

        w, h = mask.size
        th, tw = self.size  # target size
        if w == tw and h == th:
            return {'image': image,
                    'label': mask}
        if w < tw or h < th:
            if isinstance(image, list):
                new_image = []
                for item in image:
                    assert item.size == mask.size
                    item = item.resize((tw, th), Image.BILINEAR)
                    new_image.append(item)
                image = new_image
            else:
                assert image.size == mask.size
                image = image.resize((tw, th), Image.BILINEAR)
            mask = mask.resize((tw, th), Image.NEAREST)
            return {'image': image,
                    'label': mask}

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        if isinstance(image, list):
            new_image = []
            for item in image:
                item = item.crop((x1, y1, x1 + tw, y1 + th))
                new_image.append(item)
            image = new_image
        else:
            image = image.crop((x1, y1, x1 + tw, y1 + th))
        mask = mask.crop((x1, y1, x1 + tw, y1 + th))

        return {'image': image,
                'label': mask}


class RandomSized(object):
    def __init__(self, size, scale_min, scale_max):
        self.size = size
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.padding = Padding(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, sample):
        image = sample['image']
        mask = sample['label']

        scale = random.uniform(self.scale_min, self.scale_max)

        w = int(scale * mask.size[0])
        h = int(scale * mask.size[1])

        if isinstance(image, list):
            new_image = []
            for item in image:
                assert item.size == mask.size
                item = item.resize((w, h), Image.BILINEAR)
                new_image.append(item)
            image = new_image
        else:
            assert image.size == mask.size
            image = image.resize((w, h), Image.BILINEAR)
        mask = mask.resize((w, h), Image.NEAREST)
        sample = {'image': image, 'label': mask}

        padded = self.padding(sample)
        cropped = self.crop(padded)
        return cropped


class FixedResize(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, sample):
        image = sample['image']
        mask = sample['label']

        assert image.size == mask.size

        image = image.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': image,
                'label': mask}


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        image = sample['image']
        mask = sample['label']

        w, h = mask.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        if isinstance(sample['image'], list):
            new_image = []
            for item in image:
                assert item.size == mask.size
                item = item.crop((x1, y1, x1 + tw, y1 + th))
                new_image.append(item)
            image = new_image
        else:
            assert image.size == mask.size
            image = image.crop((x1, y1, x1 + tw, y1 + th))
        mask = mask.crop((x1, y1, x1 + tw, y1 + th))

        return {'image': image,
                'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        image = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            if isinstance(image, list):
                new_image = []
                for item in image:
                    item = item.transpose(Image.FLIP_LEFT_RIGHT)
                    new_image.append(item)
                image = new_image
            else:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': image,
                'label': mask}


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """

    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        image = np.array(sample['image']).astype(np.float32)
        mask = np.array(sample['label']).astype(np.float32)
        image /= 255.0
        image -= self.mean
        image /= self.std

        return {'image': image,
                'label': mask}


class ToTensorRGB(object):
    """Convert cv2 ndarrays in sample to Tensors."""

    def __init__(self):
        pass

    def __call__(self, sample):
        # swap color axis because
        # cv2 numpy image: H x W x C
        # torch image: C X H x W
        image = sample['image']
        mask = sample['label']

        if isinstance(image, list):
            new_image = []
            for i in image:
                image = np.array(i).astype(np.float32).transpose((2, 0, 1))
                image = torch.from_numpy(image).float()
                new_image.append(image)
            image = new_image
        else:
            image = np.array(image).astype(np.float32).transpose((2, 0, 1))
            image = torch.from_numpy(image).float()

        mask = np.array(mask).astype(np.float32)
        mask = torch.from_numpy(mask).long()

        return {'image': image,
                'label': mask}


class ToTensor(object):
    """Convert cv2 ndarrays in sample to Tensors."""

    def __init__(self):
        pass

    def __call__(self, sample):
        # swap color axis because
        # cv2 numpy image: H x W x C
        # torch image: C X H X W
        image = sample['image']
        mask = sample['label']

        if isinstance(image, list):
            new_image = []
            for item in image:
                item = np.expand_dims(np.array(item).astype(np.float32), -1).transpose((2, 0, 1))
                item = torch.from_numpy(item).float()
                new_image.append(item)
            image = torch.stack(new_image, dim=0)
        else:
            image = np.expand_dims(np.array(image).astype(np.float32), -1).transpose((2, 0, 1))
            image = torch.from_numpy(image).float()

        mask = np.array(mask).astype(np.float32)
        mask = torch.from_numpy(mask).long()

        return {'image': image,
                'label': mask}


class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        image = sample['image']
        mask = sample['label']
        assert image.size == mask.size
        for attempt in range(10):
            area = image.size[0] * image.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= image.size[0] and h <= image.size[1]:
                x1 = random.randint(0, image.size[0] - w)
                y1 = random.randint(0, image.size[1] - h)

                image = image.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert (image.size == (w, h))

                image = image.resize((self.size, self.size), Image.BILINEAR)
                mask = mask.resize((self.size, self.size), Image.NEAREST)

                return {'image': image,
                        'label': mask}

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        sample = crop(scale(sample))
        return sample


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        image = sample['image']
        mask = sample['label']
        rotate_degree = random.random() * 2 * self.degree - self.degree

        if isinstance(image, list):
            new_image = []
            for item in image:
                item = item.rotate(rotate_degree, Image.BILINEAR)
                new_image.append(item)
            image = new_image
        else:
            image = image.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': image,
                'label': mask}


class Scale(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        image = sample['image']
        mask = sample['label']
        assert image.size == mask.size
        w, h = image.size
        if (w >= h and w == self.size[1]) or (h >= w and h == self.size[0]):
            return {'image': image,
                    'label': mask}
        oh, ow = self.size
        image = image.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)

        return {'image': image,
                'label': mask}


class RandomScale(object):
    def __init__(self, limit):
        self.limit = limit

    def __call__(self, sample):
        image = sample['image']
        mask = sample['label']

        scale = random.uniform(self.limit[0], self.limit[1])
        w = int(scale * image.size[0])
        h = int(scale * image.size[1])

        if isinstance(image, list):
            new_image = []
            for item in image:
                item = item.resize((w, h), Image.BILINEAR)
                new_image.append(item)
            image = new_image
        else:
            image = image.resize((w, h), Image.BILINEAR)
        mask = mask.resize((w, h), Image.NEAREST)

        return {'image': image, 'label': mask}


class Padding(object):
    """padding zero to image to match the maximum value of target width or height"""

    def __init__(self, size, fill=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.fill = fill

    def __call__(self, sample):
        image = sample['image']
        mask = sample['label']

        w, h = mask.size
        target_h, target_w = self.size

        left = top = right = bottom = 0
        doit = False
        if target_w > w:
            delta = target_w - w
            left = delta // 2
            right = delta - left
            doit = True

        if target_h > h:
            delta = target_h - h
            top = delta // 2
            bottom = delta - top
            doit = True
        if doit:
            if isinstance(image, list):
                new_image = []
                for item in image:
                    item = ImageOps.expand(item, border=(left, top, right, bottom), fill=self.fill)
                    new_image.append(item)
                image = new_image
            else:
                image = ImageOps.expand(image, border=(left, top, right, bottom), fill=self.fill)
            mask = ImageOps.expand(mask, border=(left, top, right, bottom), fill=self.fill)

        return {'image': image,
                'label': mask}


class FilterAndNormalizeByWL(object):
    def __init__(self, win_width, win_loc):
        self.win_width = win_width
        self.win_loc = win_loc

        self.HU_min = int(win_loc - win_width / 2.0 + 0.5)
        self.HU_max = int(win_loc + win_width / 2.0 + 0.5)

    def __call__(self, sample):
        image = sample["image"]
        mask = sample["label"]

        if isinstance(image, list):
            new_image = []
            for item in image:
                np.place(item, item > self.HU_max, self.HU_max)
                np.place(item, item < self.HU_min, self.HU_min)
                item = (np.array(item) - float(self.HU_min)) / (self.HU_max - self.HU_min)
                new_image.append(item)
            image = new_image
        else:
            np.place(image, image > self.HU_max, self.HU_max)
            np.place(image, image < self.HU_min, self.HU_min)
            image = (np.array(image) - float(self.HU_min)) / (self.HU_max - self.HU_min)

        return {'image': image,
                'label': mask}


class Arr2Image(object):
    def __call__(self, sample):
        image = sample["image"]
        mask = sample["label"]

        if isinstance(image, list):
            new_image = []
            for item in image:
                item = Image.fromarray(item)
                new_image.append(item)
            image = new_image
        else:
            image = Image.fromarray(image)

        mask = Image.fromarray(mask)

        return {'image': image,
                'label': mask}


class RandomColor(object):
    def __init__(self, rate):
        self.rate = rate

    def __call__(self, sample):
        image = sample['image']
        mask = sample['label']

        random_rate = random.uniform(1 - self.rate, 1 + self.rate)

        if isinstance(image, list):
            new_image = []
            for item in image:
                item = ImageEnhance.Color(item).enhance(random_rate)
                new_image.append(item)
            image = new_image
        else:
            image = ImageEnhance.Color(image).enhance(random_rate)

        return {'image': image, 'label': mask}
