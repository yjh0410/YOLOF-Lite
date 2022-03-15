import random
import numpy as np
import cv2
import torch
import torchvision.transforms.functional as F



class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target=None):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


# Convert ndarray to tensor
class ToTensor(object):
    def __init__(self, format='RGB'):
        self.format = format

    def __call__(self, image, target=None):
        # to rgb
        if self.format == 'RGB':
            image = image[..., (2, 1, 0)]
        elif self.format == 'BGR':
            image = image
        else:
            print('Unknown color format !!')
            exit()
        image = F.to_tensor(image)
        if target is not None:
            target["boxes"] = torch.as_tensor(target["boxes"]).float()
            target["labels"] = torch.as_tensor(target["labels"]).long()
        return image, target


# RandomSizeCrop
class RandomSizeCrop(object):
    def __init__(self):
        self.sample_options = (
            # use entile image
            None,
            # sample a patch s.t. MIN jaccard w/ obj in .1,.3,.4,.7,.9
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            # randomly sample a patch
            (None, None),
        )


    def jaccard_numpy(self, box_a, box_b):
        """Compute the jaccard overlap of two sets of boxes"""
        max_xy = np.minimum(box_a[:, 2:], box_b[2:])
        min_xy = np.maximum(box_a[:, :2], box_b[:2])
        inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
        inter = inter[:, 0] * inter[:, 1]

        area_a = ((box_a[:, 2]-box_a[:, 0]) *
                (box_a[:, 3]-box_a[:, 1]))  # [A,B]
        area_b = ((box_b[2]-box_b[0]) *
                (box_b[3]-box_b[1]))  # [A,B]
        union = area_a + area_b - inter
        
        return inter / union  # [A,B]


    def __call__(self, image, target):
        height, width, _ = image.shape
        while True:
            # randomly choose a mode
            sample_id = np.random.randint(len(self.sample_options))
            mode = self.sample_options[sample_id]
            if mode is None:
                return image, target

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            boxes = target['boxes']
            labels = target['labels']
            # max trails (50)
            for _ in range(50):
                current_image = image

                w = random.uniform(0.3 * width, width)
                h = random.uniform(0.3 * height, height)

                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue

                left = random.uniform(0, width - w)
                top = random.uniform(0, height - h)

                # convert to integer rect x1,y1,x2,y2
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])

                # calculate IoU (jaccard overlap) b/t the cropped and gt boxes
                overlap = self.jaccard_numpy(boxes, rect)

                # is min and max overlap constraint satisfied? if not try again
                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                # cut the crop from the image
                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2],
                                            :]

                # keep overlap with gt box IF center in sampled patch
                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0

                # mask in all gt boxes that above and to the left of centers
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])

                # mask in all gt boxes that under and to the right of centers
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])

                # mask in that both m1 and m2 are true
                mask = m1 * m2

                # have any valid boxes? try again if not
                if not mask.any():
                    continue

                # take only matching gt boxes
                current_boxes = boxes[mask, :].copy()

                # take only matching gt labels
                current_labels = labels[mask]

                # should we use the box left and top corner or the crop's
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2],
                                                rect[:2])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, :2] -= rect[:2]

                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:],
                                                rect[2:])
                # adjust to crop (by substracting crop's left,top)
                current_boxes[:, 2:] -= rect[:2]

                target['boxes'] = current_boxes
                target['labels'] = current_labels

                return current_image, target


# RandomHFlip
class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target=None):
        if random.random() < self.p:
            image = image[:, ::-1]
            if target is not None:
                h, w = target["orig_size"]
                if "boxes" in target:
                    boxes = target["boxes"].copy()
                    boxes[..., [0, 2]] = w - boxes[..., [2, 0]]
                    target["boxes"] = boxes

        return image, target


# RandomShift
class RandomShift(object):
    def __init__(self, p=0.5, max_shift=32):
        self.p = p
        self.max_shift = max_shift

    def __call__(self, image, target=None):
        if random.random() < self.p:
            shift_x = random.randint(-self.max_shift, self.max_shift)
            shift_y = random.randint(-self.max_shift, self.max_shift)
            if shift_x < 0:
                new_x = 0
                orig_x = -shift_x
            else:
                new_x = shift_x
                orig_x = 0
            if shift_y < 0:
                new_y = 0
                orig_y = -shift_y
            else:
                new_y = shift_y
                orig_y = 0
            new_image = np.zeros_like(image)
            img_h, img_w = image.shape[:-1]
            new_h = img_h - abs(shift_y)
            new_w = img_w - abs(shift_x)
            new_image[new_y:new_y + new_h, new_x:new_x + new_w, :] = image[
                                                                orig_y:orig_y + new_h,
                                                                orig_x:orig_x + new_w, :]
            boxes_ = target["boxes"].copy()
            boxes_[..., [0, 2]] += shift_x
            boxes_[..., [1, 3]] += shift_y
            boxes_[..., [0, 2]] = boxes_[..., [0, 2]].clip(0, img_w)
            boxes_[..., [1, 3]] = boxes_[..., [1, 3]].clip(0, img_h)
            target["boxes"] = boxes_

            return new_image, target

        return image, target


# DistortTransform
class DistortTransform(object):
    """
    Distort image.
    """
    def __call__(self, image, target=None):
        """
        Args:
            img (ndarray): of shape HxWxC. The Tensor is floating point in range[0, 255].

        Returns:
            ndarray: the distorted image(s).
        """
        def _convert(image, alpha=1, beta=0):
            tmp = image.astype(float) * alpha + beta
            tmp[tmp < 0] = 0
            tmp[tmp > 255] = 255
            image[:] = tmp

        image = image.copy()

        _convert(image, beta=random.uniform(-32, 32))
        _convert(image, alpha=random.uniform(0.5, 1.5))
        # BGR -> HSV
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
        tmp %= 180
        image[:, :, 0] = tmp
        _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))
        # HSV -> BGR
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        return image, target


# Normalize tensor image
class Normalize(object):
    def __init__(self, pixel_mean, pixel_std):
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std

    def __call__(self, image, target=None):
        # normalize image
        image = F.normalize(image, 
                            mean=self.pixel_mean, 
                            std=self.pixel_std)
        # normalize bboxes
        if target is not None:
            boxes_ = target['boxes'].clone()
            img_h, img_w = image.shape[1:]
            boxes_[..., [0, 2]] /= img_w
            boxes_[..., [1, 3]] /= img_h
            target['boxes'] = boxes_

        return image, target


# Resize tensor image
class Resize(object):
    def __init__(self, img_size=800):
        self.img_size = img_size

    def rescale_targets(self, target, origin_size, output_size):
        img_h0, img_w0 = origin_size
        img_h, img_w = output_size
        # rescale bbox
        boxes_ = target["boxes"].copy()
        boxes_[:, [0, 2]] = boxes_[:, [0, 2]] / img_w0 * img_w
        boxes_[:, [1, 3]] = boxes_[:, [1, 3]] / img_h0 * img_h
        target["boxes"] = boxes_

        return target


    def __call__(self, image, target=None):
        origin_size = image.shape[:-1]
        # resize
        image = cv2.resize(image, (self.img_size, self.img_size))

        if target is not None:
            output_size = image.shape[:-1]
            # rescale bbox
            target = self.rescale_targets(target, origin_size, output_size)

        return image, target


# Pad tensor image
class PadImage(object):
    def __init__(self, img_size=800) -> None:
        self.img_size = img_size

    def __call__(self, image, target=None):
        img_h0, img_w0 = image.shape[1:]
        pad_image = torch.zeros([image.size(0), self.img_size, self.img_size]).float()
        pad_image[:, :img_h0, :img_w0] = image

        return pad_image, target


# BaseTransforms
class BaseTransforms(object):
    def __init__(self, 
                 img_size=800, 
                 pixel_mean=(0.485, 0.456, 0.406), 
                 pixel_std=(0.229, 0.224, 0.225), 
                 format='RGB'):
        self.img_size = img_size
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.transforms = Compose([
            DistortTransform(),
            RandomHorizontalFlip(),
            Resize(img_size=img_size),
            ToTensor(format=format),
            Normalize(pixel_mean, pixel_std),
            PadImage(img_size=img_size)
        ])

    def __call__(self, image, target):
        return self.transforms(image, target)


# TrainTransform
class TrainTransforms(object):
    def __init__(self, 
                 img_size=800,
                 trans_config=None,
                 pixel_mean=(0.485, 0.456, 0.406), 
                 pixel_std=(0.229, 0.224, 0.225), 
                 format='RGB'):
        self.img_size = img_size
        self.trans_config = trans_config
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.format = format
        self.transforms = Compose(self.build_transforms(trans_config))


    def build_transforms(self, trans_config):
        transform = []
        for t in trans_config:
            if t['name'] == 'DistortTransform':
                transform.append(DistortTransform())
            elif t['name'] == 'RandomHorizontalFlip':
                transform.append(RandomHorizontalFlip())
            elif t['name'] == 'RandomShift':
                transform.append(RandomShift(max_shift=t['max_shift']))
            elif t['name'] == 'RandomSizeCrop':
                transform.append(RandomSizeCrop())
            elif t['name'] == 'Resize':
                transform.append(Resize(img_size=self.img_size))
            elif t['name'] == 'ToTensor':
                transform.append(ToTensor(format=self.format))
            elif t['name'] == 'Normalize':
                transform.append(Normalize(pixel_mean=self.pixel_mean,
                                           pixel_std=self.pixel_std))
            elif t['name'] == 'PadImage':
                transform.append(PadImage(img_size=self.img_size))
        
        return transform


    def __call__(self, image, target):
        return self.transforms(image, target)


# ValTransform
class ValTransforms(object):
    def __init__(self, 
                 img_size=800, 
                 pixel_mean=(0.485, 0.456, 0.406), 
                 pixel_std=(0.229, 0.224, 0.225),
                 format='RGB'):
        self.img_size = img_size
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std
        self.transforms = Compose([
            Resize(img_size=img_size),
            ToTensor(format),
            Normalize(pixel_mean, pixel_std)
        ])


    def __call__(self, image, target=None):
        return self.transforms(image, target)
