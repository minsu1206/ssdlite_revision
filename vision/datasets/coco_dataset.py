import numpy as np
import logging
import os
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
import cv2
import matplotlib.pyplot as plt
import skimage.io as io
import random
import torch
import torch.nn.functional as F


def dir_coco(ann_dir, mode_num=0, task_num=1):
    tasks = ['captions_', 'instances_', 'person_keypoints_']
    modes = ['train', 'val', 'test']
    task = tasks[task_num]
    mode = modes[mode_num]
    json_name = task + mode + '2017.json'
    json_dir = os.path.join(ann_dir, json_name)
    coco = COCO(json_dir)
    return coco


class CustomCOCO(Dataset):
    def __init__(self,
                 ann_root: str,
                 transform=None,
                 target_transform=None,
                 mode=0):
        self.batch_count = 0
        self.root = ann_root
        self.transform = transform
        self.target_transform = target_transform
        # self.data_local = img_root if type(img_root) == 'str' else False
        self.coco = dir_coco(self.root, mode_num=mode)
        cats = self.coco.loadCats(self.coco.getCatIds())
        nms = [cat['name'] for cat in cats]
        self.class_names = tuple(nms)
        self.num_classes = len(nms)

        # print('COCO 2017 categories: {} classes \n{}\n'.format(self.num_classes,
        #                                                        ' '.join(nms)))

        with open(os.path.join(os.getcwd(), 'coco-labels-paper.txt'), 'r') as f:
            lines = f.readlines()
        self.class_dict = {i+1: class_name.split('\n')[0] for i, class_name in enumerate(lines)}
        self.class_dict_inv = {class_name.split('\n')[0]: i+1 for i, class_name in enumerate(lines)}

        with open(os.path.join(os.getcwd(), 'coco-labels-2014_2017.txt'), 'r') as f:
            lines = f.readlines()

        self.old2new = {}
        self.new_dict = {}
        for i in range(len(lines)):
            line = lines[i]
            cls = line.split('\n')[0]
            self.old2new[self.class_dict_inv[cls]] = i
            self.new_dict[i] = cls

        self.ids = list(sorted(self.coco.imgs.keys()))
        logging.info("Read COCO Dataset")

        self.multiscale = True
        self.img_size = 416
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __getitem__(self, idx):
        self.batch_count += 1
        img_id = self.ids[idx]
        path = self.coco.loadImgs(img_id)[0]["coco_url"]
        img = io.imread(path)
        target = self.coco.loadAnns(self.coco.getAnnIds(img_id))
        classes = []
        bboxes = []
        for target_ in target:
            classes.append(self.old2new[target_['category_id']])
            bboxes.append(target_['bbox'])
        boxes = np.array(bboxes, dtype=np.float32)
        classes = np.array(classes, dtype=np.int64).reshape(len(boxes), 1)
        try:
            if self.transform:
                img, boxes, classes = self.transform(img, boxes, classes)
            if self.target_transform:
                boxes, classes = self.target_transform(boxes, classes)
        # bbox = np.concatenate((classes, boxes), axis=1)
        # print('original', 'id=', img_id, img.shape, '\n', bbox.shape)

        # """
        # bounding box = [x,y,w,h]
        #     Note that (x,y) of COCO dataset stands for box' top left coordinate, Not center of box
        #
        # We choose augmentation method from pytorch-yolov3's method
        # Please note that bbox = (N, 6) after iaa augmentation
        #     bbox = [sample index, label, x, y, w, h]
        # """
        except Exception:
            return

        return img, boxes, classes

    def __len__(self):
        return len(self.ids)

    def collate_fn(self, batch):
        self.batch_count += 1

        # Drop invalid images
        batch = [data for data in batch if data is not None]
        # for b_ in batch[0]:
        #     print(b_.shape)       # (3,300,300) : img.shape , (8732, 4) : box coordinate , (8732, 1) : classification
        # paths, imgs, bb_targets = list(zip(*batch))
        imgs, bb_targets, classes = list(zip(*batch))
        print('Collate', len(classes) == len(imgs) == len(bb_targets))
        # Selects new image size every 10 batch
        if self.multiscale and self.batch_count % 40 == 0:
            self.img_size = random.choice(
                range(self.min_size, self.max_size + 1, 32))

        ## Resize images to input shape
        # imgs = torch.stack([resize(img, self.img_size) for img in imgs])

        # # Add sample index to targets
        # for i, boxes in enumerate(bb_targets):
        #     boxes[:, 0] = i
        # bb_targets = torch.cat(bb_targets, 0)
        # classes = torch.cat(classes, 0)
        imgs = torch.stack(imgs)
        bb_targets = torch.stack(bb_targets)
        classes = torch.stack(classes)
        print('After Collate', imgs.shape, bb_targets.shape, classes.shape)
        # return paths, imgs, bb_targets
        return imgs, bb_targets, classes

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def box_draw(img, boxes, classes, names=None):
    colors = [(255, 0, 0), (255, 127, 0), (255, 255, 0), (0, 255, 0),
              (0, 0, 255), (46, 43, 95), (139, 0, 255)]

    for i, (box, class_) in enumerate(zip(boxes, classes)):
        tx, ty = box[0], box[1]
        bx, by = box[0] + box[2], box[1] + box[3]
        tx, ty = round(tx), round(ty)
        bx, by = round(bx), round(by)
        img = cv2.rectangle(img, pt1=(tx, ty), pt2=(bx, by), color=colors[i % 7], thickness=2)
        if names is not None:
            name = names[class_]
        else:
            name = str(class_)
        img = cv2.putText(img, name, (tx + 3, ty + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color=colors[i % 7],
                          thickness=2)
    plt.imshow(img)
    plt.show()





if __name__ == '__main__':
    custom = CustomCOCO(
        ann_root=r'C:\Users\82106\PycharmProjects\dino_lib\extra_project\pytorch-ssd\data'
    )   # No transform
    img, boxes, classes = custom[1]
    names = custom.new_dict
    print(names)
    box_draw(img, boxes, list(classes.reshape(-1, )), names)

    # loader = DataLoader(custom, batch_size=1,
    #                     num_workers=0,
    #                     shuffle=False,
    #                     collate_fn=custom.collate_fn)
    # for i, data in enumerate(loader):
    #     img_, box_, label_ = data
    #     print(img_)
    #     print(box_)
    #     print(label_)
    #     if i == 2:
    #         break

