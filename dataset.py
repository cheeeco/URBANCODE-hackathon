import os
import torch
import torchvision
import json

from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F

class SamoletDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root    # some kind of "/content/drive/My Drive/samolet/urbanhack-train/"
        self.transforms = transforms

        with open(os.path.join(root, "annotations/instances_default.json")) as json_data:
            data = json.load(json_data)

        self.images = data['images']

        self.boxes = [[] for _ in range(len(data['images']))]
        self.labels = [[] for _ in range(len(data['images']))]

        for d in data['annotations']:
            boxes_i = d['bbox']
            boxes_i = [boxes_i[0], boxes_i[1], boxes_i[0]+boxes_i[2], boxes_i[1]+boxes_i[3]]
            self.boxes[d['image_id']-1].append(boxes_i)
            self.labels[d['image_id']-1].append(d['category_id'])


    def __getitem__(self, idx):
        img_data = self.images[idx]
        img_path = os.path.join(self.root, "images/", img_data['file_name'])

        # load image tensor
        img = read_image(img_path)

        target = {}
        target['boxes'] = tv_tensors.BoundingBoxes(self.boxes[idx], format="XYXY", canvas_size=F.get_size(img))
        target['labels'] = torch.Tensor(self.labels[idx]).to(dtype=torch.long)
        target['image_id'] = torch.Tensor(img_data['id']).to(dtype=torch.long)

        if self.transforms is not None:
            # print(type(img), type(target))
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.images)
    


def collate_fn_padd(batch):
    max_w = 0
    max_h = 0

    for i in range(len(batch)):
        _, h, w = batch[i][0].shape
        if h > max_h:
            max_h = h
        if w > max_w:
            max_w = w

    padded_imgs = []
    target = []

    for i in range(len(batch)):
        # padding
        pad_y = max_h - batch[i][0].shape[1]
        pad_x = max_w - batch[i][0].shape[2]
        padded_image = torchvision.transforms.Pad((0, 0, pad_x, pad_y))(batch[i][0])
        padded_imgs.append(padded_image)

        # rearranging target
        boxes_t = batch[i][1]['boxes']
        labels_t = batch[i][1]['labels']
        target.append({'boxes': boxes_t, 'labels': labels_t, 'image_id': [batch[i][1]['image_id']]})

    return torch.stack(padded_imgs), target