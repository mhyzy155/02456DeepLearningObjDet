import os
import numpy as np
import torch
import xml.etree.ElementTree as ET
from PIL import Image


class ColaBeerDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.xmls = list(sorted(os.listdir(os.path.join(root, "frames"))))

    def __getitem__(self, idx):
        # load images and xmls
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        xml_path = os.path.join(self.root, "frames", self.xmls[idx])
        img = Image.open(img_path).convert("RGB")

        
        #Get boundary boxes and labels
        target = self.__xml_to_dict(xml_path)
        image_id = torch.tensor([idx])
        #Add image_id to target dictionary
        target["image_id"] = image_id

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __xml_to_dict(self, xml_path):
      boxes = []

      #Parsing XML
      if os.path.exists(xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        #Reading boundary boxes and labels from xml
        labels = []
        for obj in root.findall('object'):
          bbx = obj.find('bndbox')
          xmin = float(bbx.find('xmin').text)
          xmax = float(bbx.find('xmax').text)
          ymin = float(bbx.find('ymin').text)
          ymax = float(bbx.find('ymax').text)
          boxes.append([xmin, ymin, xmax, ymax])
          label = obj.findtext('name')
          #Converting label to integer
          if label == "beer":
            label = 1
          else:
            label = 2
          labels.append(label)

      #Converting everything to tensor
      num_objs = len(boxes)
      if num_objs == 0:
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        area = torch.as_tensor(0, dtype=torch.float32)
        iscrowd = torch.zeros(0, dtype=torch.int64)
      else:
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
      
      labels = torch.as_tensor(labels, dtype=torch.int64)

      #Combining boxes and labels to dictionary
      target = {}
      target['boxes'] = boxes
      target['labels'] = labels
      target["area"] = area
      target["iscrowd"] = iscrowd
      return target

    def __len__(self):
        return len(self.imgs)
