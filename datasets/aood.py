# ------------------------------------------------------------------------
# Novel Scenes & Classes: Towards Adaptive Open-set Object Detection
# Modified by Wuyang Li
# ------------------------------------------------------------------------
import functools
import torch
import os
import tarfile
import collections
import logging
import copy
from torchvision.datasets import VisionDataset
import itertools
import util.misc as utils
import xml.etree.ElementTree as ET
from PIL import Image
import datasets.transforms as T

class AOODDetection(VisionDataset):

    def get_aood_settings_cityscapes(self, setting=2):
        
        NMES = ['person', 'car', 'train', 'rider', 'truck', 'motorcycle', 'bicycle', 'bus']
        UNK = ["unknown"]
    
        if setting == 1: # different semantics
            BASE_CLASSES = ['car', 'truck', 'bus']   
            NOVEL_CLASSES = ['person','motorcycle','train', 'bicycle' , 'rider'] 
        elif setting == 2: # similar semantics
            BASE_CLASSES = ['person', 'bicycle', 'bus']   
            NOVEL_CLASSES = ['car', 'truck','train', 'motorcycle', 'rider' ] 
        elif setting == 3: # frequency down
            BASE_CLASSES = ['person', 'car', 'rider']
            NOVEL_CLASSES = [ 'bicycle', 'train', 'truck', 'motorcycle', 'bus']
        elif setting == 4: # frequency top
            BASE_CLASSES = [ 'motorcycle', 'truck', 'bus']
            NOVEL_CLASSES = ['person', 'train', 'car','bicycle', 'rider']

        ALL_CLASSES= tuple(itertools.chain(BASE_CLASSES, NOVEL_CLASSES))
        CLASS_NAMES= tuple(itertools.chain(BASE_CLASSES, UNK))

        return BASE_CLASSES, NOVEL_CLASSES, ALL_CLASSES, CLASS_NAMES
    

    def get_aood_settings_pascal_voc(self, setting=1):
        PASCAL_CLASSES =   [        
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor"]
        UNK = ["unknown"]

        # if setting == 1:
        #     BASE_CLASSES = ALL_CLASSES[:19]
        #     NOVEL_CLASSES = ALL_CLASSES[19:]
        # elif setting == 2: 
        #     BASE_CLASSES = ALL_CLASSES[:15]
        #     NOVEL_CLASSES = ALL_CLASSES[15:]
        # elif setting == 3:
        #     BASE_CLASSES = ALL_CLASSES[:10]
        #     NOVEL_CLASSES = ALL_CLASSES[10:]
        # elif setting == 4:
        #     BASE_CLASSES = ALL_CLASSES[:5]
        #     NOVEL_CLASSES = ALL_CLASSES[5:]

        BASE_CLASSES = PASCAL_CLASSES[:10]
        NOVEL_CLASSES = PASCAL_CLASSES[10:]

        ALL_CLASSES= tuple(itertools.chain(PASCAL_CLASSES))
        BASE_CLASSES= tuple(itertools.chain(BASE_CLASSES))
        NOVEL_CLASSES= tuple(itertools.chain(NOVEL_CLASSES))
    
        CLASS_NAMES = tuple(itertools.chain(BASE_CLASSES, UNK)) 
        return BASE_CLASSES, NOVEL_CLASSES, ALL_CLASSES, CLASS_NAMES


    def __init__(self,
                 # args,
                 img_folder,
                 ann_folder,
                 data_list,
                 transforms=None,
                 remove_unk=False,
                 setting=1,
                 scene='pascal',
                 multi_task_eval_id = 4,
                 is_eval=False
                 ):
        super(AOODDetection, self).__init__(img_folder)

        self.images = []
        self.annotations = []
        self.imgids = []
        self.imgid2annotations = {}
        self.image_set = []
        self.transforms = transforms
        self.remove_unk = remove_unk
        self.is_eval = is_eval

        self.id_to_task_nme = {
            1: 'het-sem',
            2: 'hom-sem',
            3: 'freq-dec',
            4: 'freq-inc'
        }

        self.scene = scene
        if self.scene == 'cityscapes' or self.scene == 'bdd_daytime' or self.scene == 'foggy_cityscapes':
            self.BASE_CLASSES, self.NOVEL_CLASSES, self.ALL_CLASSES, self.CLASS_NAMES = self.get_aood_settings_cityscapes(setting)
        elif self.scene == 'pascal' or  self.scene == 'clipart':
            self.BASE_CLASSES, self.NOVEL_CLASSES, self.ALL_CLASSES, self.CLASS_NAMES = self.get_aood_settings_pascal_voc(setting)
        else:
             raise KeyError('undefined aood scenes')

        self.num_classes = len(self.CLASS_NAMES) # K+1 for model training
        self.num_base = len(self.BASE_CLASSES) # K
        self.unk_id =  self.num_classes - 1

        self.NOVEL_CLASSES_PER_TASK = self.NOVEL_CLASSES[:multi_task_eval_id]

        num_novel_per_task = len(self.NOVEL_CLASSES_PER_TASK )

        all_classes_id = range(len(self.ALL_CLASSES))

        self.bdd2city={
            'bike':'bicycle',
            'motor': 'motorcycle', 
        }

        self.base_id = all_classes_id[:self.num_base] # 0-k
        self.novel_id = all_classes_id[self.num_base:] # k- all
        self.per_task_novel_id = all_classes_id[self.num_base:self.num_base + num_novel_per_task]  # k- sel

        with open(data_list, "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        if self.remove_unk: # remove images without base-class objects
            if utils.is_main_process():
                print(''.join(80*['=']))
                print('source domain training set:')
                if self.scene != 'pascal':
                    print('AOOD Task: {}'.format(self.id_to_task_nme[setting]))
                print("BASE_CLASSES: {}".format(self.BASE_CLASSES))
                print("REALLOCATED CLASSES: {}".format(self.CLASS_NAMES))
            file_names = self.filter_imgs_without_base_objects(ann_folder, file_names)  
        elif is_eval: # inference: remove images without base-class objects
            if utils.is_main_process():
                print(''.join(80*['=']))
                print('target domain test set (task {}):'.format(multi_task_eval_id))
                print("BASE_CLASSES: {}".format(self.BASE_CLASSES))
                print("NOVEL_CLASSES: {}".format(self.NOVEL_CLASSES_PER_TASK))
            file_names = self.filter_imgs_without_base_novel_objects(ann_folder, file_names)
        else: # target domain training set: preserve all images
            if utils.is_main_process():
                print(''.join(80*['=']))
                print('target domain training set:')
                print('num images: {}'.format(len(file_names)))
                print("BASE_CLASSES: {}".format(self.BASE_CLASSES))
                print("NOVEL_CLASSES: {}".format(self.NOVEL_CLASSES))

        self.image_set.extend(file_names)

        suffix = ".png" if self.scene == 'cityscapes' or self.scene == 'foggy_cityscapes' else '.jpg'
        self.images.extend([os.path.join(img_folder, x + suffix) for x in file_names])
        self.annotations.extend([os.path.join(ann_folder, x + ".xml") for x in file_names])

        self.imgids = list(range(len(file_names)))
        self.imgids2img = dict(zip(self.imgids, file_names))
        self.imgid2annotations.update(dict(zip(self.imgids, self.annotations)))

        assert (len(self.images) == len(self.annotations) == len(self.imgids))

    def filter_imgs_without_base_novel_objects(self, ann_folder, file_names):
        new_file_names = []
        for x in file_names:
            anno = os.path.join(ann_folder, x + ".xml")
            tree = ET.parse(anno)
            target = self.parse_voc_xml(tree.getroot())
            flag=True
            for obj in target['annotation']['object']:
                cls = obj["name"]
                # if cls in self.bdd2city.keys():
                if cls in self.bdd2city.keys() and self.scene == 'bdd_daytime':
                    cls = self.bdd2city[cls]
                if cls not in self.BASE_CLASSES and cls not in self.NOVEL_CLASSES_PER_TASK:
                    flag=False
                    break
            if flag:
                new_file_names.append(x)

        print('original images: {}, after removing images without base and novel objects: {}.'.format(len(file_names), len(new_file_names)))
        return new_file_names

    def filter_imgs_without_base_objects(self, ann_folder, file_names):
        new_file_names = []
        for x in file_names:
            anno = os.path.join(ann_folder, x + ".xml")
            tree = ET.parse(anno)
            target = self.parse_voc_xml(tree.getroot())
  
            for obj in target['annotation']['object']:
                cls = obj["name"]
                if cls in self.bdd2city.keys():
                    cls = self.bdd2city[cls]

                if cls in self.BASE_CLASSES:
                    new_file_names.append(x)
                    break
        print('original images: {}, after removing images without base objects: {}.'.format(len(file_names), len(new_file_names)))
        return new_file_names

    @functools.lru_cache(maxsize=None)
    def load_instances(self, img_id):
        tree = ET.parse(self.imgid2annotations[img_id])
        target = self.parse_voc_xml(tree.getroot())
        instances = []
        for obj in target['annotation']['object']:
            cls = obj["name"]

            if cls in self.bdd2city.keys():
                cls = self.bdd2city[cls]
            bbox = obj["bndbox"]
            bbox = [float(bbox[x]) for x in ["xmin", "ymin", "xmax", "ymax"]]
            bbox[0] -= 1.0
            bbox[1] -= 1.0
            instance = dict(
                    category_id=self.ALL_CLASSES.index(cls),
                    bbox=bbox,
                    area=(bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                    image_id=img_id
                )
            instances.append(instance)    
        return target, instances

    def remove_novel_instances(self, target):
        # for the labelled training
        entry = copy.copy(target)
        for annotation in copy.copy(entry):
            if annotation["category_id"] not in self.base_id:
                entry.remove(annotation)
        return entry

    def label_all_novel_instances_as_unk(self, target):
        # for the unlabelled training
        entry = copy.copy(target)
        for annotation in copy.copy(entry):
            # for annotation in entry:
            if annotation["category_id"] not in self.base_id:
                annotation["category_id"] = self.unk_id

        return entry

    def label_per_task_novel_instances_as_unk(self, target):
        # for the unlabelled training
        entry = copy.copy(target)
        for annotation in copy.copy(entry):
            # for annotation in entry:
            if annotation["category_id"] in self.base_id:
                continue          
            elif annotation["category_id"] in self.per_task_novel_id:
                annotation["category_id"] = self.unk_id
            else:
                entry.remove(annotation)
        return entry
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a dictionary of the XML tree.
        """

        img = Image.open(self.images[index]).convert('RGB')
        target, instances = self.load_instances(self.imgids[index])

        if self.remove_unk:
            instances = self.remove_novel_instances(instances)
        elif self.is_eval:
            instances = self.label_per_task_novel_instances_as_unk(instances)
        else:
            instances = self.label_all_novel_instances_as_unk(instances)

        w, h = map(target['annotation']['size'].get, ['width', 'height'])
        target = dict(
            image_id=torch.tensor([self.imgids[index]], dtype=torch.int64),
            labels=torch.tensor([i['category_id'] for i in instances], dtype=torch.int64),
            area=torch.tensor([i['area'] for i in instances], dtype=torch.float32),
            boxes=torch.as_tensor([i['bbox'] for i in instances], dtype=torch.float32),
            orig_size=torch.as_tensor([int(h), int(w)]),
            size=torch.as_tensor([int(h), int(w)]),
            iscrowd=torch.zeros(len(instances), dtype=torch.uint8)
        )
    
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.images)

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            if node.tag == 'annotation':
                def_dic['object'] = [def_dic['object']]
            voc_dict = {
                node.tag:
                    {ind: v[0] if len(v) == 1 else v
                     for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict

