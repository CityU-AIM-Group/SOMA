# ------------------------------------------------------------------------
# Novel Scenes & Classes: Towards Adaptive Open-set Object Detection
# Modified by Wuyang Li
# ----------------------------------------------
# Created by Wei-Jie Huang
# ----------------------------------------------
from pathlib import Path
from torch.utils.data import Dataset
from datasets.coco import CocoDetection, make_coco_transforms
from datasets.aood import AOODDetection
from util.misc import get_local_rank, get_local_size, nested_tensor_from_tensor_list

def get_paths(root):
    root = Path(root)
    return {
        'cityscapes': {
            'train_img': root / 'Cityscapes/leftImg8bit/train',
            'val_img': root / 'Cityscapes/leftImg8bit/val',
            'train_anno': root / 'Cityscapes/cocoAnnotations/cityscapes_train_cocostyle.json',
            'val_img': root / 'Cityscapes/leftImg8bit/val',
            'val_anno': root / 'Cityscapes/cocoAnnotations/cityscapes_foggy_val_cocostyle.json',

            'train_xml': root / 'Cityscapes/AOOD_Annotations',
            'val_xml': root / 'Cityscapes/AOOD_Annotations',
            'train_data_list': root / 'Cityscapes/AOOD_Main/train_source.txt',
            'val_data_list': root / 'Cityscapes/AOOD_Main/val_source.txt',
        },
        'cityscapes_caronly': {
            'train_img': root / 'Cityscapes/leftImg8bit/train',
            'train_anno': root / 'Cityscapes/annotations/cityscapes_caronly_train.json',
            'val_img': root / 'Cityscapes/leftImg8bit/val',
            'val_anno': root / 'Cityscapes/annotations/cityscapes_caronly_val.json',
        },
        'foggy_cityscapes': {
            'train_img': root / 'Cityscapes/leftImg8bit_foggy/train',
            'train_anno': root / 'Cityscapes/cocoAnnotations/cityscapes_foggy_train_cocostyle.json',
            # 'val_img': root / 'Cityscapes/leftImg8bit_foggy/val',
            # 'val_anno': root / 'Cityscapes/cocoAnnotations/cityscapes_foggy_val_cocostyle.json',
            'val_img': root / 'Cityscapes/leftImg8bit_foggy/train',
            'val_anno': root / 'Cityscapes/cocoAnnotations/cityscapes_foggy_train_cocostyle.json',
            
            'train_xml': root / 'Cityscapes/AOOD_Annotations',
            'train_data_list': root / 'Cityscapes/AOOD_Main/train_target.txt',

            'val_xml': root / 'Cityscapes/AOOD_Annotations',
            # 'val_data_list': root / 'Cityscapes/AOOD_Main/val_target.txt',
            'val_data_list': root / 'Cityscapes/AOOD_Main/train_target.txt',
        },
        'sim10k': {
            'train_img': root / 'sim10k/VOC2012/JPEGImages',
            'train_anno': root / 'sim10k/annotations/sim10k_caronly.json',
        },
        'bdd_daytime': {
            'train_img': root / 'bdd_daytime/JPEGImages',
            'val_img': root / 'bdd_daytime/JPEGImages',
            'train_xml': root / 'bdd_daytime/Annotations',
            'train_data_list': root / 'bdd_daytime/ImageSets/Main/train.txt',
            'val_xml': root / 'bdd_daytime/Annotations',
            'val_data_list': root / 'bdd_daytime/ImageSets/Main/val.txt',

        },
        'pascal': {
            'train_img': root / 'VOCdevkit/VOC2012/JPEGImages',
            'train_xml': root / 'VOCdevkit/VOC2012/Annotations',
            'train_data_list': root / 'VOCdevkit/VOC2012/ImageSets/Main/trainval.txt',
            'val_img': root / 'VOCdevkit/VOC2012/JPEGImages',
            'val_xml': root / 'VOCdevkit/VOC2012/Annotations',
            'val_data_list': root / 'VOCdevkit/VOC2012/ImageSets/Main/trainval.txt',
        },
        'clipart': {
            'train_img': root / 'clipart/JPEGImages',
            'train_xml': root / 'clipart/Annotations',
            'train_data_list': root / 'clipart/ImageSets/Main/all.txt',
            'val_img': root / 'clipart/JPEGImages',
            'val_xml': root / 'clipart/Annotations',
            'val_data_list': root / 'clipart/ImageSets/Main/all.txt',
        },
    }

class AOODDataset(Dataset):
    def __init__(self, source_img_folder, source_ann_folder, source_data_list, target_img_folder, target_ann_folder, target_data_list,
                 transforms, setting, scene):
        self.source = AOODDetection(
            img_folder=source_img_folder,
            ann_folder=source_ann_folder,
            data_list = source_data_list,
            remove_unk = True,
            transforms=transforms,
            setting=setting,
            scene = scene[0],
        )

        self.target = AOODDetection(
            img_folder=target_img_folder,
            ann_folder=target_ann_folder,
            data_list=target_data_list,
            transforms=transforms,
            remove_unk=False,
            setting=setting,
            scene = scene[1],

        )

    def __len__(self):
        return max(len(self.source), len(self.target))
        # return min(len(self.source), len(self.target))

    def __getitem__(self, idx):
        source_img, source_target = self.source[idx % len(self.source)]
        target_img, _ = self.target[idx % len(self.target)]
        return source_img, target_img, source_target

class DADataset(Dataset):
    def __init__(self, source_img_folder, source_ann_file, target_img_folder, target_ann_file,
                 transforms, return_masks, cache_mode=False, local_rank=0, local_size=1):
        self.source = CocoDetection(
            img_folder=source_img_folder,
            ann_file=source_ann_file,
            transforms=transforms,
            return_masks=return_masks,
            cache_mode=cache_mode,
            local_rank=local_rank,
            local_size=local_size
        )

        self.target = CocoDetection(
            img_folder=target_img_folder,
            ann_file=target_ann_file,
            transforms=transforms,
            return_masks=return_masks,
            cache_mode=cache_mode,
            local_rank=local_rank,
            local_size=local_size
        )

    def __len__(self):
        return max(len(self.source), len(self.target))

    def __getitem__(self, idx):
        source_img, source_target = self.source[idx % len(self.source)]
        target_img, _ = self.target[idx % len(self.target)]
        return source_img, target_img, source_target


def collate_fn(batch):
    source_imgs, target_imgs, source_targets = list(zip(*batch))
    samples = nested_tensor_from_tensor_list(source_imgs + target_imgs)
    return samples, source_targets


def build(image_set, cfg, multi_task_eval_id=4):
    paths = get_paths(cfg.DATASET.COCO_PATH)
    source_domain, target_domain = cfg.DATASET.DATASET_FILE.split('_to_')
    if image_set == 'val':
        if cfg.DATASET.DA_MODE == 'aood':
            return AOODDetection(
                img_folder=paths[target_domain]['val_img'],
                ann_folder=paths[target_domain]['val_xml'],
                data_list=paths[target_domain]['val_data_list'],
                transforms=make_coco_transforms(image_set),
                remove_unk=False,
                setting= cfg.DATASET.AOOD_SETTING,
                scene = target_domain,
                multi_task_eval_id = multi_task_eval_id, 
                is_eval =True,

                )
        else:
            return CocoDetection(
                img_folder=paths[target_domain]['val_img'],
                ann_file=paths[target_domain]['val_anno'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )
    elif image_set == 'train':
        if cfg.DATASET.DA_MODE == 'source_only':
            return CocoDetection(
                img_folder=paths[source_domain]['train_img'],
                ann_file=paths[source_domain]['train_anno'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size(),
            )
        elif cfg.DATASET.DA_MODE == 'oracle':
            return CocoDetection(
                img_folder=paths[target_domain]['train_img'],
                ann_file=paths[target_domain]['train_anno'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )
        elif cfg.DATASET.DA_MODE == 'uda':
            return DADataset(
                source_img_folder=paths[source_domain]['train_img'],
                source_ann_file=paths[source_domain]['train_anno'],
                target_img_folder=paths[target_domain]['train_img'],
                target_ann_file=paths[target_domain]['train_anno'],
                transforms=make_coco_transforms(image_set),
                return_masks=cfg.MODEL.MASKS,
                cache_mode=cfg.CACHE_MODE,
                local_rank=get_local_rank(),
                local_size=get_local_size()
            )

        elif cfg.DATASET.DA_MODE == 'aood':
            return AOODDataset(
                source_img_folder=paths[source_domain]['train_img'],
                source_ann_folder=paths[source_domain]['train_xml'],
                source_data_list=paths[source_domain]['train_data_list'],

                target_img_folder=paths[target_domain]['train_img'],
                target_ann_folder=paths[target_domain]['train_xml'],
                target_data_list=paths[target_domain]['train_data_list'],

                transforms=make_coco_transforms(image_set),
                setting=cfg.DATASET.AOOD_SETTING,
                scene = [source_domain, target_domain]
            )
        else:
            raise ValueError(f'Unknown argument cfg.DATASET.DA_MODE {cfg.DATASET.DA_MODE}')
    raise ValueError(f'unknown image set {image_set}')
