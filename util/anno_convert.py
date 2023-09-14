# ----------------------------------------------
# Created by Wei-Jie Huang
# A collection of annotation conversion scripts
# We provide this for reference only
# ----------------------------------------------


import json
from pathlib import Path
from tqdm import tqdm


r"""
coco_anno_dict is like {
    "images": list of image_info's
    "annotations": list of annotation_info's
    "categories": list of categorie_info's
}
where img_info is like: {
    "id": ...,                      # 0-indexed
    "width": ...,
    "height": ...,
    "file_name": ...,
}, annotation_info is like: {
    "id": ...,                      # 0-indexed
    "image_id": ...,
    "category_id": ...,
    "segmentation": ...,
    "iscrowd": ...,
    "area": ...,
    "bbox": ...,                    # (x, y, w, h)
}, and category_info is like: {
    "id": ...,                      # 1-indexed
    "name": ...,
}
"""


def sim10k_to_coco(
    src_path: str = "VOC2012/Annotations",
    des_path: str = "annotations/sim10k_caronly.json",
    categories: tuple = ("car",)
    ) -> None:

    r""" Convert Sim10k (in VOC format) into COCO format.
    Args:
        src_path: path of the directory containing VOC-format annotations
        des_path: destination of the converted COCO-fomat annotation
        categories: only category ``car`` is considered by default
    """

    from xml.etree import ElementTree

    src_path = Path(src_path)
    des_path = Path(des_path)
    assert src_path.exists(), "Annotation directory does not exist"
    if des_path.exists():
        print(f"{des_path} exists. Override? (y/n)", end=" ")
        if input() != "y":
            print("Abort")
            return
    else:
        des_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialization
    coco_anno_dict = {
        "images": [],
        "categories": [],
        "annotations": [],
    }
    num_images = 0
    num_categories = 0
    num_annotations = 0

    # Categories
    category_to_id = {}
    for category in categories:
        coco_anno_dict["categories"].append({
            "id": num_categories + 1,
            "name": category
        })
        category_to_id[category] = num_categories + 1
        num_categories += 1

    # Start Conversion
    for anno_file in tqdm(list(src_path.glob("*.xml"))):
        et_root = ElementTree.parse(anno_file).getroot()

        ##### Images #####
        img_info = {
            "id": num_images,
            "file_name": anno_file.stem + ".jpg",
        }
        num_images += 1

        # Image Size
        size = et_root.find("size")
        img_info["width"] = int(size.find("width").text)
        img_info["height"] = int(size.find("height").text)

        coco_anno_dict["images"].append(img_info)

        ##### Annotations #####
        for anno_object in et_root.findall("object"):
            category = anno_object.find("name").text
            if category not in categories:
                continue
            anno_info = {
                "id": num_annotations,
                "image_id": img_info["id"],
                "category_id": category_to_id[category],
                "segmentation": [],
                "iscrowd": 0
            }
            num_annotations += 1

            # Bounding box
            bndbox = anno_object.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)
            # COCO format expects (x, y, w, h)
            anno_info["bbox"] = [xmin, ymin, round(xmax - xmin, 2), round(ymax - ymin, 2)]
            anno_info["area"] = round(anno_info["bbox"][2] * anno_info["bbox"][3], 2)

            coco_anno_dict["annotations"].append(anno_info)

    print("# of images:", num_images)
    print("# of categories:", num_categories)
    print("# of annotations:", num_annotations)

    with open(des_path, 'w') as f:
        f.write(json.dumps(coco_anno_dict, indent=4))
    print(f"Convert successfully to {des_path}")


def bdd100k_daytime_to_coco(
    src_path: str = "labels/bdd100k_labels_images_train.json",
    des_path: str = "annotations/bdd_daytime_train.json",
    categories: tuple = (
        "person", "rider", "car", "truck", "bus", "train", "motor", "bike")
    ) -> None:

    r""" Extract ``daytime`` subset from BDD100k dataset and convert into COCO format.
    Args:
        src_path: source of the annotation json file
        des_path: destination of the converted COCO-fomat annotation
        categories: categories used
    """

    src_path = Path(src_path)
    des_path = Path(des_path)
    assert src_path.exists(), "Source annotation file does not exist"
    if des_path.exists():
        print(f"{des_path} exists. Override? (y/n)", end=" ")
        if input() != "y":
            print("Abort")
            return
    else:
        des_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialization
    coco_anno_dict = {
        "images": [],
        "categories": [],
        "annotations": [],
    }
    num_images = 0
    num_categories = 0
    num_annotations = 0

    # Categories
    category_to_id = {}
    for category in categories:
        coco_anno_dict["categories"].append({
            "id": num_categories + 1,
            "name": category
        })
        category_to_id[category] = num_categories + 1
        num_categories += 1

    with open(src_path, 'r') as f:
        raw_img_annos = json.load(f)
    # Start Conversion
    for raw_img_anno in tqdm(raw_img_annos):
        if raw_img_anno["attributes"]["timeofday"] != "daytime":
            continue

        ##### Images #####
        img_info = {
            "id": num_images,
            "file_name": raw_img_anno["name"],
            "height": 720,
            "width": 1280
        }
        coco_anno_dict["images"].append(img_info)
        num_images += 1

        ##### Annotations #####
        for label in raw_img_anno["labels"]:
            if label["category"] not in category_to_id or "box2d" not in label:
                continue
            anno_info = {
                "id": num_annotations,
                "image_id": img_info["id"],
                "category_id": category_to_id[label["category"]],
                "segmentation": [],
                "iscrowd": 0,
            }
            num_annotations += 1

            # Bbox
            x1 = label["box2d"]["x1"]
            y1 = label["box2d"]["y1"]
            x2 = label["box2d"]["x2"]
            y2 = label["box2d"]["y2"]
            anno_info["bbox"] = [x1, y1, x2 - x1, y2 - y1]
            anno_info["area"] = float((x2 - x1) * (y2 - y1))
            coco_anno_dict["annotations"].append(anno_info)

    print("# of images:", num_images)
    print("# of categories:", num_categories)
    print("# of annotations:", num_annotations)

    with open(des_path, 'w') as f:
        f.write(json.dumps(coco_anno_dict, indent=4))
    print(f"Convert successfully to {des_path}")


def cityscapes_to_coco(
    src_path: str = "gtFine/train",
    des_path: str = "annotations/cityscapes_train.json",
    car_only: bool = False,
    foggy: bool = False,
    categories: tuple = (
        "person", "rider", "car", "truck", "bus", "train", "motor", "bike")
    ) -> None:

    r"""Convert Cityscapes into COCO format.
        Ref: https://github.com/facebookresearch/Detectron/blob/7aa91aa/tools/convert_cityscapes_to_coco.py
    Args:
        src_path: path of the directory containing Cityscapes annotations
        des_path: destination of the converted COCO-fomat annotation
        car_only: whether extract category ``car`` only. used in Syn-to-real adaptation
        foggy: whether extract from foggy cityscapes. used in weather adaptation
        categories: categories used
    """

    def get_instances_with_polygons(imageFileName):
        r""" Ref: https://github.com/facebookresearch/Detectron/issues/111#issuecomment-363425465"""
        import os
        import sys
        import cv2
        import numpy as np
        from PIL import Image
        from cityscapesscripts.evaluation.instance import Instance
        from cityscapesscripts.helpers.csHelpers import labels, id2label

        # Load image
        img = Image.open(imageFileName)

        # Image as numpy array
        imgNp = np.array(img)

        # Initialize label categories
        instances = {}
        for label in labels:
            instances[label.name] = []

        # Loop through all instance ids in instance image
        for instanceId in np.unique(imgNp):
            if instanceId < 1000:
                continue

            instanceObj = Instance(imgNp, instanceId)
            instanceObj_dict = instanceObj.toDict()

            if id2label[instanceObj.labelID].hasInstances:
                mask = (imgNp == instanceId).astype(np.uint8)
                contour, hier = cv2.findContours(
                    mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                polygons = [c.reshape(-1).tolist() for c in contour]
                instanceObj_dict["contours"] = polygons

            instances[id2label[instanceObj.labelID].name].append(
                instanceObj_dict)
        return instances

    def polygon_to_bbox(polygon: list) -> list:
        """Convert polygon into COCO-format bounding box."""

        # https://github.com/facebookresearch/maskrcnn-benchmark/issues/288#issuecomment-449098063
        TO_REMOVE = 1

        x0 = min(min(p[::2]) for p in polygon)
        x1 = max(max(p[::2]) for p in polygon)
        y0 = min(min(p[1::2]) for p in polygon)
        y1 = max(max(p[1::2]) for p in polygon)

        bbox = [x0, y0, x1 -x0 + TO_REMOVE, y1 - y0 + TO_REMOVE]
        return bbox

    src_path = Path(src_path)
    des_path = Path(des_path)
    assert src_path.exists(), "Source annotation file does not exist"
    if des_path.exists():
        print(f"{des_path} exists. Override? (y/n)", end=" ")
        if input() != "y":
            print("Abort")
            return
    else:
        des_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialization
    coco_anno_dict = {
        "images": [],
        "categories": [],
        "annotations": [],
    }
    num_images = 0
    num_categories = 0
    num_annotations = 0

    # Categories
    if car_only:
        categories = ("car",)
    category_to_id = {}
    for category in categories:
        coco_anno_dict["categories"].append({
            "id": num_categories + 1,
            "name": category
        })
        category_to_id[category] = num_categories + 1
        num_categories += 1

    # Start Conversion
    for file in tqdm(list(src_path.rglob("*instanceIds.png"))):
        ##### Images #####
        img_info = {"id": num_images}
        num_images += 1
        img_info["file_name"] = \
            str(file.name).split("_", maxsplit=1)[0] + "/" + \
            str(file.name).replace("gtFine", "leftImg8bit").replace("_instanceIds", "")
        if foggy:
            img_info["file_name"] = \
                img_info["file_name"].replace("leftImg8bit", "leftImg8bit_foggy_beta_0.02")
        with open(str(file).replace("instanceIds.png", "polygons.json"), "r") as f:
            polygon_info = json.load(f)
            img_info["width"] = polygon_info["imgWidth"]
            img_info["height"] = polygon_info["imgHeight"]
        coco_anno_dict["images"].append(img_info)

        ##### Annotations #####
        instances = get_instances_with_polygons(str(file.absolute()))
        for category in instances.keys():
            if category not in categories:
                continue
            for instance in instances[category]:
                anno_info = {
                    "id": num_annotations,
                    "image_id": img_info["id"],
                    "category_id": category_to_id[category],
                    "segmentation": [],
                    "iscrowd": 0,
                    "area": instance["pixelCount"],
                    "bbox": polygon_to_bbox(instance["contours"]),
                }
                num_annotations += 1
                coco_anno_dict["annotations"].append(anno_info)

    print("# of images:", num_images)
    print("# of categories:", num_categories)
    print("# of annotations:", num_annotations)

    with open(des_path, 'w') as f:
        f.write(json.dumps(coco_anno_dict, indent=4))
    print(f"Convert successfully to {des_path}")


if __name__ == "__main__":
    sim10k_to_coco(
        src_path="VOC2012/Annotations",
        des_path="annotations/sim10k_caronly.json"
    )
    bdd100k_daytime_to_coco(
        src_path="labels/bdd100k_labels_images_train.json",
        des_path="annotations/bdd_daytime_train.json"
    )
    bdd100k_daytime_to_coco(
        src_path="labels/bdd100k_labels_images_val.json",
        des_path="annotations/bdd_daytime_val.json"
    )
    cityscapes_to_coco(
        src_path="gtFine/train",
        des_path="annotations/cityscapes_train.json",
    )
    cityscapes_to_coco(
        src_path="gtFine/val",
        des_path="annotations/cityscapes_val.json",
    )
    cityscapes_to_coco(
        src_path="gtFine/train",
        des_path="annotations/cityscapes_caronly_train.json",
        car_only=True,
    )
    cityscapes_to_coco(
        src_path="gtFine/val",
        des_path="annotations/cityscapes_caronly_val.json",
        car_only=True,
    )
    cityscapes_to_coco(
        src_path="gtFine/train",
        des_path="annotations/foggy_cityscapes_train.json",
        foggy=True,
    )
    cityscapes_to_coco(
        src_path="gtFine/val",
        des_path="annotations/foggy_cityscapes_val.json",
        foggy=True,
    )
