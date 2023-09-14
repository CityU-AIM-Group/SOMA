from yacs.config import CfgNode as CN
from numpy import pi

_C = CN()

# ------------------------------------------------------------------------
# Training
# ------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.LR = 2e-4
_C.TRAIN.LR_BACKBONE_NAMES = ["backbone.0"]
_C.TRAIN.LR_BACKBONE = 2e-5
_C.TRAIN.LR_LINEAR_PROJ_NAMES = ['reference_points', 'sampling_offsets']
_C.TRAIN.LR_LINEAR_PROJ_MULT = 0.1
_C.TRAIN.BATCH_SIZE = 2
_C.TRAIN.WEIGHT_DECAY = 1e-4
_C.TRAIN.EPOCHS = 50
_C.TRAIN.LR_DROP = 40
_C.TRAIN.LR_DROP_EPOCHS = None
_C.TRAIN.CLIP_MAX_NORM = 0.1 # gradient clipping max norm
_C.TRAIN.SGD = False # AdamW is used when setting this false


# ------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------
_C.MODEL = CN()

# Variants of Deformable DETR
_C.MODEL.WITH_BOX_REFINE = False
_C.MODEL.TWO_STAGE = False

# Model parameters
_C.MODEL.FROZEN_WEIGHTS = None # Path to the pretrained model. If set, only the mask head will be trained

# * Backbone
_C.MODEL.BACKBONE = 'resnet50' # Name of the convolutional backbone to use
_C.MODEL.DILATION = False # If true, we replace stride with dilation in the last convolutional block (DC5)
_C.MODEL.POSITION_EMBEDDING = 'sine' # ('sine', 'learned') Type of positional embedding to use on top of the image features
_C.MODEL.POSITION_EMBEDDING_SCALE = 2 * pi # position / size * scale
_C.MODEL.NUM_FEATURE_LEVELS = 4 # number of feature levels

# * Transformer
_C.MODEL.ENC_LAYERS = 6 # Number of encoding layers in the transformer
_C.MODEL.DEC_LAYERS = 6 # Number of decoding layers in the transformer
_C.MODEL.DIM_FEEDFORWARD = 1024 # Intermediate size of the feedforward layers in the transformer blocks
_C.MODEL.HIDDEN_DIM = 256 # Size of the embeddings (dimension of the transformer)
_C.MODEL.DROPOUT = 0.1 # Dropout applied in the transformer
_C.MODEL.NHEADS = 8 # Number of attention heads inside the transformer's attentions
_C.MODEL.NUM_QUERIES = 300 # Number of query slots
_C.MODEL.DEC_N_POINTS = 4
_C.MODEL.ENC_N_POINTS = 4

# * Segmentation
_C.MODEL.MASKS = False # Train segmentation head if the flag is provided

# * Domain Adaptation
_C.MODEL.BACKBONE_ALIGN = False
_C.MODEL.SPACE_ALIGN = False
_C.MODEL.CHANNEL_ALIGN = False
_C.MODEL.INSTANCE_ALIGN = False

# ------------------------------------------------------------------------
# Deformable DETR baseline Loss
# ------------------------------------------------------------------------
_C.LOSS = CN()
_C.LOSS.AUX_LOSS = True # auxiliary decoding losses (loss at each layer)

# * Matcher
_C.LOSS.SET_COST_CLASS = 2. # Class coefficient in the matching cost
_C.LOSS.SET_COST_BBOX = 5. # L1 box coefficient in the matching cost
_C.LOSS.SET_COST_GIOU = 2. # giou box coefficient in the matching cost

# * Loss coefficients
_C.LOSS.MASK_LOSS_COEF = 1.
_C.LOSS.DICE_LOSS_COEF = 1.
_C.LOSS.CLS_LOSS_COEF = 2.
_C.LOSS.BBOX_LOSS_COEF = 5.
_C.LOSS.GIOU_LOSS_COEF = 2.

_C.LOSS.SPACE_QUERY_LOSS_COEF = 0.1
_C.LOSS.CHANNEL_QUERY_LOSS_COEF = 0.1
_C.LOSS.INSTANCE_QUERY_LOSS_COEF = 0.1
_C.LOSS.FOCAL_ALPHA = 0.25
_C.LOSS.DA_GAMMA = 0


# ------------------------------------------------------------------------
# dataset parameters
# ------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.DA_MODE = 'source_only' # ('source_only', 'uda', 'oracle')
_C.DATASET.NUM_CLASSES = 9 # This should be set as max_class_id + 1
_C.DATASET.DATASET_FILE = 'cityscapes_to_foggy_cityscapes'
_C.DATASET.COCO_PATH = '../datasets'
_C.DATASET.COCO_PANOPTIC_PATH = None
_C.DATASET.REMOVE_DIFFICULT = False


# ------------------------------------------------------------------------
# Distributed
# ------------------------------------------------------------------------
_C.DIST = CN()
_C.DIST.DISTRIBUTED = False
_C.DIST.RANK = None
_C.DIST.WORLD_SIZE = None
_C.DIST.GPU = None
_C.DIST.DIST_URL = None
_C.DIST.DIST_BACKEND = None

# ------------------------------------------------------------------------
# Miscellaneous
# ------------------------------------------------------------------------
_C.OUTPUT_DIR = '' # path where to save, empty for no saving
_C.DEVICE = 'cuda' # device to use for training / testing
_C.RESUME = '' # resume from checkpoint
_C.START_EPOCH = 0 # start epoch
_C.EVAL = False
_C.NUM_WORKERS = 2
_C.CACHE_MODE = False # whether to cache images on memory
_C.SEED = 42 # Note this this cannot strictly control the same results. I don not know why

# ------------------------------------------------------------------------
# Adaptive Open-set Object Detection (AOOD)
# ------------------------------------------------------------------------
_C.AOOD = CN()
_C.AOOD.OPEN_SET = CN()
_C.AOOD.CROSS_DOMAIN = CN()

_C.AOOD.OW_DETR_ON = False
_C.AOOD.MOTIF_ON = False
_C.AOOD.OPENDET_DETR_ON = False
_C.DATASET.AOOD_SETTING = 1
_C.DATASET.AOOD_TASK = 4
_C.DATASET.AOOD_SCENE = 'cityscapes'
# _C.DATASET.AOOD_SCENE = 'pascal'

# global alignment + def-detr baseline
_C.AOOD.CROSS_DOMAIN.BACKBONE_LAMBDA = 1.0
_C.LOSS.BACKBONE_LOSS_COEF = 0.1
_C.LOAD_OPTIMIZER = True
_C.EVAL_EPOCH = 29

# For novel-class
_C.AOOD.OPEN_SET.MOTIF_ON = False
_C.AOOD.OPEN_SET.KNN = 5
_C.AOOD.OPEN_SET.TH = 0.5
_C.AOOD.OPEN_SET.MOTIF_LOSS_COEF = 1.0
_C.AOOD.OPEN_SET.WITH_SELF_LABELING = False
_C.AOOD.OPEN_SET.UNK_PROB = 0.0
_C.AOOD.OPEN_SET.WARM_UP = -1 # -1 indicates no warm-up
_C.AOOD.OPEN_SET.MOTIF_UPDATE = True
_C.AOOD.OPEN_SET.ALPHA = 0.01

# For novel-scene
_C.AOOD.CROSS_DOMAIN.WARM_UP = -1
_C.AOOD.CROSS_DOMAIN.MOTIF_ON = False
_C.AOOD.CROSS_DOMAIN.MOTIF_LOSS_COEF = 0.01
_C.AOOD.CROSS_DOMAIN.KNN = 5
_C.AOOD.CROSS_DOMAIN.BETA = 1.0


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()
