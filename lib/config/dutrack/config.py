from easydict import EasyDict as edict
import yaml

"""
Add default config for OSTrack.
"""
cfg = edict()

# MODEL
cfg.MODEL = edict()
cfg.MODEL.PRETRAIN_FILE = "mae_pretrain_vit_base.pth"
cfg.MODEL.EXTRA_MERGER = False

cfg.MODEL.RETURN_INTER = False
cfg.MODEL.RETURN_STAGES = []

# MODEL.BACKBONE
cfg.MODEL.BACKBONE = edict()
cfg.MODEL.BACKBONE.TYPE = "vit_base_patch16_224"
cfg.MODEL.BACKBONE.STRIDE = 16
cfg.MODEL.BACKBONE.MID_PE = False
cfg.MODEL.BACKBONE.SEP_SEG = False
cfg.MODEL.BACKBONE.CAT_MODE = 'direct'
cfg.MODEL.BACKBONE.MERGE_LAYER = 0
cfg.MODEL.BACKBONE.ADD_CLS_TOKEN = False
cfg.MODEL.BACKBONE.TOKEN_LEN = 1
cfg.MODEL.BACKBONE.TOP_K = 3
cfg.MODEL.BACKBONE.CLS_TOKEN_USE_MODE = 'ignore'
cfg.MODEL.BACKBONE.ATTN_TYPE = 'concat'

cfg.MODEL.BACKBONE.CE_LOC = []
cfg.MODEL.BACKBONE.CE_KEEP_RATIO = []
cfg.MODEL.BACKBONE.CE_TEMPLATE_RANGE = 'ALL'  # choose between ALL, CTR_POINT, CTR_REC, GT_BOX
cfg.MODEL.BACKBONE.BERT_DIR = 'ALL'
cfg.MODEL.BACKBONE.BLIP_DIR = 'ALL'

# MODEL.HEAD
cfg.MODEL.HEAD = edict()
cfg.MODEL.HEAD.TYPE = "CENTER"
cfg.MODEL.HEAD.NUM_CHANNELS = 256


# TRAIN
cfg.TRAIN = edict()
cfg.TRAIN.LR = 0.0001
cfg.TRAIN.WEIGHT_DECAY = 0.0001
cfg.TRAIN.EPOCH = 500
cfg.TRAIN.LR_DROP_EPOCH = 400
cfg.TRAIN.BATCH_SIZE = 16
cfg.TRAIN.NUM_WORKER = 8
cfg.TRAIN.OPTIMIZER = "ADAMW"
cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1
cfg.TRAIN.GIOU_WEIGHT = 2.0
cfg.TRAIN.L1_WEIGHT = 5.0
cfg.TRAIN.FREEZE_LAYERS = [0, ]
cfg.TRAIN.PRINT_INTERVAL = 50
cfg.TRAIN.VAL_EPOCH_INTERVAL = 20
cfg.TRAIN.GRAD_CLIP_NORM = 0.1
cfg.TRAIN.AMP = False
cfg.TRAIN.BBOX_TASK = False

cfg.TRAIN.CE_START_EPOCH = 20  # candidate elimination start epoch
cfg.TRAIN.CE_WARM_EPOCH = 80  # candidate elimination warm up epoch
cfg.TRAIN.DROP_PATH_RATE = 0.1  # drop path rate for ViT backbone

# TRAIN.SCHEDULER
cfg.TRAIN.SCHEDULER = edict()
cfg.TRAIN.SCHEDULER.TYPE = "step"
cfg.TRAIN.SCHEDULER.DECAY_RATE = 0.1

# DATA
cfg.DATA = edict()
cfg.DATA.SAMPLER_MODE = "causal"  # sampling methods
cfg.DATA.MEAN = [0.485, 0.456, 0.406]
cfg.DATA.STD = [0.229, 0.224, 0.225]
cfg.DATA.MAX_SAMPLE_INTERVAL = 200
# DATA.TRAIN
cfg.DATA.TRAIN = edict()
cfg.DATA.TRAIN.DATASETS_NAME = ["LASOT", "GOT10K_vottrain"]
cfg.DATA.TRAIN.DATASETS_RATIO = [1, 1]
cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 60000
# DATA.VAL
cfg.DATA.VAL = edict()
cfg.DATA.VAL.DATASETS_NAME = ["GOT10K_votval"]
cfg.DATA.VAL.DATASETS_RATIO = [1]
cfg.DATA.VAL.SAMPLE_PER_EPOCH = 10000
# DATA.SEARCH
cfg.DATA.SEARCH = edict()
cfg.DATA.SEARCH.SIZE = 320
cfg.DATA.SEARCH.FACTOR = 5.0
cfg.DATA.SEARCH.CENTER_JITTER = 4.5
cfg.DATA.SEARCH.SCALE_JITTER = 0.5
cfg.DATA.SEARCH.NUMBER = 1
# DATA.TEMPLATE
cfg.DATA.TEMPLATE = edict()
cfg.DATA.TEMPLATE.NUMBER = 1
cfg.DATA.TEMPLATE.SIZE = 128
cfg.DATA.TEMPLATE.FACTOR = 2.0
cfg.DATA.TEMPLATE.CENTER_JITTER = 0
cfg.DATA.TEMPLATE.SCALE_JITTER = 0

# TEST
cfg.TEST = edict()
cfg.TEST.TEMPLATE_FACTOR = 2.0
cfg.TEST.TEMPLATE_SIZE = 128
cfg.TEST.TEMPLATE_NUMBER = 1
cfg.TEST.MEMORY_THRESHOLD = 1000
cfg.TEST.SEARCH_FACTOR = 5.0
cfg.TEST.SEARCH_SIZE = 320
cfg.TEST.EPOCH = 500
cfg.TEST.FLOW_WINDOW_SIZE = 5
cfg.TEST.FLOW_UPDATE_INTERVAL = 10
cfg.TEST.SAVE_SCORES = True
cfg.TEST.PR_IOU_THRESHOLD = 0.5
# AQADU: Adaptive Quality-Aware Dynamic Update parameters
cfg.TEST.ITM_THRESHOLD = 0.005  # ITM threshold for consensus update quality filtering
cfg.TEST.MIN_UPDATE_INTERVAL = 5  # Minimum frames between consensus updates
cfg.TEST.MAX_UPDATE_INTERVAL = 15  # Maximum frames between consensus updates
# Enhanced: Confidence-Aware Template Selection and Adaptive Search
cfg.TEST.HIGH_CONF_THRESHOLD = 0.7  # High confidence threshold for template update
cfg.TEST.LOW_CONF_THRESHOLD = 0.3   # Low confidence threshold for search expansion
cfg.TEST.FAILURE_THRESHOLD = 0.15   # Failure detection threshold
cfg.TEST.MIN_SEARCH_FACTOR = 3.0    # Minimum search factor (high confidence)
cfg.TEST.MAX_SEARCH_FACTOR = 6.0    # Maximum search factor (failure recovery)
cfg.TEST.MOTION_WINDOW = 5          # Motion estimation window size
cfg.TEST.MAX_TEMPLATES = 10         # Maximum template pool size
# Enhanced V2: Smoothing and Multi-Hypothesis parameters
cfg.TEST.SCALE_SMOOTH_FACTOR = 0.4      # Scale smoothing factor
cfg.TEST.POSITION_SMOOTH_FACTOR = 0.2   # Position smoothing factor  
cfg.TEST.TEMPLATE_UPDATE_INTERVAL = 5   # Min frames between template updates
cfg.TEST.USE_MULTI_HYPOTHESIS = True    # Enable multi-hypothesis tracking
cfg.TEST.NUM_HYPOTHESES = 3             # Number of hypothesis candidates

# V4b/V5d: Quality-gated template management parameters
cfg.TEST.TEMPLATE_STORE_THRESHOLD = 0.5    # Min confidence to store template
cfg.TEST.MID_CONF_THRESHOLD = 0.4          # Mid-level confidence threshold
cfg.TEST.MIN_TEMPLATE_INTERVAL = 3         # Min frames between template storage
cfg.TEST.SCALE_DIVERSITY_THRESHOLD = 0.25  # Scale diversity threshold
cfg.TEST.SIZE_MATCH_WEIGHT = 0.0           # Size matching weight in template selection
cfg.TEST.CONF_WEIGHT = 1.0                 # Confidence weight in template selection
cfg.TEST.USE_RESPONSE_WEIGHTED_FUSION = False  # Enable response-weighted prediction fusion
cfg.TEST.FUSION_TEMPERATURE = 1.0          # Temperature for softmax in fusion

# V5e: Peak sharpness parameters for enhanced quality assessment
cfg.TEST.PEAK_SHARPNESS_TOPK = 10          # Top-k for peak sharpness calculation
cfg.TEST.SHARPNESS_WEIGHT = 0.3            # Sharpness weight in enhanced quality

# V6: Appearance diversity parameters for template selection
cfg.TEST.DIVERSITY_WEIGHT = 0.2            # Diversity weight in final template selection score
cfg.TEST.QUALITY_WEIGHT = 0.8              # Quality weight in final template selection score

# V7: Temporal consistency parameters for enhanced template selection
cfg.TEST.CONFIDENCE_EMA_ALPHA = 0.3        # EMA smoothing coefficient (smaller = smoother)
cfg.TEST.CONSISTENCY_WEIGHT = 0.1          # Temporal consistency weight in template selection

# V8: Spatial concentration parameters for quality assessment
cfg.TEST.SPATIAL_CONCENTRATION_RADIUS = 0.15  # Radius as fraction of response map size
cfg.TEST.SPATIAL_CONCENTRATION_TOPK = 10      # Top-k for concentration calculation
cfg.TEST.CONCENTRATION_WEIGHT = 0.15          # Concentration weight in enhanced quality

# V12: Peak Centrality parameters
cfg.TEST.PEAK_CENTRALITY_WEIGHT = 0.15        # Peak centrality weight in enhanced quality

# V9: Recency-guaranteed template selection and pool pruning
cfg.TEST.RECENCY_WEIGHT = 0.05                # Recency weight in template selection (within segment)
cfg.TEST.MAX_STORED_TEMPLATES = 30             # Maximum stored templates before pruning


def _edict2dict(dest_dict, src_edict):
    if isinstance(dest_dict, dict) and isinstance(src_edict, dict):
        for k, v in src_edict.items():
            if not isinstance(v, edict):
                dest_dict[k] = v
            else:
                dest_dict[k] = {}
                _edict2dict(dest_dict[k], v)
    else:
        return


def gen_config(config_file):
    cfg_dict = {}
    _edict2dict(cfg_dict, cfg)
    with open(config_file, 'w') as f:
        yaml.dump(cfg_dict, f, default_flow_style=False)


def _update_config(base_cfg, exp_cfg):
    if isinstance(base_cfg, dict) and isinstance(exp_cfg, edict):
        for k, v in exp_cfg.items():
            if k in base_cfg:
                if not isinstance(v, dict):
                    base_cfg[k] = v
                else:
                    _update_config(base_cfg[k], v)
            else:
                raise ValueError("{} not exist in config.py".format(k))
    else:
        return


def update_config_from_file(filename, base_cfg=None):
    exp_config = None
    with open(filename) as f:
        exp_config = edict(yaml.safe_load(f))
        if base_cfg is not None:
            _update_config(base_cfg, exp_config)
        else:
            _update_config(cfg, exp_config)
