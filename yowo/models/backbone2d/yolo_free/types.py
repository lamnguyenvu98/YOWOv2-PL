from typing import Literal

ELANNET = Literal['elannet_large', 'elannet_tiny', 'elannet_nano', 'elannet_huge']
SHUFFLENETV2 = Literal['shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0']
FPN_SIZE = Literal['nano', 'tiny', 'large', 'huge']
NECK = Literal['spp_block', 'spp_block_csp', 'sppf']
YOLO_FREE_VERSION = Literal['yolo_free_nano', 'yolo_free_tiny', 'yolo_free_large']