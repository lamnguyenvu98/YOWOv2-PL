import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url

from yowo.utils.validate import validate_literal_types
from .backbone import build_backbone
from .neck import build_neck
from .fpn import build_fpn
from .head import build_head
from .types import YOLO_FREE_VERSION
from .config import (
    YOLO_FREE_CONFIG,
    MODEL_URLS
)

# Anchor-free YOLO
class FreeYOLO(nn.Module):
    def __init__(self, config):
        super(FreeYOLO, self).__init__()
        # --------- Basic Config -----------
        self.cfg = config

        # --------- Network Parameters ----------
        ## backbone
        self.backbone, bk_dim = build_backbone(self.cfg['backbone'])

        ## neck
        self.neck = build_neck(
            model_name=self.cfg['neck'],
            expand_ratio=self.cfg['expand_ratio'],
            neck_act=self.cfg['neck_act'],
            neck_norm=self.cfg['neck_norm'],
            neck_depthwise=self.cfg['neck_depthwise'],
            in_dim=bk_dim[-1], 
            out_dim=self.cfg['neck_dim']
        )
        
        ## fpn
        self.fpn = build_fpn(
            fpn_size=self.cfg['fpn_size'],
            fpn_depthwise=self.cfg['fpn_depthwise'],
            fpn_norm=self.cfg['fpn_norm'],
            fpn_act=self.cfg['fpn_act'],
            in_dims=self.cfg['fpn_dim'], 
            out_dim=self.cfg['head_dim']
        )

        ## non-shared heads
        self.non_shared_heads = nn.ModuleList(
            [
                build_head(
                    num_cls_head=self.cfg['num_cls_head'],
                    num_reg_head=self.cfg['num_reg_head'],
                    head_act=self.cfg['head_act'],
                    head_norm=self.cfg['head_norm'],
                    head_dim=self.cfg['head_dim'],
                    head_depthwise=self.cfg['head_depthwise']
                ) 
                for _ in range(len(self.cfg['stride']))
            ]
        )

    def forward(self, x):
        # backbone
        feats = self.backbone(x)

        # neck
        feats['layer4'] = self.neck(feats['layer4'])

        # fpn
        pyramid_feats = [feats['layer2'], feats['layer3'], feats['layer4']]
        pyramid_feats = self.fpn(pyramid_feats)

        # non-shared heads
        all_cls_feats = []
        all_reg_feats = []
        for feat, head in zip(pyramid_feats, self.non_shared_heads):
            # [B, C, H, W]
            cls_feat, reg_feat = head(feat)

            all_cls_feats.append(cls_feat)
            all_reg_feats.append(reg_feat)

        return all_cls_feats, all_reg_feats

# build FreeYOLO
def build_yolo_free(model_name: YOLO_FREE_VERSION = 'yolo_free_large', pretrained: bool = False):
    validate_literal_types(model_name, YOLO_FREE_VERSION)
    # model config
    cfg = YOLO_FREE_CONFIG[model_name]

    # FreeYOLO
    model = FreeYOLO(cfg)
    feat_dims = [model.cfg['head_dim']] * 3

    # Load COCO pretrained weight
    if pretrained:
        url = MODEL_URLS[model_name]

        # check
        if url is None:
            print('No 2D pretrained weight ...')
            return model, feat_dims
        else:
            print('Loading 2D backbone pretrained weight: {}'.format(model_name.upper()))

            # state dict
            checkpoint = load_state_dict_from_url(url, map_location='cpu')
            checkpoint_state_dict = checkpoint.pop('model')

            # model state dict
            model_state_dict = model.state_dict()
            # check
            for k in list(checkpoint_state_dict.keys()):
                if k in model_state_dict:
                    shape_model = tuple(model_state_dict[k].shape)
                    shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                    if shape_model != shape_checkpoint:
                        # print(k)
                        checkpoint_state_dict.pop(k)
                else:
                    checkpoint_state_dict.pop(k)
                    # print(k)

            model.load_state_dict(checkpoint_state_dict, strict=False)

    return model, feat_dims


if __name__ == '__main__':
    model, fpn_dim = build_yolo_free(model_name='yolo_free_nano', pretrained=True)
    model.eval()

    x = torch.randn(2, 3, 64, 64)
    cls_feats, reg_feats = model(x)

    for cls_feat, reg_feat in zip(cls_feats, reg_feats):
        print(cls_feat.shape, reg_feat.shape)