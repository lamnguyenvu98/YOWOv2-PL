from .resnet import build_resnet_3d
from .resnext import build_resnext_3d
from .shufflenetv2 import build_shufflenetv2_3d

def build_backbone_3d(model_name: str, pretrained=False):
    if "resnet" in model_name:
        model, feat = build_resnet_3d(
            model_name=model_name,
            pretrained=pretrained
        )
    elif "resnext" in model_name:
        model, feat = build_resnext_3d(
            model_name=model_name,
            pretrained=pretrained
        )
    elif "shufflenet" in model_name:
        model, feat = build_shufflenetv2_3d(
            model_name=model_name,
            pretrained=pretrained
        )
    else:
        raise ValueError(f"Model {model_name} is not supported.")
    
    return model, feat