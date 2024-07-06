from .yolo_free.model import build_yolo_free

def build_backbone_2d(model_name: str, pretrained: bool = False):
    model, feat = build_yolo_free(
        model_name=model_name,
        pretrained=pretrained
    )
    return model, feat