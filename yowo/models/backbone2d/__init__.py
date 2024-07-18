from .yolo_free.model import build_yolo_free


def build_backbone_2d(model_name: str, pretrained: bool = False, use_blurpool: bool = False):
    model, feat = build_yolo_free(
        model_name=model_name,
        pretrained=pretrained
    )
    if use_blurpool:
        from .blurpool2d import apply_blurpool
        apply_blurpool(model, replace_conv=False)
    return model, feat
