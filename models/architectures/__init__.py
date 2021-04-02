def backbone(backbone):
    if "vgg" in backbone:
        from .vgg import VGGBackbone
    return VGGBackbone(backbone)
