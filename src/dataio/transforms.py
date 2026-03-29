# import albumentations as A
# from albumentations.pytorch import ToTensorV2

# def get_train_tf(size=224):
#     return A.Compose([
#         A.Resize(size,size,always_apply=True),
#         A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=15, border_mode=0, p=0.8),
#         A.RandomBrightnessContrast(0.1,0.1,p=0.5),
#         A.GaussNoise(p=0.2),
#         ToTensorV2()
#     ])

# def get_val_tf(size=224):
#     return A.Compose([A.Resize(size,size,always_apply=True), ToTensorV2()])


import albumentations as A
from albumentations.pytorch import ToTensorV2
import inspect


def get_train_tf_basic(size=224, translate_px=25, rotate_deg=25, scale_range=(0.8, 1.7)):
    """Basic augmentation: only translation (±px), rotation (±deg), scaling.

    Always includes Resize -> Affine -> Normalize -> ToTensorV2.
    No color/noise/blur/flip operations.
    """
    # Prefer pixel-based translation if supported; else fall back to percent of size
    use_px = 'translate_px' in inspect.signature(A.Affine.__init__).parameters
    if use_px:
        aff = A.Affine(
            translate_px={"x": (-int(translate_px), int(translate_px)), "y": (-int(translate_px), int(translate_px))},
            scale=tuple(float(x) for x in scale_range),
            rotate=(-float(rotate_deg), float(rotate_deg)),
            p=1.0,
        )
    else:
        # approximate using percent of the resized size
        shift = float(translate_px) / float(size)
        aff = A.Affine(
            translate_percent=shift,
            scale=tuple(float(x) for x in scale_range),
            rotate=(-float(rotate_deg), float(rotate_deg)),
            p=1.0,
        )
    return A.Compose([
        A.Resize(size, size),
        aff,
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_train_tf(size=224):
    """Legacy/extended augmentation pipeline (flip, affine, color, noise).

    Kept for backward-compatibility. Prefer get_train_tf_basic for default
    affine-only augmentation.
    """
    return A.Compose([
        A.Resize(size, size),
        A.Affine(translate_percent=0.05, scale=(0.85, 1.15), rotate=(-15, 15), p=0.8),
        A.RandomBrightnessContrast(0.1, 0.1, p=0.5),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_val_tf(size=224):
    return A.Compose([
        A.Resize(size, size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
