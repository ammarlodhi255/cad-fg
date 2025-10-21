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

def get_train_tf(size=224):
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
