import os, yaml, random, numpy as np, torch, pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from torch.utils.data import DataLoader
from src.dataio.transforms import get_train_tf, get_val_tf
from src.dataio.dataset import CadicaDataset
from src.models.resnet_backbone import ResNetBackbone
from src.models.late_fusion import TabularMLP, LateFusionClassifier
from src.models.early_fusion_patient import ResNetEarlyFusionPatient
from src.trainloop.engine import train_one_epoch, evaluate
from src.models.inception_backbone import InceptionV3Backbone, InceptionV3EarlyFusionPatient

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# cfg = yaml.safe_load(open("configs/default.yaml"))
# set_seed(cfg["seed"])
# device = "cuda" if torch.cuda.is_available() else "cpu"

# train_csv, val_csv = "outputs/index/train.csv", "outputs/index/val.csv"
# metadata = cfg.get("metadata_path", "data/CADICA/metadata.xlsx")

# sel = cfg.get("metadata_features", None)

# train_ds = CadicaDataset(train_csv, metadata, get_train_tf(cfg["img_size"]), selected_features=sel)
# val_ds   = CadicaDataset(val_csv,   metadata, get_val_tf(cfg["img_size"]), selected_features=sel)


# print("[meta] Using features:",
#       train_ds.get_feature_groups(),
#       f"tab_dim={train_ds.tab_dim}")
# print("[meta] Expanded order:", train_ds.get_feature_names())

cfg = yaml.safe_load(open("configs/default.yaml"))
set_seed(cfg["seed"])
device = "cuda" if torch.cuda.is_available() else "cpu"

train_csv, val_csv = "outputs/index/train.csv", "outputs/index/val.csv"
metadata = cfg.get("metadata_path", "data/CADICA/metadata.xlsx")

sel = cfg.get("metadata_features", {})
# split numeric/categorical out (unchanged), pick text list (normalized names already canonical in dataset.py)
sel_nc = {"numeric": sel.get("numeric"), "categorical": sel.get("categorical")}
text_cols = sel.get("text", None)  # e.g. ["another_comorb","indication","later_events"]

train_ds = CadicaDataset(
    train_csv,
    metadata,
    get_train_tf(cfg["img_size"]),
    selected_features=sel_nc,
    text_features=text_cols,
    text_model=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
    text_cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
)
val_ds = CadicaDataset(
    val_csv,
    metadata,
    get_val_tf(cfg["img_size"]),
    selected_features=sel_nc,
    text_features=text_cols,
    text_model=cfg.get("text_encoder", "sentence-transformers/all-MiniLM-L6-v2"),
    text_cache_dir=cfg.get("text_cache_dir", "outputs/text_embeds"),
)

print("[meta] Using features:", train_ds.get_feature_groups(), f"tab_dim={train_ds.tab_dim}")
print("[meta] Expanded order:", train_ds.get_feature_names())

tr = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
va = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, num_workers=4)

fusion = cfg["fusion"]
os.makedirs("outputs/checkpoints", exist_ok=True)

# A huge if-else hell, super ugly but it has to do for now
backbone = cfg.get("backbone", "resnet50").lower()

if fusion == "late":
    if backbone == "resnet50":
        from src.models.resnet_backbone import ResNetBackbone
        img_model = ResNetBackbone(pretrained=True).to(device)
        img_feat_dim = 2048
    elif backbone == "resnet18":
        from src.models.resnet18_backbone import ResNet18Backbone
        img_model = ResNet18Backbone(pretrained=True).to(device)
        img_feat_dim = img_model.feat_dim   # 512
    # elif backbone == "effnet_b3":
    #     from src.models.effnet_b3_backbone import EfficientNetB3Backbone
    #     img_model = EfficientNetB3Backbone(pretrained=True).to(device)
    #     img_feat_dim = img_model.feat_dim  # 1536
    elif backbone == "effnet_b0":
        from src.models.effnet_backbone import EfficientNetB0Backbone
        img_model = EfficientNetB0Backbone(pretrained=True).to(device)
        img_feat_dim = img_model.feat_dim  # 1280
    elif backbone == "inception_v3":
        from src.models.inception_backbone import InceptionV3Backbone
        img_model = InceptionV3Backbone(pretrained=True).to(device)
        img_feat_dim = img_model.feat_dim  # 2048
    elif backbone == "convnext_tiny":
        from src.models.convnext_backbone import ConvNeXtBackbone
        img_model = ConvNeXtBackbone(variant="tiny", pretrained=True).to(device)
        img_feat_dim = img_model.feat_dim  # 768
    elif backbone == "convnext_base":
        from src.models.convnext_backbone import ConvNeXtBackbone
        img_model = ConvNeXtBackbone(variant="base", pretrained=True).to(device)
        img_feat_dim = img_model.feat_dim  # 1024
    # elif backbone == "mobilenet_v3_large":
    #     from src.models.mobilenetv3_backbone import MobileNetV3LargeBackbone
    #     img_model = MobileNetV3LargeBackbone(pretrained=True).to(device)
    #     img_feat_dim = img_model.feat_dim  # 960
    elif backbone == "mobilenet_v2":
        from src.models.mobilenetv2_backbone import MobileNetV2Backbone
        img_model = MobileNetV2Backbone(pretrained=True).to(device)
        img_feat_dim = img_model.feat_dim  # 1280
    # elif backbone == "densenet201":
    #     from src.models.densenet_backbone import DenseNet201Backbone
    #     img_model = DenseNet201Backbone(pretrained=True).to(device)
    #     img_feat_dim = img_model.feat_dim  # 1920
    elif backbone == "densenet121":
        from src.models.densenet121_backbone import DenseNet121Backbone
        img_model = DenseNet121Backbone(pretrained=True).to(device)
        img_feat_dim = img_model.feat_dim  # 1024
    elif backbone == "alexnet":
        from src.models.alexnet_backbone import AlexNetBackbone
        img_model = AlexNetBackbone(pretrained=True).to(device)
        img_feat_dim = img_model.feat_dim  # 9216
    elif backbone == "vit_b16":
        from src.models.vit_backbone import ViTBackbone
        img_model = ViTBackbone(pretrained=True).to(device)
        img_feat_dim = img_model.feat_dim
    # elif backbone == "effnet_b7":
    #     from src.models.effnet_b7_backbone import EfficientNetB7Backbone
    #     img_model = EfficientNetB7Backbone(pretrained=True).to(device)
    #     img_feat_dim = img_model.feat_dim  # 2560
    elif backbone == "effnet_b0_cbam":
        from src.models.effnetb0_cbam_backbone import EfficientNetB0Backbone
        img_model = EfficientNetB0Backbone(pretrained=True, cbam=True).to(device)   # <-- enable CBAM
        img_feat_dim = img_model.feat_dim
    elif backbone == "convnext_base_cbam":
        from src.models.convnext_base_cbam_backbone import ConvNeXtBaseBackboneCBAM
        img_model = ConvNeXtBaseBackboneCBAM(pretrained=True).to(device)
        img_feat_dim = img_model.feat_dim  # 1024
    elif backbone == "metafusionnet_base":
        from src.models.metafusionnet_backbone import MetaFusionConvNeXtBackbone
        img_model = MetaFusionConvNeXtBackbone(variant="base", pretrained=True)  # no out_dim -> features only
        img_feat_dim = img_model.feat_dim  # 1024
    elif backbone == "metafusionnet_tiny":
        from src.models.metafusionnet_backbone import MetaFusionConvNeXtBackbone
        img_model = MetaFusionConvNeXtBackbone(variant="tiny", pretrained=True)  # no out_dim -> features only
        img_feat_dim = img_model.feat_dim  # 768

    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    from src.models.late_fusion import TabularMLP, LateFusionClassifier
    clin_mlp  = TabularMLP(in_dim=train_ds.tab_dim, hid=64, out_dim=32).to(device)
    head      = LateFusionClassifier(img_dim=img_feat_dim, clin_dim=32, out_dim=2).to(device)
    params = list(img_model.parameters()) + list(clin_mlp.parameters()) + list(head.parameters())

elif fusion == "early":
    if backbone == "resnet50":
        from src.models.early_fusion_patient import ResNetEarlyFusionPatient
        img_model = ResNetEarlyFusionPatient(tab_dim=train_ds.tab_dim, c_pat=cfg["pat_channels"],
                                             pretrained=True, out_dim=2).to(device)
    elif backbone == "resnet18":
        from src.models.resnet18_backbone import ResNet18EarlyFusionPatient
        img_model = ResNet18EarlyFusionPatient(
            tab_dim=train_ds.tab_dim, c_pat=cfg["pat_channels"], pretrained=True, out_dim=2
        ).to(device)
    # elif backbone == "effnet_b3":
    #     from src.models.effnet_b3_backbone import EfficientNetB3EarlyFusionPatient
    #     img_model = EfficientNetB3EarlyFusionPatient(
    #         tab_dim=train_ds.tab_dim, c_pat=cfg["pat_channels"],
    #         pretrained=True, out_dim=2
    #     ).to(device)
    elif backbone == "effnet_b0":
        from src.models.effnet_backbone import EfficientNetB0EarlyFusionPatient
        img_model = EfficientNetB0EarlyFusionPatient(tab_dim=train_ds.tab_dim, c_pat=cfg["pat_channels"],
                                                     pretrained=True, out_dim=2).to(device)
    elif backbone == "inception_v3":
        from src.models.inception_backbone import InceptionV3EarlyFusionPatient
        img_model = InceptionV3EarlyFusionPatient(tab_dim=train_ds.tab_dim, c_pat=cfg["pat_channels"],
                                                 pretrained=True, out_dim=2).to(device)
    elif backbone == "convnext_tiny":
        from src.models.convnext_backbone import ConvNeXtEarlyFusionPatient
        img_model = ConvNeXtEarlyFusionPatient(
            variant="tiny", tab_dim=train_ds.tab_dim, c_pat=cfg["pat_channels"],
            pretrained=True, out_dim=2
        ).to(device)
    elif backbone == "convnext_base":
        from src.models.convnext_backbone import ConvNeXtEarlyFusionPatient
        img_model = ConvNeXtEarlyFusionPatient(
            variant="base", tab_dim=train_ds.tab_dim, c_pat=cfg["pat_channels"],
            pretrained=True, out_dim=2
        ).to(device)
    # elif backbone == "mobilenet_v3_large":
    #     from src.models.mobilenetv3_backbone import MobileNetV3LargeEarlyFusionPatient
    #     img_model = MobileNetV3LargeEarlyFusionPatient(
    #         tab_dim=train_ds.tab_dim, c_pat=cfg["pat_channels"], pretrained=True, out_dim=2
    #     ).to(device)
    elif backbone == "mobilenet_v2":
        from src.models.mobilenetv2_backbone import MobileNetV2EarlyFusionPatient
        img_model = MobileNetV2EarlyFusionPatient(
            tab_dim=train_ds.tab_dim, c_pat=cfg["pat_channels"], pretrained=True, out_dim=2
        ).to(device)
    # elif backbone == "densenet201":
    #     from src.models.densenet_backbone import DenseNet201EarlyFusionPatient
    #     img_model = DenseNet201EarlyFusionPatient(
    #         tab_dim=train_ds.tab_dim, c_pat=cfg["pat_channels"], pretrained=True, out_dim=2
    #     ).to(device)
    elif backbone == "densenet121":
        from src.models.densenet121_backbone import DenseNet121EarlyFusionPatient
        img_model = DenseNet121EarlyFusionPatient(
            tab_dim=train_ds.tab_dim, c_pat=cfg["pat_channels"], pretrained=True, out_dim=2
        ).to(device)
    elif backbone == "alexnet":
        from src.models.alexnet_backbone import AlexNetEarlyFusionPatient
        img_model = AlexNetEarlyFusionPatient(
            tab_dim=train_ds.tab_dim, c_pat=cfg["pat_channels"], pretrained=True, out_dim=2
        ).to(device)
    elif backbone == "vit_b16":
        from src.models.vit_backbone import ViTEarlyFusionPatient
        img_model = ViTEarlyFusionPatient(
            tab_dim=train_ds.tab_dim, c_pat=cfg["pat_channels"], pretrained=True, out_dim=2
        ).to(device)
    # elif backbone == "effnet_b7":
    #     from src.models.effnet_b7_backbone import EfficientNetB7EarlyFusionPatient
    #     img_model = EfficientNetB7EarlyFusionPatient(
    #         tab_dim=train_ds.tab_dim, c_pat=cfg["pat_channels"],
    #         pretrained=True, out_dim=2
    #     ).to(device)
    elif backbone == "effnet_b0_cbam":
        from src.models.effnetb0_cbam_backbone import EfficientNetB0EarlyFusionPatient
        img_model = EfficientNetB0EarlyFusionPatient(
            tab_dim=train_ds.tab_dim,
            c_pat=cfg["pat_channels"],
            pretrained=True,
            out_dim=2,
            cbam=True,                         # <-- enable CBAM
        ).to(device)
    elif backbone == "convnext_base_cbam":
        from src.models.convnext_base_cbam_backbone import ConvNeXtBaseEarlyFusionPatientCBAM
        img_model = ConvNeXtBaseEarlyFusionPatientCBAM(
            tab_dim=train_ds.tab_dim,
            c_pat=cfg["pat_channels"],
            pretrained=True,
            out_dim=2
        ).to(device)
    elif backbone == "metafusionnet_base":
        from src.models.metafusionnet_backbone import MetaFusionNetEarlyFusionPatient
        img_model = MetaFusionNetEarlyFusionPatient(
            variant="base",
            tab_dim=train_ds.tab_dim,
            pretrained=True,
            out_dim=2
        ).to(device)

    elif backbone == "metafusionnet_tiny":
        from src.models.metafusionnet_backbone import MetaFusionNetEarlyFusionPatient
        img_model = MetaFusionNetEarlyFusionPatient(
            variant="tiny",
            tab_dim=train_ds.tab_dim,
            pretrained=True,
            out_dim=2
        ).to(device)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    clin_mlp = head = None
    params = list(img_model.parameters())

elif fusion == "imageonly":
    if backbone == "resnet50":
        from src.models.resnet_backbone import ResNetBackbone
        img_model = ResNetBackbone(pretrained=True, out_dim=2).to(device)
    elif backbone == "resnet18":
        from src.models.resnet18_backbone import ResNet18Backbone
        img_model = ResNet18Backbone(pretrained=True, out_dim=2).to(device)
    # elif backbone == "effnet_b3":
        # from src.models.effnet_b3_backbone import EfficientNetB3Backbone
        # img_model = EfficientNetB3Backbone(pretrained=True, out_dim=2).to(device)
    elif backbone == "effnet_b0":
        from src.models.effnet_backbone import EfficientNetB0Backbone
        img_model = EfficientNetB0Backbone(pretrained=True, out_dim=2).to(device)
    elif backbone == "inception_v3":
        from src.models.inception_backbone import InceptionV3Backbone
        img_model = InceptionV3Backbone(pretrained=True, out_dim=2).to(device)
    elif backbone == "convnext_tiny":
        from src.models.convnext_backbone import ConvNeXtBackbone
        img_model = ConvNeXtBackbone(variant="tiny", pretrained=True, out_dim=2).to(device)
    elif backbone == "convnext_base":
        from src.models.convnext_backbone import ConvNeXtBackbone
        img_model = ConvNeXtBackbone(variant="base", pretrained=True, out_dim=2).to(device)
    # elif backbone == "mobilenet_v3_large":
    #     from src.models.mobilenetv3_backbone import MobileNetV3LargeBackbone
    #     img_model = MobileNetV3LargeBackbone(pretrained=True, out_dim=2).to(device)
    elif backbone == "mobilenet_v2":
        from src.models.mobilenetv2_backbone import MobileNetV2Backbone
        img_model = MobileNetV2Backbone(pretrained=True, out_dim=2).to(device)
    # elif backbone == "densenet201":
    #     from src.models.densenet_backbone import DenseNet201Backbone
    #     img_model = DenseNet201Backbone(pretrained=True, out_dim=2).to(device)
    elif backbone == "densenet121":
        from src.models.densenet121_backbone import DenseNet121Backbone
        img_model = DenseNet121Backbone(pretrained=True, out_dim=2).to(device)
    elif backbone == "alexnet":
        from src.models.alexnet_backbone import AlexNetBackbone
        img_model = AlexNetBackbone(pretrained=True, out_dim=2).to(device)
    elif backbone == "vit_b16":
        from src.models.vit_backbone import ViTBackbone
        img_model = ViTBackbone(pretrained=True, out_dim=2).to(device)
    # elif backbone == "effnet_b7":
    #     from src.models.effnet_b7_backbone import EfficientNetB7Backbone
    #     img_model = EfficientNetB7Backbone(pretrained=True, out_dim=2).to(device)
    elif backbone == "effnet_b0_cbam":
        from src.models.effnetb0_cbam_backbone import EfficientNetB0Backbone
        img_model = EfficientNetB0Backbone(pretrained=True, out_dim=2, cbam=True).to(device)
    elif backbone == "convnext_base_cbam":
        from src.models.convnext_base_cbam_backbone import ConvNeXtBaseBackboneCBAM
        img_model = ConvNeXtBaseBackboneCBAM(pretrained=True, out_dim=2).to(device)
    elif backbone == "metafusionnet_base":
        from src.models.metafusionnet_backbone import MetaFusionConvNeXtBackbone
        img_model = MetaFusionConvNeXtBackbone(variant="base", pretrained=True, out_dim=2).to(device)
    elif backbone == "metafusionnet_tiny":
        from src.models.metafusionnet_backbone import MetaFusionConvNeXtBackbone
        img_model = MetaFusionConvNeXtBackbone(variant="tiny", pretrained=True, out_dim=2).to(device)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    clin_mlp = head = None
    params = list(img_model.parameters())

else:
    raise ValueError("configs/default.yaml: fusion must be 'late', 'early', or 'imageonly'")

opt_name = cfg["optimizer"].lower()
if opt_name == "rmsprop":
    optimizer = torch.optim.RMSprop(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])
elif opt_name in ("sgd","sgdm"):
    optimizer = torch.optim.SGD(params, lr=cfg["lr"], momentum=0.9, weight_decay=cfg["weight_decay"])
else:
    optimizer = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"])

ROOT_DIR = f"outputs/checkpoints/all-{backbone}"
os.makedirs(ROOT_DIR, exist_ok=True)
best = 0.0; ckpt_path = f"{ROOT_DIR}/best_{backbone}_{fusion}.pt"
patience = cfg.get("early_stopping_patience", None)
epochs_no_improve = 0

for epoch in range(cfg["epochs"]):
    print(f"\nEpoch {epoch+1}/{cfg['epochs']}")

    train_one_epoch(img_model, optimizer, tr, device,
                    mode=fusion, clin_mlp=clin_mlp, late_head=head)

    metrics, y, p = evaluate(img_model, va, device,
                             mode=fusion, clin_mlp=clin_mlp, late_head=head)
    print("=== Evaluation metrics ===")
    for k,v in metrics.items():
        print(f"{k:10s}: {v:.4f}")

    if metrics["bal_acc"] > best:
        best = metrics["bal_acc"]
        # torch.save({
        #     "fusion": fusion,
        #     "tab_dim": train_ds.tab_dim,
        #     "cfg": cfg,
        #     "model": img_model.state_dict(),
        #     "clin_mlp": clin_mlp.state_dict() if clin_mlp else None,
        #     "head": head.state_dict() if head else None,
        # }, ckpt_path)

        torch.save({
            "fusion": fusion,
            "tab_dim": train_ds.tab_dim,
            "cfg": {
                **cfg,
                "metadata_features_used": {
                    **train_ds.get_feature_groups(),          # numeric/categorical subsets actually used
                    "text": text_cols,                         # text cols used
                },
            },
            "model": img_model.state_dict(),
            "clin_mlp": clin_mlp.state_dict() if clin_mlp else None,
            "head": head.state_dict() if head else None,
        }, ckpt_path)
        print(f"[ckpt] new best saved ({best:.3f})")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"[early stopping] no improvement for {epochs_no_improve} epochs")

    # check patience
    if patience and epochs_no_improve >= patience:
        print(f"[early stopping] stopping early after {epoch+1} epochs")
        break

print(f"Best val balanced acc: {best:.3f} -> {ckpt_path}")