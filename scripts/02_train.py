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

cfg = yaml.safe_load(open("configs/default.yaml"))
set_seed(cfg["seed"])
device = "cuda" if torch.cuda.is_available() else "cpu"

train_csv, val_csv = "outputs/index/train.csv", "outputs/index/val.csv"
metadata = cfg.get("metadata_path", "data/CADICA/metadata.xlsx")

# sel = cfg.get("metadata_features", None)

# train_ds = CadicaDataset(train_csv, metadata, get_train_tf(cfg["img_size"]), selected_features=sel)
# val_ds   = CadicaDataset(val_csv,   metadata, get_val_tf(cfg["img_size"]), selected_features=sel)

# train_ds = CadicaDataset(train_csv, metadata, get_train_tf(cfg["img_size"]))
# val_ds   = CadicaDataset(val_csv,   metadata, get_val_tf(cfg["img_size"]))

sel = cfg.get("metadata_features", None)

train_ds = CadicaDataset(train_csv, metadata, get_train_tf(cfg["img_size"]), selected_features=sel)
val_ds   = CadicaDataset(val_csv,   metadata, get_val_tf(cfg["img_size"]), selected_features=sel)


print("[meta] Using features:",
      train_ds.get_feature_groups(),
      f"tab_dim={train_ds.tab_dim}")
print("[meta] Expanded order:", train_ds.get_feature_names())


tr = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=4, pin_memory=True)
va = DataLoader(val_ds,   batch_size=cfg["batch_size"], shuffle=False, num_workers=4)

fusion = cfg["fusion"]
os.makedirs("outputs/checkpoints", exist_ok=True)


# if fusion == "late":
#     img_model = ResNetBackbone(pretrained=True).to(device)
#     clin_mlp  = TabularMLP(in_dim=train_ds.tab_dim, hid=64, out_dim=32).to(device)
#     head      = LateFusionClassifier(img_dim=2048, clin_dim=32, out_dim=2).to(device)
#     params = list(img_model.parameters()) + list(clin_mlp.parameters()) + list(head.parameters())
# elif fusion == "early":
#     img_model = ResNetEarlyFusionPatient(tab_dim=train_ds.tab_dim, c_pat=cfg["pat_channels"], pretrained=True, out_dim=2).to(device)
#     clin_mlp = head = None
#     params = list(img_model.parameters())
# elif fusion == "imageonly":
#     img_model = ResNetBackbone(pretrained=True, out_dim=2).to(device)
#     clin_mlp = head = None
#     params = list(img_model.parameters())
# else:
#     raise ValueError("configs/default.yaml: fusion must be 'late', 'early' or 'imageonly'")

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
    elif backbone == "effnet_b0":
        from src.models.effnet_backbone import EfficientNetB0Backbone
        img_model = EfficientNetB0Backbone(pretrained=True).to(device)
        img_feat_dim = img_model.feat_dim  # 1280
    elif backbone == "inception_v3":
        from src.models.inception_backbone import InceptionV3Backbone
        img_model = InceptionV3Backbone(pretrained=True).to(device)
        img_feat_dim = img_model.feat_dim  # 2048
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
    elif backbone == "effnet_b0":
        from src.models.effnet_backbone import EfficientNetB0EarlyFusionPatient
        img_model = EfficientNetB0EarlyFusionPatient(tab_dim=train_ds.tab_dim, c_pat=cfg["pat_channels"],
                                                     pretrained=True, out_dim=2).to(device)
    elif backbone == "inception_v3":
        from src.models.inception_backbone import InceptionV3EarlyFusionPatient
        img_model = InceptionV3EarlyFusionPatient(tab_dim=train_ds.tab_dim, c_pat=cfg["pat_channels"],
                                                 pretrained=True, out_dim=2).to(device)
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
    elif backbone == "effnet_b0":
        from src.models.effnet_backbone import EfficientNetB0Backbone
        img_model = EfficientNetB0Backbone(pretrained=True, out_dim=2).to(device)
    elif backbone == "inception_v3":
        from src.models.inception_backbone import InceptionV3Backbone
        img_model = InceptionV3Backbone(pretrained=True, out_dim=2).to(device)
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

best = 0.0; ckpt_path = f"outputs/checkpoints/best_{fusion}.pt"
# for epoch in range(cfg["epochs"]):
#     train_one_epoch(img_model, optimizer, tr, device, mode=fusion, clin_mlp=clin_mlp, late_head=head)
#     metrics, y, p = evaluate(img_model, va, device, mode=fusion, clin_mlp=clin_mlp, late_head=head)
#     print(f"[{fusion}] Epoch {epoch}: {metrics}")
#     if metrics["bal_acc"] > best:
#         best = metrics["bal_acc"]
#         torch.save({
#             "fusion": fusion,
#             "model": img_model.state_dict(),
#             "clin_mlp": None if clin_mlp is None else clin_mlp.state_dict(),
#             "head": None if head is None else head.state_dict(),
#             "tab_dim": train_ds.tab_dim,
#             "cfg": cfg
#         }, ckpt_path)
# print("Best val balanced acc:", best, "->", ckpt_path)


patience = cfg.get("early_stopping_patience", None)
# best = 0.0
epochs_no_improve = 0

for epoch in range(cfg["epochs"]):
    print(f"\nEpoch {epoch+1}/{cfg['epochs']}")

    train_one_epoch(img_model, optimizer, tr, device,
                    mode=fusion, clin_mlp=clin_mlp, late_head=head)

    # metrics, y, p = evaluate(img_model, va, device,
    #                          mode=fusion, clin_mlp=clin_mlp, late_head=head)
    # print("val metrics:", metrics)


    metrics, y, p = evaluate(img_model, va, device,
                             mode=fusion, clin_mlp=clin_mlp, late_head=head)
    print("=== Evaluation metrics ===")
    for k,v in metrics.items():
        print(f"{k:10s}: {v:.4f}")

    if metrics["bal_acc"] > best:
        best = metrics["bal_acc"]
        torch.save({
            "fusion": fusion,
            "tab_dim": train_ds.tab_dim,
            "cfg": cfg,
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