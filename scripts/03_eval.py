import torch, pandas as pd, numpy as np, yaml, os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from torch.utils.data import DataLoader
from src.dataio.transforms import get_val_tf
from src.dataio.dataset import CadicaDataset
from src.models.resnet_backbone import ResNetBackbone
from src.models.late_fusion import TabularMLP, LateFusionClassifier
from src.models.early_fusion_patient import ResNetEarlyFusionPatient
from src.trainloop.engine import evaluate
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

# def main(ckpt):
#     ck = torch.load(ckpt, map_location="cpu")
#     fusion = ck["fusion"]; tab_dim = ck["tab_dim"]; cfg = ck["cfg"]
#     device = "cuda" if torch.cuda.is_available() else "cpu"

#     te_csv = "outputs/index/test.csv"
#     ds = CadicaDataset(te_csv, cfg.get("metadata_path","data/CADICA/metadata.xlsx"), get_val_tf(cfg["img_size"]))
#     ld = DataLoader(ds, batch_size=32, shuffle=False, num_workers=4)
    

#     if fusion == "late":
#         model = ResNetBackbone(pretrained=False).to(device)
#         clin_mlp = TabularMLP(in_dim=tab_dim, hid=64, out_dim=32).to(device)
#         head = LateFusionClassifier(img_dim=2048, clin_dim=32, out_dim=2).to(device)
#         model.load_state_dict(ck["model"]); clin_mlp.load_state_dict(ck["clin_mlp"]); head.load_state_dict(ck["head"])
#     elif fusion == "early":
#         model = ResNetEarlyFusionPatient(tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2).to(device)
#         clin_mlp = head = None
#         model.load_state_dict(ck["model"])
#     elif fusion == "imageonly":
#         model = ResNetBackbone(pretrained=False, out_dim=2).to(device)
#         model.load_state_dict(ck["model"])
#         clin_mlp = head = None
#     else:
#         raise ValueError("configs/default.yaml: fusion must be 'late', 'early' or 'imageonly'")

#     metrics, y, p = evaluate(model, ld, device, mode=fusion, clin_mlp=clin_mlp, late_head=head)
#     print("Frame-level:", metrics)

#     df = pd.read_csv(te_csv); df["prob"]=p; df["label"]=y
#     agg = df.groupby(["patient","video"]).agg(prob=("prob","max"), label=("label","max")).reset_index()
#     y_pred = (agg.prob>=0.5).astype(int)
#     print("Video-level:", dict(
#         bal_acc=balanced_accuracy_score(agg.label, y_pred),
#         f1=f1_score(agg.label, y_pred),
#         auc=roc_auc_score(agg.label, agg.prob)
#     ))
#     agg.to_csv("outputs/test_video_probs.csv", index=False)

# if __name__ == "__main__":
#     import argparse
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--ckpt", required=True)
#     args = ap.parse_args()
#     main(args.ckpt)


def main(ckpt):
    import torch, pandas as pd, numpy as np
    from torch.utils.data import DataLoader
    from src.dataio.transforms import get_val_tf
    from src.dataio.dataset import CadicaDataset
    from src.models.resnet_backbone import ResNetBackbone
    from src.models.late_fusion import TabularMLP, LateFusionClassifier
    from src.models.early_fusion_patient import ResNetEarlyFusionPatient
    from src.trainloop.engine import evaluate
    from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

    ck = torch.load(ckpt, map_location="cpu")
    fusion = ck["fusion"]
    tab_dim = ck["tab_dim"]
    cfg = ck["cfg"]  # this contains the exact metadata_features used for training
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- dataset: use the SAME metadata subset as training ---
    te_csv = "outputs/index/test.csv"
    metadata_path = cfg.get("metadata_path", "data/CADICA/metadata.xlsx")
    selected = cfg.get("metadata_features", None)  # <- ablation subset used in training

    ds = CadicaDataset(
        te_csv,
        metadata_path,
        get_val_tf(cfg["img_size"]),
        selected_features=selected
    )


    print("[meta] (TEST) Using features:",
      ds.get_feature_groups(),
      f"tab_dim={ds.tab_dim}")
    print("[meta] (TEST) Expanded order:", ds.get_feature_names())

    # you can lower workers if your system warned earlier
    ld = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2)

    # --- rebuild model(s) matching the fusion mode and load weights ---
    # if fusion == "late":
    #     model = ResNetBackbone(pretrained=False).to(device)
    #     clin_mlp = TabularMLP(in_dim=tab_dim, hid=64, out_dim=32).to(device)
    #     head = LateFusionClassifier(img_dim=2048, clin_dim=32, out_dim=2).to(device)
    #     model.load_state_dict(ck["model"])
    #     clin_mlp.load_state_dict(ck["clin_mlp"])
    #     head.load_state_dict(ck["head"])

    # elif fusion == "early":
    #     model = ResNetEarlyFusionPatient(
    #         tab_dim=tab_dim,
    #         c_pat=cfg["pat_channels"],
    #         pretrained=False,
    #         out_dim=2
    #     ).to(device)
    #     clin_mlp = head = None
    #     model.load_state_dict(ck["model"])

    # elif fusion == "imageonly":
    #     model = ResNetBackbone(pretrained=False, out_dim=2).to(device)
    #     clin_mlp = head = None
    #     model.load_state_dict(ck["model"])

    # else:
    #     raise ValueError(f"Unknown fusion mode in checkpoint: {fusion}")

    backbone = cfg.get("backbone", "resnet50").lower()

    if fusion == "late":
        if backbone == "resnet50":
            from src.models.resnet_backbone import ResNetBackbone
            model = ResNetBackbone(pretrained=False).to(device)
            img_feat_dim = 2048
        elif backbone == "resnet18":
            from src.models.resnet18_backbone import ResNet18Backbone
            model = ResNet18Backbone(pretrained=False).to(device)
            img_feat_dim = model.feat_dim  # 512
        elif backbone == "effnet_b0":
            from src.models.effnet_backbone import EfficientNetB0Backbone
            model = EfficientNetB0Backbone(pretrained=False).to(device)
            img_feat_dim = model.feat_dim  # 1280
        elif backbone == "inception_v3":
            from src.models.inception_backbone import InceptionV3Backbone
            model = InceptionV3Backbone(pretrained=False).to(device)
            img_feat_dim = model.feat_dim  # 2048
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        from src.models.late_fusion import TabularMLP, LateFusionClassifier
        clin_mlp = TabularMLP(in_dim=tab_dim, hid=64, out_dim=32).to(device)
        head = LateFusionClassifier(img_dim=img_feat_dim, clin_dim=32, out_dim=2).to(device)
        model.load_state_dict(ck["model"]); clin_mlp.load_state_dict(ck["clin_mlp"]); head.load_state_dict(ck["head"])

    elif fusion == "early":
        if backbone == "resnet50":
            from src.models.early_fusion_patient import ResNetEarlyFusionPatient
            model = ResNetEarlyFusionPatient(tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2).to(device)
        elif backbone == "resnet18":
            from src.models.resnet18_backbone import ResNet18EarlyFusionPatient
            model = ResNet18EarlyFusionPatient(tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2).to(device)
        elif backbone == "effnet_b0":
            from src.models.effnet_backbone import EfficientNetB0EarlyFusionPatient
            model = EfficientNetB0EarlyFusionPatient(tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2).to(device)
        elif backbone == "inception_v3":
            from src.models.inception_backbone import InceptionV3EarlyFusionPatient
            model = InceptionV3EarlyFusionPatient(tab_dim=tab_dim, c_pat=cfg["pat_channels"], pretrained=False, out_dim=2).to(device)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        clin_mlp = head = None
        model.load_state_dict(ck["model"])

    elif fusion == "imageonly":
        if backbone == "resnet50":
            from src.models.resnet_backbone import ResNetBackbone
            model = ResNetBackbone(pretrained=False, out_dim=2).to(device)
        elif backbone == "resnet18":
            from src.models.resnet18_backbone import ResNet18Backbone
            model = ResNet18Backbone(pretrained=False, out_dim=2).to(device)
        elif backbone == "effnet_b0":
            from src.models.effnet_backbone import EfficientNetB0Backbone
            model = EfficientNetB0Backbone(pretrained=False, out_dim=2).to(device)
        elif backbone == "inception_v3":
            from src.models.inception_backbone import InceptionV3Backbone
            model = InceptionV3Backbone(pretrained=False, out_dim=2).to(device)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        clin_mlp = head = None
        model.load_state_dict(ck["model"])

    else:
        raise ValueError("Unknown fusion mode in checkpoint")
    # --- evaluate on TEST set ---
    # metrics, y, p = evaluate(
    #     model, ld, device,
    #     mode=fusion,
    #     clin_mlp=clin_mlp,
    #     late_head=head
    # )
    # print("Frame-level:", metrics)


    frame_metrics, y_frame, p_frame, video_metrics, video_df = evaluate(
    model, ld, device, mode=fusion, clin_mlp=clin_mlp, late_head=head,
    also_video=True, video_reduce="max"   # "mean" or "max"
)

    print("=== Frame-level metrics ===")
    for k, v in frame_metrics.items():
        print(f"{k:10s}: {v:.4f}")

    print("\n=== Video-level metrics ===")
    for k, v in video_metrics.items():
        print(f"{k:10s}: {v:.4f}")

    # optional: save the per-video probabilities/labels
    # video_df.to_csv("outputs/test_video_probs.csv", index=False)

    # --- aggregate to video-level (max over frames) ---
    # df = pd.read_csv(te_csv)
    # df["prob"] = p
    # df["label"] = y
    # agg = df.groupby(["patient", "video"]).agg(prob=("prob", "max"), label=("label", "max")).reset_index()
    # y_pred = (agg.prob >= 0.5).astype(int)

    # vid_metrics = dict(
    #     bal_acc=balanced_accuracy_score(agg.label, y_pred),
    #     f1=f1_score(agg.label, y_pred),
    #     auc=roc_auc_score(agg.label, agg.prob)
    # )


    # print("\n=== Video-level metrics ===")
    # for k, v in video_metrics.items():
    #     print(f"{k:10s}: {v:.4f}")
    # print("Video-level:", vid_metrics)

    # agg.to_csv("outputs/test_video_probs.csv", index=False)

    # helpful print so you know exactly what subset was used
    if selected:
        n_num = len(selected.get("numeric", []))
        n_cat = len(selected.get("categorical", []))
        print(f"[info] evaluated with metadata subset: numeric={n_num}, categorical={n_cat}, tab_dim(checkpoint)={tab_dim}")
    else:
        print(f"[info] evaluated with ALL metadata features, tab_dim(checkpoint)={tab_dim}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    args = ap.parse_args()
    main(args.ckpt)