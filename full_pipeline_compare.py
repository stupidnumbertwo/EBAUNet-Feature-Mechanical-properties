import os
import cv2
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor

from my_dataset import DriveDataset
from train import get_transform
from src import UNet, EBAUNet

# ===================== 配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

UNET_WEIGHT = "save_weights/UNet_model.pth"
EBA_WEIGHT  = "save_weights/EBA_UNet_model.pth"

MASK_UNET_DIR = "mask_unet"
MASK_EBA_DIR  = "mask_eba"

Y_PATH = "labels.csv"   # CSV: sample_id, YS

# ===================== 加载模型 =====================
def load_model(model_class, weight_path):
    model = model_class(in_channels=1, num_classes=2, base_c=32)

    checkpoint = torch.load(weight_path, map_location=DEVICE, weights_only=False)

    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()
    return model

# ===================== 推理 =====================
def infer_dataset(model, dataset, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    for i in tqdm(range(len(dataset)), desc=f"Infer {save_dir}"):
        img, _ = dataset[i]
        img = img.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(img)
            pred = output["out"]             # ✅ 修复 dict 问题
            pred = torch.softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1)

        mask = pred.squeeze().cpu().numpy().astype(np.uint8) * 255
        cv2.imwrite(os.path.join(save_dir, f"{i}.png"), mask)

# ===================== 特征提取 =====================
def extract_features(image_path):

    img = cv2.imread(image_path, 0)

    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # ✅ 降低阈值避免丢数据
    valid = np.where(stats[:, cv2.CC_STAT_AREA] >= 50)[0]
    valid = valid[valid != 0]

    if len(valid) == 0:
        return [0, 0, 0, 0]

    areas = stats[valid, cv2.CC_STAT_AREA]

    radii = np.sqrt(areas / np.pi)

    mean_r = np.mean(radii)
    max_r  = np.max(radii)
    min_r  = np.min(radii)

    ferrite_fraction = np.sum(areas) / (img.shape[0] * img.shape[1])

    return [mean_r, max_r, min_r, ferrite_fraction]

# ===================== 构建特征矩阵 =====================
def build_feature_matrix(mask_dir):
    files = sorted(os.listdir(mask_dir), key=lambda x: int(x.split('.')[0]))

    X = []

    for f in tqdm(files, desc=f"Extract {mask_dir}"):
        feat = extract_features(os.path.join(mask_dir, f))
        X.append(feat)

    return np.array(X)

# ===================== LOOCV评估 =====================
def evaluate(X, y):
    loo = LeaveOneOut()

    preds = []
    trues = []

    for train_idx, test_idx in loo.split(X):
        model = XGBRegressor(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1
        )

        model.fit(X[train_idx], y[train_idx])
        pred = model.predict(X[test_idx])

        preds.append(pred[0])
        trues.append(y[test_idx][0])

    r2 = r2_score(trues, preds)
    rmse = np.sqrt(mean_squared_error(trues, preds))

    return r2, rmse, np.array(trues), np.array(preds)

# ===================== 主流程 =====================
def main():

    # ===== 数据 =====
    dataset = DriveDataset(
        root="./",
        train=False,
        transforms=get_transform(train=False)
    )

    # ===== 模型 =====
    unet = load_model(UNet, UNET_WEIGHT)
    eba  = load_model(EBAUNet, EBA_WEIGHT)

    # ===== 推理 =====
    infer_dataset(unet, dataset, MASK_UNET_DIR)
    infer_dataset(eba, dataset, MASK_EBA_DIR)

    # ===== 特征 =====
    X_unet = build_feature_matrix(MASK_UNET_DIR)
    X_eba  = build_feature_matrix(MASK_EBA_DIR)

    # ===== 标签 =====
    df = pd.read_csv(Y_PATH)
    y = df["YS"].values   # ✅ 一维

    # ===== 检查 =====
    print("X_unet:", X_unet.shape)
    print("X_eba :", X_eba.shape)
    print("y     :", y.shape)

    # ===== 评估 =====
    r2_u, rmse_u, y_true_u, y_pred_u = evaluate(X_unet, y)
    r2_e, rmse_e, y_true_e, y_pred_e = evaluate(X_eba, y)

    print("\n===== FINAL RESULT =====")
    print(f"U-Net   → R2: {r2_u:.4f}, RMSE: {rmse_u:.4f}")
    print(f"EBAUNet → R2: {r2_e:.4f}, RMSE: {rmse_e:.4f}")

    # ===== 图1：R2对比 =====
    plt.figure()
    plt.bar(["U-Net", "EBAUNet"], [r2_u, r2_e])
    plt.ylabel("R²")
    plt.title("Segmentation Impact on Prediction")
    plt.savefig("r2_compare.png", dpi=300)

    # ===== 图2：预测散点 =====
    plt.figure()
    plt.scatter(y_true_u, y_pred_u, label="U-Net")
    plt.scatter(y_true_e, y_pred_e, label="EBAUNet")

    plt.plot([min(y), max(y)], [min(y), max(y)], linestyle='--')

    plt.xlabel("True YS")
    plt.ylabel("Predicted YS")
    plt.legend()
    plt.savefig("scatter_compare.png", dpi=300)

    plt.show()

# ===================== 入口 =====================
if __name__ == "__main__":
    main()