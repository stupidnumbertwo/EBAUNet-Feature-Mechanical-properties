import os
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image, ImageEnhance
from src import UNet, EBAUNet
from boundary_metrics import boundary_iou, boundary_f1
from torch.serialization import add_safe_globals
import argparse

def load_mask_as_numpy(mask_path, target_shape):
    """
    使用cv2加载红色掩膜，并resize为target_shape(H, W)，确保尺寸和阈值正确
    """
    import cv2
    import numpy as np

    mask_bgr = cv2.imread(mask_path)
    if mask_bgr is None:
        raise FileNotFoundError(f"Could not read mask image at {mask_path}")

    mask_rgb = cv2.cvtColor(mask_bgr, cv2.COLOR_BGR2RGB)

    r = mask_rgb[:, :, 0]
    g = mask_rgb[:, :, 1]
    b = mask_rgb[:, :, 2]

    #宽松阈值
    red_mask = (r > 80) & (g < 80) & (b < 80)

    resized = cv2.resize(
        red_mask.astype(np.uint8),
        (target_shape[1], target_shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    return resized.astype(bool)


def tensor_to_bool_np(mask_tensor):
    """
    把 torch tensor 转 bool numpy
    """
    mask_np = mask_tensor.cpu().numpy()
    return mask_np.astype(bool)


def load_image(path, mean=(0.5,), std=(0.5,)):
    img = Image.open(path).convert('L')
    # 增加对比度增强
    enhancer = ImageEnhance.Contrast(img)
    enhanced_img = enhancer.enhance(1.5)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform(enhanced_img).unsqueeze(0)


def predict_mask(model, image, device):
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        logits = outputs["out"]
        pred = torch.softmax(logits, dim=1)
        pred_mask = pred.argmax(dim=1).float()
    return pred_mask


import cv2
import numpy as np


def postprocess_mask(mask_tensor, kernel_size=5, iterations=1):
    """
    mask_tensor: torch.Tensor [H, W] 0/1
    returns: processed torch.Tensor [H, W]
    """
    mask_np = (mask_tensor.cpu().numpy() * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    processed = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel, iterations=iterations)
    processed = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    processed = (processed > 127).astype(np.float32)

    return torch.from_numpy(processed)


def visualize_compare(image, mask_unet, mask_ebaunet, save_dir=None, name="compare_output"):
    image = image.squeeze().cpu()
    mask_unet = mask_unet.squeeze().cpu()
    mask_ebaunet = mask_ebaunet.squeeze().cpu()

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image, cmap='gray')
    axs[0].set_title('Input Image')
    axs[1].imshow(mask_unet, cmap='gray')
    axs[1].set_title('Original UNet Prediction')
    axs[2].imshow(mask_ebaunet, cmap='gray')
    axs[2].set_title('EBA_UNet Prediction')

    for ax in axs:
        ax.axis('off')

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{name}.png")
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


def visualize_model_comparison(image, gt_mask, pred_unet, pred_ebaunet,save_dir=None, name="model_comparison",grid_figsize=(18, 10), overlay_figsize=(18, 6),
    wspace=0.02, hspace=0.02):
    """可视化两个模型的预测结果与GT的对比"""
    image = image.squeeze().cpu()

    # ===== 第一组图：Grid =====
    fig, axs = plt.subplots(2, 3, figsize=grid_figsize)

    axs[0, 0].imshow(image, cmap='gray')
    axs[0, 0].set_title('Input Image')

    axs[0, 1].imshow(gt_mask, cmap='gray')
    axs[0, 1].set_title('Ground Truth')

    axs[0, 2].imshow(pred_unet, cmap='gray')
    axs[0, 2].set_title('UNet Prediction')

    axs[1, 0].imshow(image, cmap='gray')
    axs[1, 0].set_title('Input Image')

    axs[1, 1].imshow(gt_mask, cmap='gray')
    axs[1, 1].set_title('Ground Truth')

    axs[1, 2].imshow(pred_ebaunet, cmap='gray')
    axs[1, 2].set_title('EBAUNet Prediction')

    for ax in axs.flatten():
        ax.axis('off')
    # 调整子图之间的宽高间距
    plt.subplots_adjust(wspace=wspace, hspace=hspace)

    # ===== 第二组图：Overlay =====
    fig2, axs2 = plt.subplots(1, 3, figsize=overlay_figsize)

    axs2[0].imshow(image, cmap='gray')
    axs2[0].imshow(gt_mask, alpha=0.5, cmap='Blues')
    axs2[0].set_title('Ground Truth Overlay')

    axs2[1].imshow(image, cmap='gray')
    axs2[1].imshow(pred_unet, alpha=0.5, cmap='Reds')
    axs2[1].set_title('UNet Prediction Overlay')

    axs2[2].imshow(image, cmap='gray')
    axs2[2].imshow(pred_ebaunet, alpha=0.5, cmap='Greens')
    axs2[2].set_title('EBAUNet Prediction Overlay')

    for ax in axs2.flatten():
        ax.axis('off')

    plt.subplots_adjust(wspace=wspace, hspace=hspace)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path1 = os.path.join(save_dir, f"{name}_grid.png")
        save_path2 = os.path.join(save_dir, f"{name}_overlay.png")

        fig.savefig(save_path1, bbox_inches='tight')
        fig2.savefig(save_path2, bbox_inches='tight')
    else:
        plt.show()

    plt.close(fig)
    plt.close(fig2)


def load_model(weights_path, device):
    try:
        add_safe_globals([argparse.Namespace])
        model = EBAUNet(in_channels=1, num_classes=2, base_c=32)
        weights = torch.load(weights_path, map_location='cpu', weights_only=True)
        if isinstance(weights, dict) and'model' in weights:
            weights = weights['model']
        model.load_state_dict(weights)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def calculate_iou(pred, gt):
    """计算普通IoU (交并比)"""
    intersection = np.logical_and(pred, gt)
    union = np.logical_or(pred, gt)
    iou_score = np.sum(intersection) / np.sum(union) if np.sum(union) > 0 else 0
    return iou_score


def calculate_f1(pred, gt):
    """计算普通F1分数"""
    intersection = np.logical_and(pred, gt)
    precision = np.sum(intersection) / np.sum(pred) if np.sum(pred) > 0 else 0
    recall = np.sum(intersection) / np.sum(gt) if np.sum(gt) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return f1_score


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 路径配置
    model_path_unet = r"save_weights\UNet_model.pth"
    model_path_ebaunet = r"save_weights\EBA_UNet_model.pth"
    image_path = r"images/hongjd-4.jpg"
    gt_mask_path = r"masks/hongjd-4.png"
    save_dir = r"vis_compare"

    # 加载模型
    model_unet = UNet(in_channels=1, num_classes=2, base_c=32).to(device)
    model_ebaunet = load_model(model_path_ebaunet, device)

    # 加载权重
    ckpt_unet = torch.load(model_path_unet, map_location=device, weights_only=False)
    model_unet.load_state_dict(ckpt_unet["model"])

    # 加载图片
    image = load_image(image_path).to(device)

    # 分别推理
    mask_unet = predict_mask(model_unet, image, device)
    mask_ebaunet = predict_mask(model_ebaunet, image, device)

    # 后处理
    mask_unet_post = postprocess_mask(mask_unet.squeeze())
    mask_ebaunet_post = postprocess_mask(mask_ebaunet.squeeze())

    # 转成bool numpy
    pred_unet_np = tensor_to_bool_np(mask_unet_post)
    pred_ebaunet_np = tensor_to_bool_np(mask_ebaunet_post)

    # 加载GT mask
    gt_mask = load_mask_as_numpy(gt_mask_path, target_shape=mask_unet_post.shape)

    # 核对
    print("Pred UNet sum:", pred_unet_np.sum())
    print("Pred EBAUNet sum:", pred_ebaunet_np.sum())
    print("GT mask sum:", gt_mask.sum())

    # 计算边界指标
    boundary_iou_unet = boundary_iou(pred_unet_np, gt_mask)
    boundary_f1_unet = boundary_f1(pred_unet_np, gt_mask)

    boundary_iou_ebaunet = boundary_iou(pred_ebaunet_np, gt_mask)
    boundary_f1_ebaunet = boundary_f1(pred_ebaunet_np, gt_mask)

    # 计算普通指标
    iou_unet = calculate_iou(pred_unet_np, gt_mask)
    f1_unet = calculate_f1(pred_unet_np, gt_mask)

    iou_ebaunet = calculate_iou(pred_ebaunet_np, gt_mask)
    f1_ebaunet = calculate_f1(pred_ebaunet_np, gt_mask)

    # 打印结果
    print("\n===== 边界指标对比 =====")
    print(f"UNet:    Boundary IoU = {boundary_iou_unet:.4f}, Boundary F1 = {boundary_f1_unet:.4f}")
    print(f"EBAUNet: Boundary IoU = {boundary_iou_ebaunet:.4f}, Boundary F1 = {boundary_f1_ebaunet:.4f}")

    print("\n===== 普通指标对比 =====")
    print(f"UNet:    IoU = {iou_unet:.4f}, F1 = {f1_unet:.4f}")
    print(f"EBAUNet: IoU = {iou_ebaunet:.4f}, F1 = {f1_ebaunet:.4f}")

    # 可视化原始预测对比
    #visualize_compare(image, mask_unet, mask_ebaunet, save_dir, name="compare_original")

    # 可视化后处理预测对比
    visualize_compare(image, mask_unet_post, mask_ebaunet_post, save_dir, name="compare_postprocessed")

    # 可视化两个模型与GT的对比
    visualize_model_comparison(image, gt_mask, pred_unet_np, pred_ebaunet_np, save_dir, name="model_comparison")


if __name__ == "__main__":
    main()