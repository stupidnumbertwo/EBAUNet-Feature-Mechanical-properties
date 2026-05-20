import os
import time
import torch
from torchvision import transforms
import numpy as np
from PIL import Image, ImageEnhance
from src import EBAUNet
import argparse
from torch.serialization import add_safe_globals


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


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


def preprocess_image(img_path, mean, std):
    try:
        original_img = Image.open(img_path).convert('L')
        # 将图像转换为黑白模式
        # bw_image = original_img.convert('L')
        # 使用PIL增强对比度
        enhancer = ImageEnhance.Contrast(original_img)
        # 调整对比度，数值越大对比度越强，可根据效果调整
        enhanced_img = enhancer.enhance(3)
        # 将黑白图像转换回RGB模式
        # enhanced_img = enhanced_img.convert('RGB')

        data_transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=mean, std=std)])
        img = data_transform(enhanced_img)
        img = torch.unsqueeze(img, dim=0)
        return img, original_img
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None, None


def infer_image(model, img, device):
    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 1, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        output = model(img.to(device))

        # 检查 output 是否为字典，并提取分割结果张量
        if isinstance(output, dict) and 'out' in output:
            output = output['out']
        else:
            raise ValueError("Output does not contain the expected segmentation result.")

        t_end = time_synchronized()
        print("inference time: {}".format(t_end - t_start))

        prediction = output.argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        # 将前景对应的像素值改成255(白色)
        prediction[prediction == 1] = 255
        # 颜色互换，将 0 变成 255，将 255 变成 0
        #prediction = 255 - prediction
        return prediction


def main():
    classes = 1  # exclude background
    weights_path = "save_weights/EBA_UNet_model.pth"  # 权重文件路径
    # 自定义输入图像路径列表，可以继续添加更多图像路径
    input_image_paths = [
        #"images/heibi-6.jpg",
        #"images/970C-5M-Q-2.png",
        "images/heix-6.jpg",
    ]

    # 自定义输出图像保存目录
    output_dir = "result"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mean = (0.5,)
    std = (0.5,)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # load model
    model = load_model(weights_path, device)
    if model is None:
        return

    for img_path in input_image_paths:
        assert os.path.exists(img_path), f"image {img_path} not found."
        img, original_img = preprocess_image(img_path, mean, std)
        if img is None or original_img is None:
            continue

        prediction = infer_image(model, img, device)

        # 生成输出图像文件名
        file_name = os.path.basename(img_path).split('.')[0] + "_result.png"
        output_path = os.path.join(output_dir, file_name)
        mask = Image.fromarray(prediction)
        mask.save(output_path)


if __name__ == '__main__':
    main()