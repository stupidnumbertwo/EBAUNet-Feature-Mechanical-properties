import os
import numpy as np
from PIL import Image

def convert_red_to_white_on_black(input_path, output_path):
    """
    将黑底红字转换为黑底白字
    """
    # 1. 加载图片并转换为 RGB
    img = Image.open(input_path).convert("RGB")
    data = np.array(img)

    # 2. 提取 RGB 通道
    r, g, b = data[:,:,0], data[:,:,1], data[:,:,2]

    # 3. 判定红色的掩膜 (Mask)
    # 逻辑：红色通道值较高，且明显高于绿色和蓝色
    # 你可以根据实际图片的颜色深度调整 100 这个阈值
    red_mask = (r > 100) & (g < 100) & (b < 100)

    # 4. 执行替换：将掩膜选中的“红色区域”改为纯白色 [255, 255, 255]
    # 其余背景色（黑色）不作变动
    data[red_mask] = [255, 255, 255]

    # 5. 保存结果
    new_img = Image.fromarray(data)
    
    # 如果原图是 JPG，保存时建议指定质量
    if input_path.lower().endswith(('.jpg', '.jpeg')):
        new_img.save(output_path, quality=95)
    else:
        new_img.save(output_path)

def process_folder(folder_path):
    """
    遍历处理文件夹
    """
    supported_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    if not os.path.exists(folder_path):
        print(f"找不到路径: {folder_path}")
        return

    for filename in os.listdir(folder_path):
        if any(filename.lower().endswith(ext) for ext in supported_extensions):
            file_path = os.path.join(folder_path, filename)
            try:
                convert_red_to_white_on_black(file_path, file_path)
                print(f"成功转换: {filename}")
            except Exception as e:
                print(f"处理 {filename} 时发生错误: {e}")

# --- 配置区 ---
target_folder = r'E:/test/image'  # 确保路径正确
process_folder(target_folder)