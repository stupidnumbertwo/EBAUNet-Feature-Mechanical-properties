import cv2
import numpy as np
from matplotlib import pyplot as plt

def analyze_binary_image(image_path, output_file="region_analysis.txt", 
                         min_pixels=10,  # 最小像素数阈值
                         image_width_um=5330.16, image_height_um=4091.54,
                         image_width_pixels=3584, image_height_pixels=2746):
    # 计算像素到微米的转换因子
    pixel_size_x = image_width_um / image_width_pixels
    pixel_size_y = image_height_um / image_height_pixels
    pixel_area = pixel_size_x * pixel_size_y
    total_image_area = image_width_um * image_height_um
    
    # 读取二值图像并预处理
    img = cv2.imread(image_path, 0)
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    
    # 连通组件分析
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # 过滤小区域（保留像素数 >= min_pixels 的区域）
    valid_indices = np.where(stats[:, cv2.CC_STAT_AREA] >= min_pixels)[0]
    valid_indices = valid_indices[valid_indices != 0]  # 排除背景(0)
    
    # 如果没有有效区域，输出警告并返回
    if len(valid_indices) == 0:
        print(f"警告：没有找到像素数 >= {min_pixels} 的区域！")
        return 0, [], [], 0
    
    # 提取并排序有效区域
    areas_pixels = stats[valid_indices, cv2.CC_STAT_AREA]
    sorted_indices = np.argsort(-areas_pixels)
    valid_indices_sorted = valid_indices[sorted_indices]
    
    # 转换为物理单位
    areas_um2 = areas_pixels * pixel_area
    radii_um = np.sqrt(areas_um2 / np.pi)
    total_black_area = np.sum(areas_um2)
    area_fraction = total_black_area / total_image_area * 100
    
    # 创建过滤后的标记图
    filtered_labels = np.zeros_like(labels)
    for i, idx in enumerate(valid_indices_sorted, 1):
        filtered_labels[labels == idx] = i
    
    # 保存结果到文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"区域分析结果 - 共{len(valid_indices)}个区域 (像素数 >= {min_pixels})\n")
        f.write(f"图像尺寸: {image_width_um:.2f} x {image_height_um:.2f} 微米\n")
        f.write(f"像素尺寸: {image_width_pixels} x {image_height_pixels} 像素\n")
        f.write(f"像素面积: {pixel_area:.6f} 平方微米\n")
        f.write("="*60 + "\n")
        f.write(f"{'区域ID':<8}{'像素数':<12}{'面积(μm²)':<16}{'等效圆半径(μm)':<16}\n")
        f.write("-"*60 + "\n")
        
        for i, idx in enumerate(sorted_indices, 1):
            f.write(f"{i:<8}{areas_pixels[idx]:<12}{areas_um2[idx]:<16.2f}{radii_um[idx]:<16.2f}\n")
        
        f.write("="*60 + "\n")
        f.write(f"{'统计量':<20}{'值':<20}\n")
        f.write("-"*40 + "\n")
        f.write(f"{'平均半径(μm)':<20}{np.mean(radii_um):<20.2f}\n")
        f.write(f"{'最大半径(μm)':<20}{np.max(radii_um):<20.2f}\n")
        f.write(f"{'最小半径(μm)':<20}{np.min(radii_um):<20.2f}\n")
        f.write(f"{'黑色区域面积分数(%)':<20}{area_fraction:<20.2f}\n")
    
    print(f"分析结果已保存至 {output_file}")
    
    # 创建彩色标记图
    colors = np.random.randint(0, 255, size=(len(valid_indices)+1, 3), dtype=np.uint8)
    colors[0] = [255, 255, 255]  # 背景为白色
    colored_labels = colors[filtered_labels]
    
    # 可视化结果
    plt.figure(figsize=(12, 5))
    plt.subplot(121), plt.imshow(img, cmap='gray')
    plt.axis('off')
    
    plt.subplot(122), plt.imshow(colored_labels)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("region_visualization.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    return len(valid_indices), areas_um2, radii_um, area_fraction

# 使用示例
if __name__ == "__main__":
    image_path = "image/heix-6.png"  # 替换为你的图像路径
    min_pixels_threshold = 700  # 设置最小像素数阈值
    
    count, sorted_areas, sorted_radii, area_fraction = analyze_binary_image(
        image_path, 
        min_pixels=min_pixels_threshold
    )
    
    # 打印统计信息
    if count > 0:
        print(f"\n统计结果 (像素数 >= {min_pixels_threshold}):")
        print(f"平均半径: {np.mean(sorted_radii):.2f} 微米")
        print(f"最大半径: {np.max(sorted_radii):.2f} 微米")
        print(f"最小半径: {np.min(sorted_radii):.2f} 微米")
        print(f"黑色区域面积分数: {area_fraction:.2f}%")