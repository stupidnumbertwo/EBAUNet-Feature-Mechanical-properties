import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class DriveDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(DriveDataset, self).__init__()
        self.flag = "training" if train else "test"
        self.root = root
        self.transforms = transforms

        # 构建当前数据集（训练集或测试集）的根路径
        dataset_root = os.path.join(root, "DRIVE", self.flag)
        assert os.path.exists(dataset_root), f"Dataset root {dataset_root} does not exist."

        # 定义原图和标注图的文件夹路径
        self.image_folder = os.path.join(dataset_root, "images")
        self.label_folder = os.path.join(dataset_root, "labels")

        # 检查文件夹是否存在
        assert os.path.exists(self.image_folder), f"Image folder {self.image_folder} does not exist."
        assert os.path.exists(self.label_folder), f"Label folder {self.label_folder} does not exist."

        # 获取原图文件名列表，筛选 .jpg 格式的文件
        self.img_names = [i for i in os.listdir(self.image_folder) if i.endswith(".jpg")]
        self.img_list = [os.path.join(self.image_folder, i) for i in self.img_names]
        self.manual = [os.path.join(self.label_folder, os.path.splitext(i)[0] + ".png")
                       for i in self.img_names]

        # 检查文件是否存在
        for img_path, label_path in zip(self.img_list, self.manual):
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image file {img_path} does not exist.")
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Label file {label_path} does not exist.")

    def __getitem__(self, idx):
        img_path = self.img_list[idx]
        label_path = self.manual[idx]

        # 读取原图和标注图
        img = Image.open(img_path).convert('L')
        manual = Image.open(label_path).convert('L')
        manual = np.array(manual)

        # 分类数量为 2（根据实际情况修改）
        num_classes = 2
        ignore_index = 1

        # 将超出范围的标签设置为忽略索引
        manual[manual > 0] = ignore_index

        # 确保标签范围在 [0, num_classes - 1]
        manual = np.clip(manual, 0, num_classes - 1).astype(np.uint8)

        # 这里转回 PIL 的原因是，transforms 中是对 PIL 数据进行处理
        mask = Image.fromarray(manual)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        def cat_list(images, fill_value=0):
            max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
            batch_shape = (len(images),) + max_size
            batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
            for img, pad_img in zip(images, batched_imgs):
                pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
            return batched_imgs
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets

