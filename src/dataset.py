import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from PIL import Image
import h5py
from tqdm import tqdm
import glob

# 将项目根目录添加到sys.path
import sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)

from cadlib.macro import *
from src.unified_vocab.converter import convert_13d_to_unified_tokens

class OmniCADDataset(Dataset):
    """
    为 Omni-CAD 设计的多模态数据集，能够加载 CAD 向量序列、文本描述和8个视图的图像。
    支持多视图融合：从8个视图中随机选择指定数量的视图。
    """
    def __init__(self, cad_vec_dir, text_dir, image_dir, split='all', sample_limit=None,
                 category_start=0, category_end=None, text_only=False,
                 num_selected_views=2, random_select_views=True):
        """
        初始化 Omni-CAD 数据集。

        Args:
            cad_vec_dir (str): 预处理后的 .h5 CAD向量文件根目录 (e.g., 'data/Omni-CAD/cad_vec').
            text_dir (str): 文本描述JSON文件所在的目录 (e.g., 'data/Omni-CAD/txt').
            image_dir (str): 图像文件的根目录 (e.g., 'data/Omni-CAD/step_img').
            split (str): 数据集划分 (当前版本中未使用，默认为 'all').
            sample_limit (int, optional): 限制加载的样本数量，用于快速测试。
            category_start (int): 起始类别编号 (默认0，即'0000').
            category_end (int, optional): 结束类别编号 (包含)。None表示加载所有可用类别。
            text_only (bool): 仅使用文本模态，跳过图像加载（第一阶段训练）。
            num_selected_views (int): 多视图融合时选择的视图数量 (默认2)。
            random_select_views (bool): 是否随机选择视图 (训练时True，推理时False)。
        """
        self.cad_vec_dir = cad_vec_dir
        self.text_dir = text_dir
        self.image_dir = image_dir
        self.split = split
        self.sample_limit = sample_limit
        self.category_start = category_start
        self.category_end = category_end
        self.text_only = text_only
        self.num_views = 8
        self.num_selected_views = num_selected_views
        self.random_select_views = random_select_views

        self.samples = []
        self._load_samples()

        self.text_captions = self._load_text_captions()

        # 图像预处理
        self.image_transform = Compose([
            Resize((224, 224)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _load_samples(self):
        """
        遍历 cad_vec 目录，收集所有 .h5 文件的路径作为样本。
        支持按类别范围筛选。
        """
        print(f"正在从 '{self.cad_vec_dir}' 收集 CAD 向量样本...")
        if not os.path.isdir(self.cad_vec_dir):
            raise FileNotFoundError(f"CAD 向量目录不存在: {self.cad_vec_dir}")

        # 获取所有类别子目录
        all_categories = sorted([d for d in os.listdir(self.cad_vec_dir)
                                  if os.path.isdir(os.path.join(self.cad_vec_dir, d))])

        # 筛选类别范围
        if self.category_end is not None:
            selected_categories = [c for c in all_categories
                                   if self.category_start <= int(c) <= self.category_end]
            print(f"选择类别范围: {self.category_start:04d} - {self.category_end:04d}")
        else:
            selected_categories = [c for c in all_categories
                                   if int(c) >= self.category_start]
            print(f"选择类别范围: {self.category_start:04d} - 全部")

        print(f"找到 {len(selected_categories)} 个类别: {selected_categories[:5]}{'...' if len(selected_categories) > 5 else ''}")

        # 收集选定类别中的样本
        for category in selected_categories:
            category_path = os.path.join(self.cad_vec_dir, category)
            h5_files = glob.glob(os.path.join(category_path, '*.h5'))

            for h5_path in h5_files:
                filename = os.path.splitext(os.path.basename(h5_path))[0]  # e.g., "00000007_00001"
                sample_id = f"{category}/{filename}"  # e.g., "0000/00000007_00001"
                self.samples.append({
                    'id': sample_id,
                    'h5_path': h5_path,
                    'category': category,
                    'filename': filename
                })

        if self.sample_limit:
            self.samples = self.samples[:self.sample_limit]

        print(f"收集到 {len(self.samples)} 个样本。")

    def _load_text_captions(self):
        """
        加载所有文本描述到内存中。
        假设每个子目录（如 '0000'）对应一个 '0000.json' 文件，
        该文件是一个字典，键是模型ID，值是描述文本。
        """
        print(f"正在从 '{self.text_dir}' 加载文本描述...")
        captions = {}
        if not self.text_dir or not os.path.exists(self.text_dir):
            print(f"警告: 文本目录不存在: {self.text_dir}")
            return {}

        json_files = glob.glob(os.path.join(self.text_dir, '*.json'))
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # 假设data是{'model_id': 'caption', ...}
                    if isinstance(data, dict):
                         captions.update(data)
                    # 兼容[{'id': ..., 'text caption': ...}] 的格式
                    elif isinstance(data, list):
                        for item in data:
                            captions[item.get('id')] = item.get('text caption', item.get('text'))

            except json.JSONDecodeError:
                print(f"警告: 无法解析JSON文件 {json_file}")
            except Exception as e:
                print(f"警告: 加载文本文件 {json_file} 时出错: {e}")

        print(f"加载了 {len(captions)} 条文本描述。" )
        return captions

    def __len__(self):
        return len(self.samples)

    def _convert_17d_to_13d(self, vec_17d):
        """
        将单个17维CAD向量转换为13维格式。
        """
        vec_13d = np.full(13, PAD_VAL, dtype=np.int32)
        command = int(vec_17d[0])
        vec_13d[0] = command

        if command in [LINE_IDX, ARC_IDX, CIRCLE_IDX]:
            # 复制草图参数
            vec_13d[1:6] = vec_17d[1:6]
        elif command == EXT_IDX:
            # 角度转换为Token
            theta, phi, gamma = vec_17d[6], vec_17d[7], vec_17d[8]
            i_theta = np.clip(np.floor((theta / np.pi) * 9), 0, 8)
            i_phi = np.clip(np.floor(((phi + np.pi) / (2 * np.pi)) * 9), 0, 8)
            i_gamma = np.clip(np.floor(((gamma + np.pi) / (2 * np.pi)) * 9), 0, 8)
            vec_13d[6] = i_theta * 81 + i_phi * 9 + i_gamma

            # 位置转换为Token
            px, py, pz = vec_17d[9], vec_17d[10], vec_17d[11]
            i_x = np.clip(np.floor(px * 36), 0, 35)
            i_y = np.clip(np.floor(py * 36), 0, 35)
            i_z = np.clip(np.floor(pz * 36), 0, 35)
            vec_13d[7] = i_z * 1296 + i_y * 36 + i_x
            
            # 复制剩余的拉伸参数
            vec_13d[8:13] = vec_17d[12:17]
            
        return vec_13d

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        sample_id = sample_info['id']  # e.g., '0000/00000007_00001'
        h5_path = sample_info['h5_path']
        category = sample_info['category']  # e.g., '0000'
        filename = sample_info['filename']  # e.g., '00000007_00001'

        # 1. 加载并转换CAD向量序列
        try:
            with h5py.File(h5_path, 'r') as fp:
                cad_vec_17d = fp['vec'][:]  # type:ignore

            # 将17维向量序列转换为13维
            cad_vec_13d = np.array([self._convert_17d_to_13d(v) for v in cad_vec_17d], dtype=np.int32)  # type:ignore

            # 确保序列长度统一，进行填充
            padded_cad_vec = np.full((MAX_TOTAL_LEN, 13), EOS_IDX, dtype=np.int32)
            seq_len = min(len(cad_vec_13d), MAX_TOTAL_LEN)
            padded_cad_vec[:seq_len] = cad_vec_13d[:seq_len]
            cad_tensor = torch.from_numpy(padded_cad_vec).long()
        except Exception as e:
            print(f"警告: 无法加载或处理h5文件 {h5_path}: {e}")
            # 返回一个表示错误的空Tensor
            cad_tensor = torch.full((MAX_TOTAL_LEN, 13), EOS_IDX, dtype=torch.long)

        # 2. 加载文本描述
        text_caption = self.text_captions.get(sample_id, "No description available.")

        # 3. 加载视图图像（text_only模式下跳过）
        if self.text_only:
            # 纯文本模式：返回占位tensor，不实际加载图像
            images_stacked = torch.zeros(self.num_selected_views, 3, 224, 224)
            selected_view_indices = torch.zeros(self.num_selected_views, dtype=torch.long)
        else:
            # 随机选择视图索引
            if self.random_select_views:
                selected_indices = np.random.choice(
                    self.num_views, self.num_selected_views, replace=False
                )
                selected_indices = np.sort(selected_indices)  # 保持顺序一致性
            else:
                # 固定选择前 num_selected_views 个视图（用于推理）
                selected_indices = np.arange(self.num_selected_views)

            # 图像路径格式: step_img/{category}/{filename}_{view:03d}.png
            image_tensors = []
            for i in selected_indices:
                img_path = os.path.join(self.image_dir, category, f"{filename}_{i:03d}.png")
                try:
                    if os.path.exists(img_path):
                        image = Image.open(img_path).convert('RGB')
                        image_tensors.append(self.image_transform(image))
                    else:
                        # 如果某个视图不存在，用黑色图片填充
                        image_tensors.append(torch.zeros(3, 224, 224))
                except Exception as e:
                    print(f"警告: 无法加载或处理图片 {img_path}: {e}")
                    image_tensors.append(torch.zeros(3, 224, 224))

            # 如果一张图片都加载失败，则返回占位图片
            if not image_tensors:
                 image_tensors = [torch.zeros(3, 224, 224) for _ in range(self.num_selected_views)]

            images_stacked = torch.stack(image_tensors)  # 堆叠成 (num_selected_views, C, H, W)
            selected_view_indices = torch.from_numpy(selected_indices).long()

        # 4. 转换为统一词表格式
        commands, args_tokens = convert_13d_to_unified_tokens(padded_cad_vec)
        commands_tensor = torch.from_numpy(commands).long()
        args_tokens_tensor = torch.from_numpy(args_tokens).long()

        return {
            'id': sample_id,
            'cad_sequence': cad_tensor,
            'commands': commands_tensor,
            'args_tokens': args_tokens_tensor,
            'text_caption': text_caption,
            'images': images_stacked,
            'view_indices': selected_view_indices  # 选中的视图索引
        }


# --- 测试代码 ---
if __name__ == '__main__':
    # 请确保以下路径正确
    cad_dir = 'data/Omni-CAD/cad_vec'
    text_dir_path = 'data/Omni-CAD/txt'
    image_dir_path = 'data/Omni-CAD/step_img'

    print("正在初始化 OmniCADDataset...")
    
    # 检查所需目录是否存在
    if not all(os.path.exists(p) for p in [cad_dir, text_dir_path, image_dir_path]):
        print("错误: 一个或多个数据目录不存在。请检查路径。")
        print(f"CAD目录: {cad_dir} {'(存在)' if os.path.exists(cad_dir) else '(不存在)'}")
        print(f"文本目录: {text_dir_path} {'(存在)' if os.path.exists(text_dir_path) else '(不存在)'}")
        print(f"图像目录: {image_dir_path} {'(存在)' if os.path.exists(image_dir_path) else '(不存在)'}")
    else:
        # 使用 limit 进行快速测试
        dataset = OmniCADDataset(
            cad_vec_dir=cad_dir,
            text_dir=text_dir_path,
            image_dir=image_dir_path,
            sample_limit=5 
        )

        print(f"\n数据集大小: {len(dataset)} 个样本")

        if len(dataset) > 0:
            print("\n正在获取第一个样本...")
            sample = dataset[0]
            
            print(f"样本ID: {sample['id']}")
            print(f"CAD序列张量形状: {sample['cad_sequence'].shape}")
            print(f"文本描述: '{sample['text_caption']}'")
            print(f"图像张量形状: {sample['images'].shape}")
            
            # 检查CAD序列内容
            print("\nCAD序列前5行 (原始命令索引):")
            print(sample['cad_sequence'][:5, :])
            print(f"序列中第一个命令的索引: {sample['cad_sequence'][0, 0]}")
            
            # 检查图像张量
            print(f"\n图像张量的最大值: {sample['images'].max()}")
            print(f"图像张量的最小值: {sample['images'].min()}")
        else:
            print("数据集中没有样本，请检查数据目录和文件。")
