import torch
import torch.nn as nn
import numpy as np
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers.utils.quantization_config import BitsAndBytesConfig
from dataclasses import dataclass
from PIL import Image
from typing import List, Dict, Any, Optional

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.model.layers.transformer import TransformerDecoder, TransformerDecoderLayer, LayerNorm
from src.model.layers.positional_encoding import PositionalEncodingLUT
from src.model.multiview_fusion import PerceiverFusion
from cadlib.macro import *
from src.unified_vocab.vocab import VOCAB_SIZE, MAX_ARGS_PER_CMD


@dataclass
class MechCADConfig:
    """MechCAD 模型的配置。"""
    # Transformer 和解码器设置
    d_model: int = 256
    n_heads: int = 8
    dim_feedforward: int = 512
    dropout: float = 0.1
    n_layers_decode: int = 4
    
    # CAD 序列属性
    cad_max_total_len: int = MAX_TOTAL_LEN  # 60
    cad_n_commands: int = len(ALL_COMMANDS)
    cad_n_args: int = 12  # 基于 13D 向量（1 个命令 + 12 个参数）
    
    # 来自 vector_definition.md 的参数标记化设置
    angle_bins: int = 9
    pos_grid_size: int = 36
    
    @property
    def n_angle_tokens(self) -> int:
        return self.angle_bins ** 3

    @property
    def n_pos_tokens(self) -> int:
        return self.pos_grid_size ** 3
        
    # LLM/MLLM 设置
    llm_hidden_dim: int = 4096  # LLaVA-1.5-7B 隐层大小
    args_dim: int = 256 # 连续参数回归的维度，不被令牌头使用

# --- 解码器子模块 ---

class ConstEmbedding(nn.Module):
    """用位置编码生成一个常数、可学习的序列。"""
    def __init__(self, cfg: MechCADConfig):
        super().__init__()
        self.d_model = cfg.d_model
        self.PE = PositionalEncodingLUT(cfg.d_model, max_len=cfg.cad_max_total_len)
        self.seq_len = cfg.cad_max_total_len

    def forward(self, z):
        N = z.size(1)
        src = self.PE(z.new_zeros(self.seq_len, N, self.d_model))
        return src

class CommandFCN(nn.Module):
    """从 Transformer 输出预测命令逻辑。"""
    def __init__(self, cfg: MechCADConfig):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2), nn.ReLU(),
            nn.Linear(cfg.d_model // 2, cfg.d_model // 4), nn.ReLU(),
            nn.Linear(cfg.d_model // 4, cfg.cad_n_commands)
        )
    def forward(self, out):
        return self.mlp(out)

class ArgsFCN(nn.Module):
    """从 Transformer 输出预测参数特征。"""
    def __init__(self, cfg: MechCADConfig):
        super().__init__()
        self.n_args = cfg.cad_n_args
        self.args_dim = cfg.args_dim
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 4), nn.ReLU(),
            nn.Linear(cfg.d_model * 4, cfg.d_model * 2), nn.ReLU(),
            nn.Linear(cfg.d_model * 2, self.n_args * self.args_dim)
        )
    def forward(self, out):
        S, N, _ = out.shape
        args_logits = self.mlp(out)
        args_logits = args_logits.reshape(S, N, self.n_args, self.args_dim)
        return args_logits

class CommandDecoder(nn.Module):
    """从 LLM 特征解码命令序列。"""
    def __init__(self, cfg: MechCADConfig):
        super().__init__()
        self.embedding = ConstEmbedding(cfg)
        decoder_layer = TransformerDecoderLayer(cfg.d_model, cfg.n_heads, cfg.dim_feedforward, cfg.dropout)
        self.decoder = TransformerDecoder(decoder_layer, cfg.n_layers_decode, LayerNorm(cfg.d_model))
        self.fcn = CommandFCN(cfg)

    def forward(self, z, memory_key_padding_mask=None):
        src = self.embedding(z)
        out = self.decoder(src, z, memory_key_padding_mask=memory_key_padding_mask)
        command_logits = self.fcn(out)
        return command_logits, out

class ArgsDecoder(nn.Module):
    """从 LLM 特征解码参数序列，使用统一大词表。"""
    def __init__(self, cfg: MechCADConfig):
        super().__init__()
        self.embedding = ConstEmbedding(cfg)
        decoder_layer = TransformerDecoderLayer(cfg.d_model, cfg.n_heads, cfg.dim_feedforward, cfg.dropout)
        self.decoder = TransformerDecoder(decoder_layer, cfg.n_layers_decode, LayerNorm(cfg.d_model))

        self.n_args = MAX_ARGS_PER_CMD
        self.vocab_size = VOCAB_SIZE

        # 统一大词表输出头
        self.output_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model * 2),
            nn.LayerNorm(cfg.d_model * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model * 2, self.n_args * self.vocab_size)
        )

    def forward(self, z, guidance, memory_key_padding_mask=None):
        src = self.embedding(z)
        src = src + guidance  # 注入来自命令解码器的引导
        out = self.decoder(src, z, memory_key_padding_mask=memory_key_padding_mask)
        # out: [S, N, d_model]

        # 统一大词表输出
        logits = self.output_head(out)  # [S, N, n_args * vocab_size]
        S, N, _ = logits.shape
        logits = logits.reshape(S, N, self.n_args, self.vocab_size)

        return logits

# --- 顶级解码器 ---

class LLM2CADDecoder(nn.Module):
    """双解码器，将 LLM 特征转换为 CAD 序列。"""
    def __init__(self, cfg: MechCADConfig):
        super().__init__()
        self.d_model = cfg.d_model
        # 使用多层 adapter 并添加归一化，防止梯度爆炸
        self.adapter = nn.Sequential(
            nn.Linear(cfg.llm_hidden_dim, cfg.d_model * 2),
            nn.LayerNorm(cfg.d_model * 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model * 2, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )
        self.command_decoder = CommandDecoder(cfg)
        self.args_decoder = ArgsDecoder(cfg)

        # 确保所有参数为 float32 (避免 LLaVA float16 导致溢出)
        self.float()

    def forward(self, llm_features, memory_key_padding_mask=None):
        # 转换输入为 float32
        llm_features = llm_features.float()

        z = self.adapter(llm_features)
        z = z.permute(1, 0, 2)  # [LLM_Seq_Len, Batch, d_model]

        command_logits, guidance = self.command_decoder(z, memory_key_padding_mask)
        unified_args_logits = self.args_decoder(z, guidance, memory_key_padding_mask)

        # 置换到 [Batch, Seq_Len, ...] 以进行损失计算
        command_logits = command_logits.permute(1, 0, 2)
        unified_args_logits = unified_args_logits.permute(1, 0, 2, 3)

        return command_logits, unified_args_logits

# --- 完整 MLLM 模型 ---

class MechCADModel(nn.Module):
    """
    完整的 MechCAD 模型，将多模态输入转换为 CAD 序列。
    支持多视图融合：使用 PerceiverFusion 融合多个视图的特征。
    """
    def __init__(self, cfg: MechCADConfig, llava_model_name="model_weights/llava-hf/llava-1.5-7b-hf",
                 num_views=2, n_latents=64):
        super().__init__()
        self.cfg = cfg
        self.num_views = num_views

        # --- 编码层 ---
        self.processor = AutoProcessor.from_pretrained(llava_model_name)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )

        self.llava_model = LlavaForConditionalGeneration.from_pretrained(
            llava_model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )
        self.llava_model.eval()  # 冻结 LLaVA 模型参数

        # --- 多视图融合模块 ---
        self.multiview_fusion = PerceiverFusion(
            hidden_dim=cfg.llm_hidden_dim,  # 4096
            n_views=num_views,
            n_latents=n_latents,
            n_heads=8,
            n_layers=2,
            dropout=cfg.dropout
        )

        # --- Decoder ---
        self.llm2cad_decoder = LLM2CADDecoder(cfg)

    def _denormalize_image(self, img_tensor: torch.Tensor) -> np.ndarray:
        """将归一化的图像张量转换回 PIL 可用的 numpy 数组。"""
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np * std + mean) * 255
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        return img_np

    def forward(self, batch: Dict[str, Any], text_only: bool = False) -> Dict[str, torch.Tensor]:
        """
        处理来自 OmniCADDataset 的一个批次。

        Args:
            batch: 包含 'images', 'text_caption', 'cad_sequence' 的字典
                   images: (B, num_views, C, H, W) 多视图图像
            text_only: 是否仅使用文本模态（第一阶段训练）

        Returns:
            包含 'command_logits', 'unified_args_logits' 的字典
        """
        texts = batch['text_caption']    # 字符串列表

        if text_only:
            # --- 纯文本模式：不使用图像 ---
            prompts = [f"USER: {caption}\nGenerate the CAD command sequence. ASSISTANT:"
                       for caption in texts]
            inputs = self.processor(text=prompts, return_tensors="pt", padding=True)

            # 将输入移到与模型相同的设备上
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.llava_model.device)

            # 使用 LLaVA 编码
            with torch.no_grad():
                outputs = self.llava_model(**inputs, output_hidden_states=True)
                llm_features = outputs.hidden_states[-1]  # [B, Seq, 4096]
                memory_key_padding_mask = (inputs.attention_mask == 0)

        else:
            # --- 多视图模式：使用多个图像视图 + 文本 ---
            images_tensor = batch['images']  # (B, num_views, C, H, W)
            batch_size = images_tensor.size(0)
            num_views = images_tensor.size(1)

            # 为每个视图分别获取 LLaVA 特征
            all_view_features = []
            all_attention_masks = []

            for v in range(num_views):
                # 反归一化后转为 PIL 图像
                pil_images = []
                for i in range(batch_size):
                    img_np = self._denormalize_image(images_tensor[i, v])
                    pil_images.append(Image.fromarray(img_np))

                # 创建 LLaVA 风格的提示
                prompts = [f"USER: <image>\n{caption} ASSISTANT:" for caption in texts]
                inputs = self.processor(text=prompts, images=pil_images, return_tensors="pt", padding=True)

                # 将输入移到与模型相同的设备上
                for k, val in inputs.items():
                    if isinstance(val, torch.Tensor):
                        inputs[k] = val.to(self.llava_model.device)

                # 使用 LLaVA 编码
                with torch.no_grad():
                    outputs = self.llava_model(**inputs, output_hidden_states=True)
                    view_features = outputs.hidden_states[-1]  # [B, Seq, 4096]
                    all_view_features.append(view_features)
                    all_attention_masks.append(inputs.attention_mask)

            # 将多视图特征堆叠: [B, num_views, Seq, Hidden]
            # 由于不同视图可能有不同序列长度，需要对齐
            max_seq_len = max(f.size(1) for f in all_view_features)
            padded_features = []
            for f in all_view_features:
                if f.size(1) < max_seq_len:
                    pad = torch.zeros(batch_size, max_seq_len - f.size(1), f.size(2),
                                     device=f.device, dtype=f.dtype)
                    f = torch.cat([f, pad], dim=1)
                padded_features.append(f)

            multiview_features = torch.stack(padded_features, dim=1)  # [B, num_views, Seq, 4096]

            # 使用 PerceiverFusion 融合多视图特征
            self.multiview_fusion.to(multiview_features.device)
            llm_features = self.multiview_fusion(multiview_features.float())  # [B, n_latents, 4096]

            # 融合后的特征没有 padding，创建全零的 mask
            memory_key_padding_mask = torch.zeros(
                batch_size, llm_features.size(1),
                dtype=torch.bool, device=llm_features.device
            )

        # --- 解码为 CAD 序列 ---
        self.llm2cad_decoder.to(llm_features.device)

        command_logits, unified_args_logits = self.llm2cad_decoder(
            llm_features, memory_key_padding_mask
        )

        return {
            'command_logits': command_logits,
            'unified_args_logits': unified_args_logits,
        }

# --- 示例用法 ---
if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from src.dataset import OmniCADDataset

    # 此块演示如何初始化和运行模型。
    # 注意：运行此代码需要大量 GPU 内存。
    
    # 1. 创建配置
    config = MechCADConfig()

    # 2. 初始化完整模型
    print("初始化 MechCADModel...")
    model = MechCADModel(config)
    print("模型已初始化。")

    # 3. 从真实数据集中加载一个批次
    print("\n从 OmniCADDataset 加载真实批次...")
    try:
        cad_dir = 'data/Omni-CAD/cad_vec'
        text_dir_path = 'data/Omni-CAD/txt'
        image_dir_path = 'data/Omni-CAD/step_img'
        
        dataset = OmniCADDataset(
            cad_vec_dir=cad_dir,
            text_dir=text_dir_path,
            image_dir=image_dir_path,
            sample_limit=2 
        )
        data_loader = DataLoader(dataset, batch_size=2)
        real_batch = next(iter(data_loader))
        print("真实批次已加载。")

        # 4. 执行前向传播
        print("\n对真实批次执行前向传播...")
        outputs = model(real_batch)

        # 5. 打印输出形状以验证
        print("\n--- 输出形状 ---")
        print(f"命令逻辑: {outputs['command_logits'].shape}")
        print(f"参数特征:  {outputs['args_logits'].shape}")
        print(f"角度逻辑:   {outputs['angle_logits'].shape}")
        print(f"位置逻辑:  {outputs['pos_logits'].shape}")
        
        # 预期形状
        batch_size = len(real_batch['id'])
        print(f"\n预期命令逻辑形状: ({batch_size}, {config.cad_max_total_len}, {config.cad_n_commands})")
        print(f"预期参数特征形状: ({batch_size}, {config.cad_max_total_len}, {config.cad_n_args}, {config.args_dim})")
        
    except Exception as e:
        print(f"\n前向传播期间发生错误: {e}")
        import traceback
        traceback.print_exc()
        print("\n如果您没有足够的 GPU 内存、缺少所需的软件包或数据路径不正确，这是正常的。")

