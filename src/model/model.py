import torch
import torch.nn as nn
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers.utils.quantization_config import BitsAndBytesConfig
from dataclasses import dataclass
from PIL import Image
from typing import List

# cadlib.macro 和其他本地导入可能需要调整 sys.path
# 这在数据集中处理，但在模型文件中也是好的实践。
import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.model.layers.transformer import TransformerDecoder, TransformerDecoderLayer, LayerNorm
from src.model.layers.positional_encoding import PositionalEncodingLUT
from cadlib.macro import *


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
    """从 LLM 特征解码参数序列，由命令解码器引导。"""
    def __init__(self, cfg: MechCADConfig):
        super().__init__()
        self.embedding = ConstEmbedding(cfg)
        decoder_layer = TransformerDecoderLayer(cfg.d_model, cfg.n_heads, cfg.dim_feedforward, cfg.dropout)
        self.decoder = TransformerDecoder(decoder_layer, cfg.n_layers_decode, LayerNorm(cfg.d_model))
        
        self.fcn = ArgsFCN(cfg)

        # 在 13D 向量中，Extrude 命令从索引 6 开始有 7 个参数。
        # 这意味着我们稠密的 12 元素参数向量中的参数在索引 5-11。
        # 角度标记是第 6 个参数（索引 5），位置标记是第 7 个参数（索引 6）。
        self.angle_token_idx = 5
        self.pos_token_idx = 6

        self.angle_head = nn.Linear(cfg.args_dim, cfg.n_angle_tokens)
        self.pos_head = nn.Linear(cfg.args_dim, cfg.n_pos_tokens)

    def forward(self, z, guidance, memory_key_padding_mask=None):
        src = self.embedding(z)
        src = src + guidance  # 注入来自命令解码器的引导
        out = self.decoder(src, z, memory_key_padding_mask=memory_key_padding_mask)

        # args_features: [S, N, cad_n_args, args_dim]
        args_features = self.fcn(out)

        angle_token_logits = self.angle_head(args_features[:, :, self.angle_token_idx, :])
        pos_token_logits = self.pos_head(args_features[:, :, self.pos_token_idx, :])

        return args_features, angle_token_logits, pos_token_logits

# --- 顶级解码器 ---

class LLM2CADDecoder(nn.Module):
    """双解码器，将 LLM 特征转换为 CAD 序列。"""
    def __init__(self, cfg: MechCADConfig):
        super().__init__()
        self.d_model = cfg.d_model
        self.adapter = nn.Linear(cfg.llm_hidden_dim, self.d_model)
        self.command_decoder = CommandDecoder(cfg)
        self.args_decoder = ArgsDecoder(cfg)

    def forward(self, llm_features, memory_key_padding_mask=None):
        z = self.adapter(llm_features)
        z = z.permute(1, 0, 2)  # [LLM_Seq_Len, Batch, d_model]

        command_logits, guidance = self.command_decoder(z, memory_key_padding_mask)
        args_features, angle_logits, pos_logits = self.args_decoder(z, guidance, memory_key_padding_mask)

        # 置换到 [Batch, Seq_Len, ...] 以进行损失计算
        command_logits = command_logits.permute(1, 0, 2)
        args_features = args_features.permute(1, 0, 2, 3)
        angle_logits = angle_logits.permute(1, 0, 2)
        pos_logits = pos_logits.permute(1, 0, 2)

        return command_logits, args_features, angle_logits, pos_logits

# --- 完整 MLLM 模型 ---

class MechCADModel(nn.Module):
    """
    完整的 MechCAD 模型，将多模态输入转换为 CAD 序列。
    """
    def __init__(self, cfg: MechCADConfig, llava_model_name="model_weights/llava-hf/llava-1.5-7b-hf"):
        super().__init__()
        self.cfg = cfg
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
        self.llava_model.eval()  # Freeze the MLLM encoder

        # --- Decoder ---
        self.llm2cad_decoder = LLM2CADDecoder(cfg)

    def forward(self, batch):
        """
        处理来自 OmniCADDataset 的一个批次。
        """
        images_tensor = batch['images']  # (B, 8, C, H, W)
        texts = batch['text_caption']    # 字符串列表
        
        # --- 为 LLaVA 准备输入 ---
        # 为简起见，我们仅使用每个 CAD 模型的第一个视图。
        # 更高级的策略可以涉及多个 <image> 令牌。
        pil_images = [Image.fromarray((images_tensor[i, 0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)) for i in range(len(texts))]
        
        # 创建 LLaVA 风格的提示
        prompts = [f"USER: <image>\n{caption} ASSISTANT:" for caption in texts]

        inputs = self.processor(text=prompts, images=pil_images, return_tensors="pt", padding=True)
        
        # 将输入移到与模型相同的设备上
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.llava_model.device)

        # --- 使用 LLaVA 编码 ---
        with torch.no_grad():
            outputs = self.llava_model(**inputs, output_hidden_states=True)
            llm_features = outputs.hidden_states[-1]
            
            # 处理器的注意掩码对应于提示的文本部分。
            # 我们直接将其用作解码器的存储键填充掩码。
            # LLaVA 内部为其自己的注意处理完整掩码。
            memory_key_padding_mask = (inputs.attention_mask == 0)

        # --- 解码为 CAD 序列 ---
        # 将解码器移到设备并匹配精度（如需要）
        self.llm2cad_decoder.to(llm_features.device, dtype=llm_features.dtype)
        
        return self.llm2cad_decoder(llm_features, memory_key_padding_mask)

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
        command_logits, args_features, angle_logits, pos_logits = model(real_batch)

        # 5. 打印输出形状以验证
        print("\n--- 输出形状 ---")
        print(f"命令逻辑: {command_logits.shape}")
        print(f"参数特征:  {args_features.shape}")
        print(f"角度逻辑:   {angle_logits.shape}")
        print(f"位置逻辑:  {pos_logits.shape}")
        
        # 预期形状
        batch_size = len(real_batch['id'])
        print(f"\n预期命令逻辑形状: ({batch_size}, {config.cad_max_total_len}, {config.cad_n_commands})")
        print(f"预期参数特征形状: ({batch_size}, {config.cad_max_total_len}, {config.cad_n_args}, {config.args_dim})")
        
    except Exception as e:
        print(f"\n前向传播期间发生错误: {e}")
        import traceback
        traceback.print_exc()
        print("\n如果您没有足够的 GPU 内存、缺少所需的软件包或数据路径不正确，这是正常的。")

