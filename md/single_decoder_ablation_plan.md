# Single Decoder Ablation (Shared Decoder)

## 1. 改动摘要
- 分支: `ablation/single-decoder`
- 目标: 删除双解码器（`CommandDecoder + ArgsDecoder`）架构，改为共享单解码器主干 + 双输出头。
- 输出接口保持不变:
  - `outputs["command_logits"]`
  - `outputs["unified_args_logits"]`

## 2. 与原双解码差异
- 原实现:
  - 命令解码器输出命令 logits + guidance
  - 参数解码器使用 guidance 注入后预测参数 token
- 当前实现:
  - 共享 `TransformerDecoder` 主干一次解码
  - 同时接:
    - 命令头（命令分类）
    - 参数头（统一词表 token 分类）
- 不再存在 guidance 依赖路径。

## 3. Checkpoint 兼容性
- 新 checkpoint 会写入 `decoder_arch=single_shared`。
- 旧 dual-decoder checkpoint 与本分支 decoder 结构默认不兼容。
- 兼容性报错提示已在以下加载路径补充:
  - `src/trainer/trainer.py`
  - `src/inference.py`
  - `src/app.py`

## 4. 快速冒烟命令
```bash
python -m py_compile src/model/model.py src/trainer/trainer.py src/inference.py src/app.py

python src/train.py \
  --text_only \
  --epochs 1 \
  --batch_size 1 \
  --sample_limit 8 \
  --num_workers 0 \
  --eval_every 1 \
  --save_every 1 \
  --model_dir outputs/ablation_single/checkpoints \
  --log_dir outputs/ablation_single/logs

python src/evaluate_checkpoint.py \
  --checkpoint outputs/ablation_single/checkpoints/best.pth \
  --text_only \
  --split test \
  --sample_limit 8 \
  --batch_size 1 \
  --num_workers 0 \
  --skip_full_eval
```
