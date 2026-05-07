import torch
import sys
import os
from pathlib import Path

# 配置
CKPT_PATH = "experiments_out/2026-04-24/15-47-24/model_latest.pth"
print(f"[RESUME] Loading from: {CKPT_PATH}")

# 加载 checkpoint
ckpt = torch.load(CKPT_PATH, map_location='cpu')
iteration = ckpt.get('iteration', 0)
print(f"[RESUME] Found iteration: {iteration}")
print(f"[RESUME] Best PSNR so far: {ckpt.get('best_PSNR', 'unknown')}")

# 保存恢复信息到临时文件，供 train_network.py 读取
resume_info = {
    'ckpt_path': CKPT_PATH,
    'start_iteration': iteration,
    'optimizer_state': ckpt.get('optimizer_state_dict', None),
}
torch.save(resume_info, '.resume_info.pth')
print("[RESUME] Resume info saved to .resume_info.pth")

# 现在需要修改 train_network.py 来读取这个文件
# 请手动在 train_network.py 的 model 初始化后添加：
"""
resume_info = torch.load('.resume_info.pth', map_location='cpu') if os.path.exists('.resume_info.pth') else None
if resume_info:
    ckpt = torch.load(resume_info['ckpt_path'], map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    if 'optimizer_state_dict' in ckpt:
        opt.load_state_dict(ckpt['optimizer_state_dict'])
    start_iteration = resume_info['start_iteration']
    print(f'[RESUME] Resuming from iteration {start_iteration}')
else:
    start_iteration = 0
"""

print("\n[INFO] Please add resume logic to train_network.py before running:")
print("  1. Find 'model = GaussianSplatPredictor(cfg)' in train_network.py")
print("  2. After model initialization, add the code block above")
print("  3. Change 'for iteration in range(opt.iterations):' to 'for iteration in range(start_iteration, opt.iterations):'")
