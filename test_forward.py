import torch
import hydra
from omegaconf import OmegaConf
import sys
import os

sys.path.append(os.getcwd())

from scene.gaussian_predictor import GaussianSplatPredictor

def test():
    print("Loading config...")
    with hydra.initialize(version_base=None, config_path="configs"):
        cfg = hydra.compose(config_name="default_config", overrides=[
            "+dataset=objaverse",
            "general.num_devices=1",
            "general.mixed_precision=false"
        ])
    
    B, N = 1, 1
    H = W = cfg.data.training_resolution
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    input_images = torch.randn(B, N, 3, H, W).to(device)
    source_cameras_view_to_world = torch.eye(4).view(1, 1, 4, 4).repeat(B, N, 1, 1).to(device)
    source_cv2wT_quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).view(1, 1, 4).repeat(B, N, 1).to(device)
    
    print("Instantiating GaussianSplatPredictor...")
    model = GaussianSplatPredictor(cfg.model).to(device)
    model.eval()
    
    print("Running forward pass...")
    with torch.no_grad():
        outputs = model(
            input_images, 
            source_cameras_view_to_world, 
            source_cv2wT_quat,
            return_aux=True
        )
    
    print("\nSuccess!")
    print("Output dictionary keys:", list(outputs.keys()))
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: shape {v.shape}")
        elif isinstance(v, dict):
            print(f"  {k}: keys {list(v.keys())}")
            for sub_k, sub_v in v.items():
                if isinstance(sub_v, torch.Tensor):
                    print(f"    {sub_k}: shape {sub_v.shape}")

if __name__ == "__main__":
    try:
        test()
    except Exception:
        import traceback
        traceback.print_exc()
# Final mark
print("--- TEST END ---")
