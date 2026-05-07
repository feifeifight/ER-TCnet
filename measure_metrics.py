import torch
import hydra
import sys
import os
import numpy as np

sys.path.append(os.getcwd())

from scene.gaussian_predictor import GaussianSplatPredictor


def count_flops(model, input_images, source_cameras_view_to_world, source_cv2wT_quat, focals_pixels):
    """Count FLOPs using forward hooks — handles both standard nn.* and EDM custom layers."""

    # Pre-compute module → component category mapping
    module_cat = {}
    for name, mod in model.named_modules():
        if 'gwformer' in name.lower():
            module_cat[id(mod)] = 'GWformer'
        elif 'er_tcnet' in name.lower():
            module_cat[id(mod)] = 'ER-TCNet'
        else:
            module_cat[id(mod)] = 'UNet'

    flops = {}       # by type: Conv, Linear, Attention, Norm
    flops_cat = {}   # by component: GWformer, ER-TCNet, UNet

    def conv_hook(module, input, output):
        w = getattr(module, 'weight', None)
        if w is None or w.ndim < 3:
            return
        k = w.shape[-1]
        in_ch = w.shape[1]
        # FLOPs = 2 * kernel^2 * in_ch * output_elements
        total = 2 * k * k * in_ch * output.numel()
        flops['Conv'] = flops.get('Conv', 0) + total
        cat = module_cat.get(id(module), 'UNet')
        flops_cat[cat] = flops_cat.get(cat, 0) + total

    def linear_hook(module, input, output):
        in_f = getattr(module, 'in_features', None)
        if in_f is None:
            return
        total = 2 * in_f * output.numel()
        flops['Linear'] = flops.get('Linear', 0) + total
        cat = module_cat.get(id(module), 'UNet')
        flops_cat[cat] = flops_cat.get(cat, 0) + total

    def mha_hook(module, input, output):
        q = input[0]
        d = module.embed_dim
        B = q.shape[0]
        T = q.shape[1]
        # QKV projection + attention scores + attn×V
        total = (6 * d * d + 4 * d * T) * T * B
        flops['Attention'] = flops.get('Attention', 0) + total
        cat = module_cat.get(id(module), 'UNet')
        flops_cat[cat] = flops_cat.get(cat, 0) + total

    def norm_hook(module, input, output):
        if isinstance(output, torch.Tensor):
            total = 2 * output.numel()
            flops['Norm'] = flops.get('Norm', 0) + total
            cat = module_cat.get(id(module), 'UNet')
            flops_cat[cat] = flops_cat.get(cat, 0) + total

    hooks = []
    for module in model.modules():
        tname = type(module).__name__
        if tname == 'Conv2d':
            hooks.append(module.register_forward_hook(conv_hook))
        elif tname == 'Conv1d':
            hooks.append(module.register_forward_hook(conv_hook))
        elif tname == 'MultiheadAttention':
            hooks.append(module.register_forward_hook(mha_hook))
        elif tname == 'Linear':
            hooks.append(module.register_forward_hook(linear_hook))
        elif tname in ('GroupNorm', 'LayerNorm', 'BatchNorm1d', 'BatchNorm2d'):
            hooks.append(module.register_forward_hook(norm_hook))

    with torch.no_grad():
        model(input_images, source_cameras_view_to_world, source_cv2wT_quat,
              focals_pixels=focals_pixels)

    for h in hooks:
        h.remove()

    return flops, flops_cat


def main():
    is_baseline = '--baseline' in sys.argv
    label = "Baseline" if is_baseline else "Full (GWformer+ER-TCNet)"
    print("=" * 60)
    print(f"Model Metrics Measurement — {label}")
    print("=" * 60)

    print("\nLoading config...")
    overrides = [
        "+dataset=hydrants",
        "general.num_devices=1",
        "general.mixed_precision=false",
    ]
    if is_baseline:
        overrides += ["model.encoder_type=standard", "model.er_tcnet.enabled=false"]
    with hydra.initialize(version_base=None, config_path="configs"):
        cfg = hydra.compose(config_name="default_config", overrides=overrides)

    B, N = 1, 1
    H = W = cfg.data.training_resolution
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}, Resolution: {H}x{W}")

    in_channels = 4 if cfg.data.origin_distances else 3
    input_images = torch.randn(B, N, in_channels, H, W).to(device)
    source_cameras_view_to_world = torch.eye(4).view(1, 1, 4, 4).repeat(B, N, 1, 1).to(device)
    source_cv2wT_quat = torch.tensor([1.0, 0.0, 0.0, 0.0]).view(1, 1, 4).repeat(B, N, 1).to(device)
    focals_pixels = torch.tensor([250.0]).view(B, N, 1).to(device)

    print("Instantiating model...")
    model = GaussianSplatPredictor(cfg).to(device)
    model.eval()

    # 1. Parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n--- Parameters ---")
    print(f"Total params:     {total_params:>12,d} ({total_params / 1e9:.4f} B)")
    print(f"Trainable params: {trainable_params:>12,d} ({trainable_params / 1e9:.4f} B)")

    # Per-module params (with GWformer breakdown)
    print(f"\nPer-module breakdown:")
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        if n > 0:
            print(f"  {name:30s}: {n:>12,d} ({n / 1e6:.2f} M)")
    if hasattr(model, 'network_with_offset'):
        enc = getattr(model.network_with_offset, 'encoder', None)
        if enc and hasattr(enc, 'gwformer_encoder'):
            gw_n = sum(p.numel() for p in enc.gwformer_encoder.parameters())
            unet_n = sum(p.numel() for p in model.network_with_offset.parameters()) - gw_n
            print(f"  {'  ├ UNet (CNN+decoder)':30s}: {unet_n:>12,d} ({unet_n / 1e6:.2f} M)")
            print(f"  {'  └ GWformer encoder':30s}: {gw_n:>12,d} ({gw_n / 1e6:.2f} M)")

    # 2. FLOPs
    print(f"\n--- FLOPs ---")
    flops, flops_cat = count_flops(model, input_images, source_cameras_view_to_world,
                                    source_cv2wT_quat, focals_pixels)
    total_flops = sum(flops.values())

    print(f"\nBy layer type:")
    for key in ['Conv', 'Linear', 'Attention', 'Norm']:
        v = flops.get(key, 0)
        print(f"  {key:20s}: {v / 1e12:>12.4f} T")

    print(f"\nBy component:")
    for key in ['UNet', 'GWformer', 'ER-TCNet']:
        v = flops_cat.get(key, 0)
        print(f"  {key:20s}: {v / 1e12:>12.4f} T")

    print(f"\n  Total FLOPs:         {total_flops / 1e12:>12.4f} T")

    # 3. Inference Time
    print(f"\n--- Inference Time (warming up...) ---")

    for _ in range(30):
        with torch.no_grad():
            model(input_images, source_cameras_view_to_world, source_cv2wT_quat,
                  focals_pixels=focals_pixels)
    torch.cuda.synchronize()

    repeats = 100
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            model(input_images, source_cameras_view_to_world, source_cv2wT_quat,
                  focals_pixels=focals_pixels)
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1000.0)

    times = np.array(times)
    mean_time = times.mean()
    std_time = times.std()

    # Summary
    print(f"\n{'=' * 60}")
    print(f"Summary (B={B}, N={N}, Res={H})")
    print(f"{'=' * 60}")
    print(f"Params:            {total_params / 1e9:.4f} B")
    print(f"FLOPs:             {total_flops / 1e12:.4f} T")
    print(f"Inference Time:    {mean_time:.4f} +/- {std_time:.4f} s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
