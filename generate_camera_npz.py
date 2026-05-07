import os
import json
import gzip
import numpy as np

raw_base = "/mnt/e/reaserach/pattern/triplane/co3d/hydrant_raw/hydrant"
processed_base = "/mnt/e/reaserach/pattern/triplane/co3d/co3d_preprocessed/co3d_hydrant_for_gs/train"

sequences = sorted([d for d in os.listdir(processed_base) if os.path.isdir(os.path.join(processed_base, d))])

all_Rs = []
all_Ts = []

for seq in sequences:
    jgz_path = os.path.join(raw_base, seq, "frame_annotations.jgz")
    if not os.path.exists(jgz_path):
        print(f"[SKIP] No jgz for {seq}")
        continue
    
    with gzip.open(jgz_path, 'rt') as f:
        annotations = json.load(f)
    
    for frame in annotations:
        vp = frame.get('viewpoint', {})
        R = np.array(vp.get('R', np.eye(3)))
        T = np.array(vp.get('T', np.zeros(3)))
        all_Rs.append(R)
        all_Ts.append(T)

np.savez(os.path.join(processed_base, "camera_Ts.npz"), Ts=np.array(all_Ts), Rs=np.array(all_Rs))
np.savez(os.path.join(processed_base, "camera_Rs.npz"), Rs=np.array(all_Rs), Ts=np.array(all_Ts))

print(f"[OK] Generated camera_Ts.npz with {len(all_Ts)} entries from {len(sequences)} sequences")
