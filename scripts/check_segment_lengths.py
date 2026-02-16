import json, numpy as np
from pathlib import Path

lengths_by_gesture = {}
users_to_check = [f"user{i}" for i in [1, 10, 50, 100, 150, 200, 250, 300]]

for u in users_to_check:
    p = Path(f"EMG-EPN612 Dataset/trainingJSON/{u}/{u}.json")
    if not p.exists():
        continue
    data = json.load(open(p))
    for sid, s in data["trainingSamples"].items():
        g = s["gestureName"]
        if g == "noGesture":
            continue
        gti = s.get("groundTruthIndex")
        if gti:
            seg_len = gti[1] - gti[0] + 1
            lengths_by_gesture.setdefault(g, []).append(seg_len)

print("Gesture segment lengths (samples @ 200 Hz):")
header = f"{'Gesture':12s} {'Mean':>6s} {'Std':>6s} {'Min':>5s} {'Max':>5s} {'Med':>5s}  Win(40/10)"
print(header)
print("-" * len(header))

all_lens = []
for g in sorted(lengths_by_gesture):
    arr = np.array(lengths_by_gesture[g])
    all_lens.extend(arr.tolist())
    n_win = int((np.mean(arr) - 40) // 10 + 1)
    print(f"{g:12s} {np.mean(arr):6.1f} {np.std(arr):6.1f} {np.min(arr):5d} {np.max(arr):5d} {np.median(arr):5.0f}  ~{n_win}")

all_arr = np.array(all_lens)
n_win_all = int((np.mean(all_arr) - 40) // 10 + 1)
print("-" * len(header))
print(f"{'ALL':12s} {np.mean(all_arr):6.1f} {np.std(all_arr):6.1f} {np.min(all_arr):5d} {np.max(all_arr):5d} {np.median(all_arr):5.0f}  ~{n_win_all}")

ng_crop = 260
ng_win = int((ng_crop - 40) // 10 + 1)
print(f"\nCurrent noGesture crop: {ng_crop} samples = {ng_crop/200:.2f}s  -> ~{ng_win} windows")
print(f"Avg gesture segment:   {np.mean(all_arr):.0f} samples = {np.mean(all_arr)/200:.2f}s  -> ~{n_win_all} windows")
