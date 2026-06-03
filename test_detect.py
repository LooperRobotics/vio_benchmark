"""
AprilGrid detection diagnostic using aprilgrid package.
Usage: python3 test_detect.py results/debug_start_frame0.png
"""
import sys
import cv2
import numpy as np
from aprilgrid import Detector

path = sys.argv[1] if len(sys.argv) > 1 else 'results/debug_start_frame0.png'
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print(f'Cannot load {path}'); sys.exit(1)

print(f'Image: shape={img.shape}  dtype={img.dtype}  min={img.min()}  max={img.max()}')
print()

for family in ['t36h11', 't36h11b1']:
    det = Detector(family)
    hits = det.detect(img)
    ids  = sorted(d.tag_id for d in hits)
    mark = '  <<<' if hits else ''
    print(f'  {family:<12} → {len(hits):3d} detections  IDs={ids}{mark}')

# annotated image of best result
det = Detector('t36h11')
hits = det.detect(img)
out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
colors = [(0,255,0),(0,128,255),(0,0,255),(255,0,0)]  # 0=green,1=orange,2=red,3=blue
for d in hits:
    corners = np.array(d.corners, dtype=float).reshape(4, 2).astype(int)
    cv2.polylines(out, [corners.reshape(-1, 1, 2)], True, (200, 200, 200), 1)
    cx, cy = corners.mean(axis=0).astype(int)
    cv2.putText(out, str(d.tag_id), (cx - 8, cy + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
    for i, (px, py) in enumerate(corners):
        cv2.circle(out, (px, py), 3, colors[i], -1)
        cv2.putText(out, str(i), (px + 3, py - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, colors[i], 1)
cv2.imwrite('results/debug_annotated.png', out)
print('\nAnnotated image → results/debug_annotated.png')
