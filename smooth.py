from ultralytics import YOLO
import cv2
import numpy as np
import sys
sys.path.insert(0, 'src')
from smoothing import OneEuroFilter

model = YOLO('yolov8x-pose.pt')

# MORE AGGRESSIVE SMOOTHING
# Lower min_cutoff = smoother (was 0.8, now 0.3)
# Lower beta = less reactive to speed (was 0.4, now 0.1)
smoother = OneEuroFilter(min_cutoff=0.3, beta=0.1)

cap = cv2.VideoCapture('videos/josue.MOV')
fps = cap.get(cv2.CAP_PROP_FPS)
w, h = int(cap.get(3)), int(cap.get(4))

out = cv2.VideoWriter('smoothed_v2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame, conf=0.3, verbose=False)
    
    if results[0].keypoints is not None and len(results[0].keypoints) > 0:
        kps = results[0].keypoints.xy.cpu().numpy()
        scores = results[0].keypoints.conf.cpu().numpy()
        
        smoothed_kps = smoother.smooth(kps, scores)
        
        # Draw skeleton
        skeleton = [(5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
        for start, end in skeleton:
            pt1 = tuple(smoothed_kps[0][start].astype(int))
            pt2 = tuple(smoothed_kps[0][end].astype(int))
            cv2.line(frame, pt1, pt2, (0,255,0), 2)
        
        for j in range(17):
            pt = tuple(smoothed_kps[0][j].astype(int))
            cv2.circle(frame, pt, 4, (0,255,255), -1)
    
    out.write(frame)
    frame_count += 1
    if frame_count % 50 == 0:
        print(f'Processed {frame_count} frames...')

cap.release()
out.release()
print('Saved smoothed_v2.mp4')
