import ultralytics
from ultralytics import YOLO
import cv2
from collections import deque
from datetime import datetime
import torch , time
import numpy as np
import csv
import os

model = YOLO("yolov8n.pt")
cam = cv2.VideoCapture(0)
output_dir = "outputs"
os.makedirs(output_dir, exist_ok=True)

video_path = os.path.join(output_dir, "output.mp4")
csv_path = os.path.join(output_dir, "tracking_results.csv")

if not cam.isOpened():
    raise RuntimeError("Cannot open webcam")

track_history={}
max_history=30

counted_ids = set()
track_side={}
in_count = 0
out_count = 0

margin = 10

prev_time = time.perf_counter()
inference_times = deque(maxlen=100)
fps_history = deque(maxlen=30)

frame_number = 0

# Video writer will be initialized after reading first frame
video_writer = None

csv_file= open(csv_path,mode='w',newline='',encoding='utf-8-sig')

csv_writer = csv.DictWriter(
    csv_file,
    fieldnames=[
        "timestamp",
        "frame_number",
        "track_id",
        "class_name",
        "confidence",
        "center_x",
        "center_y",
        "event",
        "fps",
        "inference_time_ms"
    ]
)

csv_writer.writeheader()

try:
    while True:
        ret, frame = cam.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        height , width = frame.shape[:2]
        line_position= height // 2
        if video_writer is None:
            camera_fps = cam.get(cv2.CAP_PROP_FPS)
            if camera_fps <= 0:
                camera_fps = 20.0
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            video_writer = cv2.VideoWriter(
                video_path,
                fourcc,
                camera_fps,
                (width, height)
            )
            if not video_writer.isOpened():
                raise RuntimeError("Cannot create output video")

        inference_start = time.perf_counter()

        results = model.track(frame,persist=True,tracker='bytetrack.yaml',imgsz=416,conf=0.35,device='cpu',half=False,verbose=False)
        inference_end = time.perf_counter()
        inference_time = (
            inference_end - inference_start
        ) * 1000
        inference_times.append(inference_time)
        annotated= results[0].plot()
        boxes = results[0].boxes

        cv2.line(
            annotated,
            (0, line_position),
            (width, line_position),
            (0, 0, 255),
            3
        )
        frame_records = []
        if boxes is not None and boxes.id is not None:
            track_ids = boxes.id.int().cpu().tolist()
            coordinates = boxes.xywh.cpu().numpy()
            class_ids = boxes.cls.int().cpu().tolist()
            confidences = boxes.conf.cpu().tolist()

            for track_id,box,class_id,confidence in zip(track_ids,coordinates,class_ids,confidences):

                center_x = int(box[0])
                center_y = int(box[1])

                class_name = model.names[class_id]

                counted_ids.add(track_id)

                # Trajectory
                if track_id not in track_history:
                    track_history[track_id] = []

                track_history[track_id].append(
                    (center_x, center_y)
                )

                track_history[track_id] = (
                    track_history[track_id][-max_history:]
                )
                points = np.array(track_history[track_id],dtype=np.int32).reshape((-1,1,2))

                if len(points) >1 :
                    cv2.polylines(
                        annotated,
                        [points],
                        False,
                        (255, 0, 255),
                        3
                    )

                if center_y < line_position - margin:
                    current_side = -1

                elif center_y > line_position + margin:
                    current_side = 1

                else:
                    current_side = None

                event = ''
                if current_side is not None:

                    if track_id not in track_side:

                        track_side[track_id] = current_side

                    else:

                        previous_side = track_side[track_id]

                        if previous_side != current_side:

                            if previous_side == -1 and current_side == 1:

                                in_count += 1
                                event = "IN"

                            elif previous_side == 1 and current_side == -1:

                                out_count += 1
                                event = "OUT"

                            track_side[track_id] = current_side

                cv2.circle(
                    annotated,
                    (center_x, center_y),
                    5,
                    (255, 255, 0),
                    -1
                )
                            
                frame_records.append({
                    "track_id": track_id,
                    "class_name": class_name,
                    "confidence": round(float(confidence), 4),
                    "center_x": center_x,
                    "center_y": center_y,
                    "event": event
                })

        current_time =time.perf_counter()
        elapsed_time = current_time - prev_time

        fps = (1 // elapsed_time if elapsed_time > 0 else 0)
        prev_time = current_time

        fps_history.append(fps)

        average_fps = np.mean(fps_history)
        average_inference = np.mean(inference_times)

        frame_number += 1
        cv2.putText(
            annotated,
            f"IN: {in_count}  OUT: {out_count}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            annotated,
            f"Unique IDs: {len(counted_ids)}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            annotated,
            f"FPS: {fps:.2f}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            annotated,
            f"Average FPS: {average_fps:.2f}",
            (20, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            annotated,
            f"Inference: {inference_time:.1f} ms",
            (20, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.putText(
            annotated,
            f"Avg Inference: {average_inference:.1f} ms",
            (20, 215),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )
        timestamp = datetime.now().isoformat(
            timespec="milliseconds"
        )

        for record in frame_records:

            csv_writer.writerow({
                "timestamp": timestamp,
                "frame_number": frame_number,
                **record,
                "fps": round(fps, 2),
                "inference_time_ms": round(inference_time, 2)
            })

        cv2.imshow(
            "YOLO Object Tracking - Recording",
            annotated
        )
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


finally:
    cam.release()

    if video_writer is not None:
        video_writer.release()

    csv_file.close()

    cv2.destroyAllWindows()

    print("\n========== FINAL REPORT ==========")

    print(f"Processed Frames: {frame_number}")
    print(f"Unique Tracking IDs: {len(counted_ids)}")
    print(f"IN Count: {in_count}")
    print(f"OUT Count: {out_count}")

    if inference_times:

        print(
            f"Average Inference: "
            f"{np.mean(inference_times):.2f} ms"
        )

    if fps_history:

        print(
            f"Average FPS: "
            f"{np.mean(fps_history):.2f}"
        )

    print("\nSaved Files:")
    print(f"Video: {video_path}")
    print(f"CSV: {csv_path}")

    print("==================================")
    