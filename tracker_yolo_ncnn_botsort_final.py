"""
tracker_ncnn_botsort_evo.py — NCNN + BoT-SORT EVO v2
=====================================================
Key changes in v2:
  • The model processes ONLY the cropped Region of Interest (ROI), not the entire frame
  • Interactive CLI input for ROI coordinates (--roi-input)
  • Supports camera as source (--cam 0) or video file (-v)
  • Retains full features: BoT-SORT, sequential IDs, JSON analytics, telemetry

Usage:
  python tracker_ncnn_botsort_evo.py -v test_video2.mp4
  python tracker_ncnn_botsort_evo.py --cam 0 --duration 60
  python tracker_ncnn_botsort_evo.py -v test_video1.mp4 --roi-input
  python tracker_ncnn_botsort_evo.py -v test_video1.mp4 --roi 0.50,0.25 0.66,0.25 0.69,0.95 0.16,0.95
"""

import os
import cv2
import time
import json
import argparse
import psutil
import numpy as np
from collections import deque, defaultdict
from ultralytics import YOLO

# ─── CPU OPTIMIZATIONS FOR RASPBERRY PI 4 ─────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"]      = "4"
os.environ["OPENBLAS_CORETYPE"]    = "ARMV8"
cv2.setNumThreads(4)

# ─── 1. GENERATE BOTSORT CONFIGURATION ────────────────────────────────────────
BOTSORT_CFG_PATH = "./botsort_small.yaml"
botsort_yaml_content = """
tracker_type: botsort
track_high_thresh: 0.20
track_low_thresh: 0.05
new_track_thresh: 0.70
track_buffer: 1200
match_thresh: 0.60
fuse_score: True
gmc_method: none
proximity_thresh: 0.5
appearance_thresh: 0.25
with_reid: False
model: auto
"""
with open(BOTSORT_CFG_PATH, "w", encoding="utf-8") as f:
    f.write(botsort_yaml_content.strip())
print(f"[Config] BoT-SORT saved: {BOTSORT_CFG_PATH}")


# ─── 2. COMMAND LINE ARGUMENTS ────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="YOLO NCNN + BoT-SORT EVO v2 (ROI-only inference)",
    formatter_class=argparse.RawTextHelpFormatter
)
# Video Source
parser.add_argument("-v", "--video",     default="",                          help="Input video (if not set, uses camera)")
parser.add_argument("--cam",            type=int,   default=-1,              help="Camera index (e.g.: --cam 0)")
parser.add_argument("-s", "--stream",    default="",                          help="Network stream (e.g.: http://10.0.0.1:8080/video)")
parser.add_argument("--duration",       type=int,   default=0,               help="Recording duration from camera in seconds (0=infinite)")
parser.add_argument("--start-frame",    type=int,   default=0,               help="Frame to start processing from (video only)")
parser.add_argument("--end-frame",      type=int,   default=0,               help="Frame to end processing at (video only)")

# Model and parameters
parser.add_argument("-m", "--model",     default="./yolo26n_ncnn_model",      help="NCNN or TFLite model path")
parser.add_argument("-o", "--output",    default="./output_botsort_evo.mp4",  help="Output video file")
parser.add_argument("--json-out",        default="./tracks_summary.json",     help="JSON tracks report file")
parser.add_argument("--conf",            type=float, default=0.20,            help="Confidence threshold (default: 0.20)")
parser.add_argument("--iou",             type=float, default=0.50,            help="NMS IoU threshold (default: 0.50)")
parser.add_argument("--imgsz",           type=int,   default=640,             help="Inference size for YOLO (default: 640)")
parser.add_argument("--skip",            type=int,   default=2,               help="Process every N-th frame (default: 2)")
parser.add_argument("--min-hits",       type=int,   default=2,               help="Minimum frames to show track ID (default: 2)")
parser.add_argument("-c", "--classes",   nargs="+", type=int, default=None,   help="COCO classes filter (e.g.: -c 2 3 7)")
parser.add_argument("--classes-all",    action="store_true",                  help="Track all classes")

# Visualization
parser.add_argument("--draw-trail",     type=int,   default=0,               help="Number of trajectory frames to draw (0 = disabled)")
parser.add_argument("--draw-vector",    action="store_true",                 help="Draw motion vector (arrow)")

# ROI
parser.add_argument("--roi-input",      action="store_true",                  help="Interactive ROI input via console")
parser.add_argument(
    "--roi", nargs=4, type=str, default=None,
    help="4 ROI points in relative coords (x,y): --roi 0.50,0.25 0.66,0.25 0.69,0.95 0.16,0.95"
)

args = parser.parse_args()
if args.classes_all:
    args.classes = None


# ─── 3. UTILITIES ─────────────────────────────────────────────────────────────
def cpu_temp() -> str:
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return f"{int(f.read().strip()) / 1000:.1f}°C"
    except Exception:
        return "N/A"

def cpu_temp_val() -> float:
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        return 0.0

def resize_to(frame: np.ndarray, max_side: int):
    h, w = frame.shape[:2]
    scale = min(max_side / w, max_side / h)
    if scale >= 1.0:
        return frame, 1.0
    nw, nh = int(w * scale), int(h * scale)
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR), scale


def parse_roi_from_cli(roi_args, fw, fh):
    """Parse 4 ROI points from CLI arguments: ['0.50,0.25', '0.66,0.25', ...]"""
    pts = []
    for p in roi_args:
        x_str, y_str = p.split(",")
        pts.append([int(float(x_str) * fw), int(float(y_str) * fh)])
    return np.array(pts, dtype=np.int32)


def input_roi_interactive(fw, fh):
    """Interactive ROI input via console."""
    print("\n" + "="*60)
    print("  REGION OF INTEREST (ROI) INPUT")
    print("="*60)
    print(f"  Frame size: {fw}x{fh}")
    print("  Enter 4 points in RELATIVE coordinates (from 0.0 to 1.0)")
    print("  Format: x,y  (example: 0.50,0.25)")
    print()

    labels = [
        "Top-left      (1/4)",
        "Top-right     (2/4)",
        "Bottom-right  (3/4)",
        "Bottom-left   (4/4)",
    ]
    defaults = ["0.15,0.25", "0.85,0.25", "0.95,0.95", "0.05,0.95"]

    pts = []
    for i, (label, default) in enumerate(zip(labels, defaults)):
        while True:
            val = input(f"  {label} [{default}]: ").strip()
            if not val:
                val = default
            try:
                x_str, y_str = val.split(",")
                x_rel, y_rel = float(x_str), float(y_str)
                pts.append([int(x_rel * fw), int(y_rel * fh)])
                break
            except Exception:
                print("    ⚠ Invalid format! Use: 0.50,0.25")

    roi = np.array(pts, dtype=np.int32)
    print(f"\n  ROI points (px): {pts}")
    print("="*60 + "\n")
    return roi


# ─── 4. MODEL AND VIDEO SOURCE INITIALIZATION ─────────────────────────────────
if not os.path.exists(args.model):
    raise FileNotFoundError(f"Model not found: '{args.model}'")

print(f"[Model]  Loading: {args.model}")
model   = YOLO(args.model)
process = psutil.Process()

# Determine video source: file / camera / stream
if args.stream:
    source = args.stream
    source_name = f"Stream: {args.stream}"
elif args.video:
    if not os.path.isfile(args.video):
        raise FileNotFoundError(f"Video not found: '{args.video}'")
    source = args.video
    source_name = f"Video: {args.video}"
elif args.cam >= 0:
    source = args.cam
    source_name = f"Camera: {args.cam}"
else:
    # Default to test video
    args.video = "./test_video1.mp4"
    if not os.path.isfile(args.video):
        raise FileNotFoundError(f"Source not specified! Use -v, --cam, or --stream")
    source = args.video
    source_name = f"Video: {args.video}"

cap = cv2.VideoCapture(source)
if not cap.isOpened():
    raise RuntimeError(f"Failed to open source: {source}")

W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
FPS   = cap.get(cv2.CAP_PROP_FPS) or 25.0
TOTAL = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Scaling — calculate output frame size
scale = min(args.imgsz / W, args.imgsz / H)
if scale >= 1.0: scale = 1.0
OUT_W, OUT_H = int(W * scale), int(H * scale)

# ─── 5. DEFINE ROI (in original frame coordinates) ────────────────────────────
if args.roi_input:
    roi_points_orig = input_roi_interactive(W, H)
elif args.roi:
    roi_points_orig = parse_roi_from_cli(args.roi, W, H)
else:
    # Default — covers most of the frame
    roi_points_orig = np.array([
        [int(W * 0.15), int(H * 0.25)],
        [int(W * 0.85), int(H * 0.25)],
        [int(W * 0.95), int(H * 0.95)],
        [int(W * 0.05), int(H * 0.95)]
    ], dtype=np.int32)

# For displaying on the output (downscaled) video
roi_points_out = (roi_points_orig * scale).astype(np.int32)

# Calculate ROI bounding box for cropping (from original frame)
roi_x_min = max(0, int(roi_points_orig[:, 0].min()))
roi_y_min = max(0, int(roi_points_orig[:, 1].min()))
roi_x_max = min(W, int(roi_points_orig[:, 0].max()))
roi_y_max = min(H, int(roi_points_orig[:, 1].max()))
roi_w = roi_x_max - roi_x_min
roi_h = roi_y_max - roi_y_min

# ROI points in cropped fragment coordinates (for pointPolygonTest)
roi_points_local_orig = roi_points_orig.copy()
roi_points_local_orig[:, 0] -= roi_x_min
roi_points_local_orig[:, 1] -= roi_y_min

# ─── WRITER ──────────────────────────────────────────────────────────────────
out_fps = FPS / max(1, args.skip)
writer = cv2.VideoWriter(
    args.output,
    cv2.VideoWriter_fourcc(*"mp4v"),
    out_fps,
    (OUT_W, OUT_H),
)

# Calculate the actual tensor size processed internally by YOLO
scale_yolo = args.imgsz / max(roi_w, roi_h)
scaled_w = int(np.round(roi_w * scale_yolo))
scaled_h = int(np.round(roi_h * scale_yolo))
yolo_tensor_w = int(np.ceil(scaled_w / 32.0)) * 32
yolo_tensor_h = int(np.ceil(scaled_h / 32.0)) * 32

print(f"[Source] {source_name} (Frame: {W}x{H} → Output: {OUT_W}x{OUT_H} @ {FPS:.0f} FPS)")
print(f"[ROI]    Original frame cropped area: {roi_w}x{roi_h} px (area saved {100 - 100*roi_w*roi_h/(W*H):.0f}%)")
print(f"[YOLO]   Actual neural network size (with imgsz={args.imgsz}): {yolo_tensor_w}x{yolo_tensor_h} px")
print(f"[Output] {args.output}")
print(f"[Opt]    BoT-SORT | Skip={args.skip} | Imgsz={args.imgsz} | Conf={args.conf}")
if args.classes is not None:
    print(f"[Filter] Classes: {args.classes}")
else:
    print(f"[Filter] All classes")

print(f"{'─'*75}")
print(f"{'Frame':>6}  {'Lat(ms)':>9}  {'FPS':>6}  {'Tracks':>7}  {'CPU%':>6}  {'RAM MB':>8}  {'Temp':>7}")
print(f"{'─'*75}")

# ─── 6. TRACKING STATE ────────────────────────────────────────────────────────
latencies  = []
cpu_vals   = []
ram_vals   = []
temp_vals  = []
win_fps    = deque(maxlen=30)
read_idx   = 0
write_idx  = 0

if args.start_frame > 0 and not args.stream and args.cam < 0:
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame - 1)
    read_idx = args.start_frame - 1
    print(f"[Info] Skipping to frame {args.start_frame}...")

display_id_map  = {}
display_counter = 0
track_hits      = defaultdict(int)

tracks_lifecycle = defaultdict(lambda: {
    "class_name": "",
    "frames": [],
    "confs": [],
    "trajectory": []
})

t_start = time.time()

# ─── 7. MAIN PROCESSING LOOP ──────────────────────────────────────────────────
try:
    while True:
        ok, frame_orig = cap.read()
        if not ok:
            break

        read_idx += 1

        # Skip frames for stream if start_frame is set
        if args.start_frame > 0 and read_idx < args.start_frame:
            continue

        # Check end frame
        if args.end_frame > 0 and read_idx > args.end_frame:
            print(f"\n[Info] Reached end frame: {args.end_frame}")
            break

        # Check duration (for camera)
        if args.duration > 0 and (time.time() - t_start) >= args.duration:
            break

        if args.skip > 1 and read_idx % args.skip != 0:
            continue

        # ── 1. CROP ROI FROM ORIGINAL FRAME ───────────────────────────────────
        # Crop the region of interest directly from the original frame (max detail)
        roi_crop_orig = frame_orig[roi_y_min:roi_y_max, roi_x_min:roi_x_max].copy()

        t0 = time.perf_counter()

        # Inference ONLY on the cropped ROI zone
        # YOLO will automatically scale/optimize this fragment to args.imgsz
        results = model.track(
            source=roi_crop_orig,
            persist=True,
            tracker=BOTSORT_CFG_PATH,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            classes=args.classes,
            device="cpu",
            verbose=False,
        )[0]

        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)
        win_fps.append(lat)
        cur_fps = 1000.0 / (sum(win_fps) / len(win_fps))

        # Collect telemetry
        cpu_pct = psutil.cpu_percent(interval=None)
        ram_mb  = process.memory_info().rss / 1024 / 1024
        temp    = cpu_temp()
        tv      = cpu_temp_val()
        cpu_vals.append(cpu_pct)
        ram_vals.append(ram_mb)
        if tv > 0:
            temp_vals.append(tv)

        # ── 2. PREPARE OUTPUT FRAME (downscale full frame) ────────────────────
        frame_out, _ = resize_to(frame_orig, args.imgsz)

        # ── Visualize ROI on the downscaled frame ─────────────────────────────
        overlay = frame_out.copy()
        cv2.fillPoly(overlay, [roi_points_out], color=(0, 255, 0))
        cv2.addWeighted(overlay, 0.08, frame_out, 0.92, 0, dst=frame_out)
        cv2.polylines(frame_out, [roi_points_out], isClosed=True, color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA)

        # ── Annotation: map coordinates from ROI-crop back to output frame ────
        n_tracks = 0
        current_frame_no = write_idx + 1

        if results.boxes is not None and results.boxes.id is not None:
            bxs  = results.boxes.xyxy.cpu().numpy()
            ids  = results.boxes.id.cpu().numpy().astype(int)
            cfs  = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy().astype(int)

            for box, raw_tid, cf, cls_id in zip(bxs, ids, cfs, clss):
                # Coordinates in crop → coordinates in full ORIGINAL frame
                cx1_orig = box[0] + roi_x_min
                cy1_orig = box[1] + roi_y_min
                cx2_orig = box[2] + roi_x_min
                cy2_orig = box[3] + roi_y_min

                # ROI check using polygon (in original crop coordinates)
                local_cx = int((box[0] + box[2]) / 2)
                local_cy = int(box[3])
                if cv2.pointPolygonTest(roi_points_local_orig, (local_cx, local_cy), False) < 0:
                    continue

                # Scale coordinates for drawing on the downscaled frame_out
                cx1 = int(cx1_orig * scale)
                cy1 = int(cy1_orig * scale)
                cx2 = int(cx2_orig * scale)
                cy2 = int(cy2_orig * scale)

                if (cx2 - cx1) < 2 or (cy2 - cy1) < 2:
                    continue

                # Global center coordinates (true object center)
                global_cx = int((cx1 + cx2) / 2)
                global_cy = int((cy1 + cy2) / 2)

                track_hits[raw_tid] += 1

                if track_hits[raw_tid] >= args.min_hits:
                    if raw_tid not in display_id_map:
                        display_counter += 1
                        display_id_map[raw_tid] = display_counter

                    disp_id = display_id_map[raw_tid]
                    name    = results.names[cls_id]
                    lbl     = f"#{disp_id} {name} {cf:.2f}"

                    # JSON
                    t_data = tracks_lifecycle[disp_id]
                    t_data["class_name"] = name
                    t_data["frames"].append(current_frame_no)
                    t_data["confs"].append(float(cf))
                    t_data["trajectory"].append({
                        "frame": current_frame_no,
                        "x": global_cx,
                        "y": global_cy
                    })

                    # ── Draw trajectory (trail) ──
                    if args.draw_trail > 0:
                        traj = t_data["trajectory"]
                        recent_pts = traj[-args.draw_trail:]
                        for i in range(1, len(recent_pts)):
                            pt1 = (recent_pts[i-1]["x"], recent_pts[i-1]["y"])
                            pt2 = (recent_pts[i]["x"], recent_pts[i]["y"])
                            cv2.line(frame_out, pt1, pt2, (0, 255, 255), 2, cv2.LINE_AA)
                    
                    # ── Draw motion vector ──
                    if args.draw_vector:
                        traj = t_data["trajectory"]
                        # Draw vector ONLY after 7 frames of object existence,
                        # to stabilize its size and direction
                        if len(traj) >= 7:
                            past_pt = traj[-7]
                            dx = global_cx - past_pt["x"]
                            dy = global_cy - past_pt["y"]
                            
                            # Ignore micro-jitters (if object is stationary)
                            if abs(dx) > 2 or abs(dy) > 2:
                                # Draw arrow (scale length x2 for visibility)
                                end_pt = (global_cx + dx * 2, global_cy + dy * 2)
                                cv2.arrowedLine(frame_out, (global_cx, global_cy), end_pt, (0, 0, 255), 2, tipLength=0.3)

                    # Draw on downscaled frame
                    cv2.rectangle(frame_out, (cx1, cy1), (cx2, cy2), (0, 255, 0), 1)
                    (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                    cv2.rectangle(frame_out, (cx1, max(0, cy1 - th - 4)), (cx1 + tw + 2, cy1), (0, 0, 0), -1)
                    cv2.putText(frame_out, lbl, (cx1 + 1, max(th + 1, cy1 - 2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

                    n_tracks += 1

        # ── Metrics Panel ─────────────────────────────────────────────────────
        cv2.rectangle(frame_out, (0, 0), (OUT_W, 56), (0, 0, 0), -1)
        cv2.putText(frame_out,
                    f"Frame:{write_idx+1}  FPS:{cur_fps:.1f}  Lat:{lat:.0f}ms  Tracks:{n_tracks}",
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(frame_out,
                    f"CPU:{cpu_pct:.0f}%  RAM:{ram_mb:.0f}MB  Temp:{temp}  ROI-Orig(High-Res)",
                    (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 255), 1, cv2.LINE_AA)

        writer.write(frame_out)
        write_idx += 1

        if write_idx % 10 == 0 or write_idx == 1:
            print(f"{write_idx:>6}  {lat:>9.1f}  {cur_fps:>6.1f}  {n_tracks:>7}"
                  f"  {cpu_pct:>5.0f}%  {ram_mb:>7.0f}  {temp:>7}")

except KeyboardInterrupt:
    print("\n[Stop] Ctrl+C — stopping")

cap.release()
writer.release()

# ─── 8. SAVE JSON REPORT ──────────────────────────────────────────────────────
events_summary = []
for disp_id, data in sorted(tracks_lifecycle.items()):
    if not data["frames"]:
        continue
    start_f = data["frames"][0]
    end_f   = data["frames"][-1]
    dur_frames = end_f - start_f + 1
    dur_sec    = dur_frames / out_fps if out_fps > 0 else 0.0
    roi_sec    = len(data["frames"]) / out_fps if out_fps > 0 else 0.0
    avg_conf   = float(np.mean(data["confs"])) if data["confs"] else 0.0

    events_summary.append({
        "track_id": int(disp_id),
        "class_name": data["class_name"],
        "start_frame": int(start_f),
        "end_frame": int(end_f),
        "duration_frames": int(dur_frames),
        "duration_seconds": round(dur_sec, 2),
        "average_confidence": round(avg_conf, 2),
        "entered_roi": True,
        "time_in_roi_seconds": round(roi_sec, 2),
        "trajectory": data["trajectory"]
    })

with open(args.json_out, "w", encoding="utf-8") as jf:
    json.dump(events_summary, jf, indent=2, ensure_ascii=False)

# ─── 9. SUMMARY ───────────────────────────────────────────────────────────────
if latencies:
    avg_fps = 1000.0 / np.mean(latencies)
    p50     = np.percentile(latencies, 50)
    p95     = np.percentile(latencies, 95)

    print(f"\n{'═'*60}")
    print("  EXECUTION SUMMARY (BoT-SORT EVO v2 — ROI-Only Inference)")
    print(f"{'═'*60}")
    print(f"  Source             : {source_name}")
    print(f"  Model              : {args.model}")
    print(f"  ROI area           : {roi_w}x{roi_h} px (instead of {OUT_W}x{OUT_H})")
    print(f"  Processed frames   : {write_idx}")
    print(f"  Unique objects     : {display_counter}")
    print(f"  Average FPS        : {avg_fps:.2f} fps")
    print(f"  Latency p50        : {p50:.1f} ms")
    print(f"  Latency p95        : {p95:.1f} ms")
    print(f"  CPU avg/peak       : {np.mean(cpu_vals):.0f}% / {np.max(cpu_vals):.0f}%")
    print(f"  RAM avg/peak       : {np.mean(ram_vals):.0f} MB / {np.max(ram_vals):.0f} MB")
    if temp_vals:
        print(f"  Temp avg/peak      : {np.mean(temp_vals):.1f}°C / {np.max(temp_vals):.1f}°C")
    print(f"  Video saved        : {args.output}")
    print(f"  JSON saved         : {args.json_out}")
    print(f"{'═'*60}")
