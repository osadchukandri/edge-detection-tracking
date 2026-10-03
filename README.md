# edge-detection-tracking
Edge AI object tracking system optimized for Raspberry Pi. Combines YOLO and BoT-SORT with High-Res ROI-only inference to maximize FPS and detail. Features include trajectory &amp; motion vector visualization, sequential IDs, and comprehensive JSON analytics export.

## Key Features
- High-Res ROI-Only Inference: Instead of scaling down the entire frame, this pipeline crops the Region of Interest from the original high-resolution frame first. YOLO then processes only this dense data, drastically improving small object detection at edge speeds.
- Optimized BoT-SORT Tracking: Custom BoT-SORT configuration tuned for CPU limits with custom anti-fragmentation logic to prevent ID switching and ghosting.
- Motion Vectors & Trajectories: Visualize object paths with customizable history trails and stabilized motion direction vectors (calculated with a 7-frame buffer to eliminate micro-jitters).
- Comprehensive Analytics: Exports detailed tracks_summary.json containing object lifecycle, time spent in ROI, average confidence, and precise trajectory coordinates.
- Hardware Telemetry: Real-time on-screen display of CPU usage, RAM footprint, temperature, and latency.
- Versatile Inputs: Process local video files, USB webcams (--cam 0), or network MJPEG streams (--stream).

## Installation

Designed to run on Raspberry Pi 5 (tested on Pi 4) running Linux, but compatible with any standard PC/Mac/Linux machine.

### 1. Repository & Environment Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/edge-ai-tracker.git
cd edge-ai-tracker

# Create and activate a virtual environment
python -m venv tracker_env
source tracker_env/bin/activate

# Install runtime dependencies on Raspberry Pi
pip install ultralytics ncnn opencv-python psutil numpy
```

### 2. Model Quantization & Export (NCNN INT8)
To achieve real-time performance on a Raspberry Pi CPU, you must use an NCNN quantized model. It is recommended to perform this export on a powerful PC/Mac, and then transfer the exported model to your Raspberry Pi.

**On your PC/Mac:**
```bash
# Install ultralytics on your PC
pip install ultralytics

# Export a YOLO model to NCNN format with INT8 quantization
# Note: INT8 quantization requires a dataset for calibration (e.g., data=coco.yaml)
yolo export model=yolo26n.pt format=ncnn int8=True data=coco.yaml
```
This will generate a folder named `yolo26n_ncnn_model`.

**Transfer the model to Raspberry Pi:**
```bash
# Copy the exported model directory to your Raspberry Pi using SCP
scp -r ./yolo26n_ncnn_model pi_username@<RPI_IP>:/path/to/edge-ai-tracker/
```

## Usage

Run the tracker using the command line.

**1. Process a video file with an interactive ROI selector:**
```bash
python tracker_ncnn_botsort_evo.py -v traffic.mp4 --roi-input
```

**2. Run on a USB Webcam with a 60-second limit, tracking trails, and motion vectors:**
```bash
python tracker_ncnn_botsort_evo.py --cam 0 --duration 60 --draw-vector --draw-trail 30
```

**3. Process an IP Camera / MJPEG Stream (& filtering only cars and trucks):**
```bash
# COCO classes: 2=car, 7=truck
python tracker_ncnn_botsort_evo.py -s http://10.0.0.1:8080/video -c 2 7
```
## Exhaustive Command Line Arguments

Below is the complete list of all supported flags and parameters.

### Input Sources

| Flag | Type | Default | Description |
|---|---|---|---|
| `-v`, `--video` | String | `""` | Path to the input video file. If not specified, the system defaults to `./test_video1.mp4` or camera. |
| `--cam` | Integer | `-1` | USB Camera index (e.g., `--cam 0` for the default webcam). |
| `-s`, `--stream` | String | `""` | HTTP/MJPEG network stream URL (e.g., `http://10.0.0.1:8080/video`). |

### Video Processing Controls

| Flag | Type | Default | Description |
|---|---|---|---|
| `--start-frame` | Integer | `0` | Frame number to start processing at. Useful for skipping video intros (Video files only). |
| `--end-frame` | Integer | `0` | Frame number to stop processing at. The script will automatically terminate and save outputs (Video files only). |
| `--duration` | Integer | `0` | Maximum recording/processing duration in seconds. Useful for automated camera capture (0 = infinite). |
| `--skip` | Integer | `2` | Frame skip multiplier for performance. 2 means process every 2nd frame (half FPS). 1 means process all frames. |

### Model & Neural Network Settings

| Flag | Type | Default | Description |
|---|---|---|---|
| `-m`, `--model` | String | `./yolo26n_ncnn_model` | Path to the directory containing the exported NCNN model. |
| `--imgsz` | Integer | `640` | Internal YOLO inference size. The longest side of the ROI crop will be scaled to this size. |
| `--conf` | Float | `0.20` | Minimum confidence threshold for object detection. Detections below this are ignored. |
| `--iou` | Float | `0.50` | Intersection Over Union (IoU) threshold for Non-Maximum Suppression (NMS). Controls overlapping box removal. |
| `-c`, `--classes` | Int List | `None` | Filter specific COCO class IDs (e.g., `-c 2 3 7` for cars, motorcycles, trucks). |
| `--classes-all` | Flag | `False` | Forces the tracker to detect and track all classes the model was trained on, overriding `-c`. |

### Region of Interest (ROI)

| Flag | Type | Default | Description |
|---|---|---|---|
| `--roi-input` | Flag | `False` | Pauses execution at launch and prompts the user to interactively type 4 ROI coordinates in the console. |
| `--roi` | String x4 | `None` | Hardcodes the 4 ROI polygon points in relative X,Y coordinates (0.0 to 1.0). Example: `--roi 0.5,0.2 0.8,0.2 0.9,0.9 0.1,0.9`. |

### Visualization & Outputs

| Flag | Type | Default | Description |
|---|---|---|---|
| `-o`, `--output` | String | `./output_botsort_evo.mp4` | File path for the resulting annotated video. |
| `--json-out` | String | `./tracks_summary.json` | File path to save the detailed JSON analytics report. |
| `--min-hits` | Integer | `2` | Number of consecutive frames an object must be detected before it is assigned an ID and drawn (Ghost/Flicker filter). |
| `--draw-trail` | Integer | `0` | Enables trajectory tails. The integer specifies the length of the tail in frames (e.g., 30). 0 disables it. |
| `--draw-vector` | Flag | `False` | Draws a red motion direction arrow originating from the object's center, stabilized over a 7-frame buffer. |

## Output Example

```json
[
  {
    "track_id": 1,
    "class_name": "car",
    "start_frame": 15,
    "end_frame": 120,
    "duration_seconds": 4.2,
    "average_confidence": 0.87,
    "entered_roi": true,
    "trajectory": [
      {"frame": 15, "x": 340, "y": 210},
      {"frame": 16, "x": 345, "y": 218}
    ]
  }
]
```

