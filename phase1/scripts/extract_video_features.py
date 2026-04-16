#!/usr/bin/env python3
"""
extract_video_features.py — Steps 1 + 2a: Frame Extraction + AU Feature Extraction

For each segment in outputs/segments/ESXXXX_segments.csv:
1. Looks up the speaker's camera from outputs/speaker_camera_map.csv
2. Opens the correct Closeup video
3. Extracts frames at 1fps within [start, end], saves as temp JPEGs
4. Runs py-feat Detector.detect_image() to extract Action Units and head pose
5. Mean-pools frame-level features per segment

If no face is detected for a segment, fills all features with 0.0 and logs warning.

Usage:
    python scripts/extract_video_features.py --meeting ES2002a

Output:
    outputs/video_features/ESXXXX_video_features.csv
    Columns: meeting_id, segment_id, AU01, AU02, AU04, AU06, AU12, AU15, AU17, AU25,
             head_pitch, head_yaw, head_roll, gaze_x, gaze_y
"""

import argparse
import os
import sys
import tempfile
import warnings

import cv2
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Fixed seed
np.random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEGMENTS_DIR = os.path.join(BASE_DIR, "outputs", "segments")
VIDEO_BASE = os.path.join(BASE_DIR, "Datasets", "amicorpus")
CAMERA_MAP_PATH = os.path.join(BASE_DIR, "outputs", "speaker_camera_map.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "video_features")

# AU columns we extract (matching TASK_video.txt spec)
AU_COLUMNS = ["AU01", "AU02", "AU04", "AU06", "AU12", "AU15", "AU17", "AU25"]

POSE_COLUMNS = ["head_pitch", "head_yaw", "head_roll"]
GAZE_COLUMNS = ["gaze_x", "gaze_y"]
ALL_FEATURE_COLUMNS = AU_COLUMNS + POSE_COLUMNS + GAZE_COLUMNS

# py-feat column name mapping
PYFEAT_POSE_MAP = {"head_pitch": "Pitch", "head_yaw": "Yaw", "head_roll": "Roll"}


def init_detector():
    """Initialize py-feat Detector."""
    from feat import Detector
    detector = Detector(
        face_model="retinaface",
        landmark_model="mobilefacenet",
        au_model="xgb",
        emotion_model="resmasknet",
    )
    return detector


def extract_and_save_frames(video_path, start_time, end_time, tmp_dir, fps_sample=1.0):
    """Extract frames from video at fps_sample rate, save as temp JPEGs.

    Returns list of file paths.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 25.0

    paths = []
    t = start_time
    frame_idx = 0
    while t < end_time:
        frame_num = int(t * video_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            path = os.path.join(tmp_dir, f"frame_{frame_idx:04d}.jpg")
            cv2.imwrite(path, frame)
            paths.append(path)
            frame_idx += 1
        t += 1.0 / fps_sample

    cap.release()
    return paths


def extract_features_from_files(detector, frame_paths):
    """Run py-feat on saved frame files, return mean-pooled features.

    Returns dict with feature columns, or None if no faces detected.
    """
    if not frame_paths:
        return None

    try:
        # Process all frames at once via detect_image (accepts list of paths)
        result = detector.detect_image(frame_paths)
        if result is None or len(result) == 0:
            return None
    except Exception:
        return None

    # Extract features, taking first face per frame
    all_features = []
    # Group by frame if multiple faces detected
    if "frame" in result.columns:
        frames = result.groupby("frame")
    else:
        # Each row is a face detection; group by input image
        frames = [(i, result.iloc[[i]]) for i in range(len(result))]

    for _, group in frames:
        row_data = group.iloc[0]  # Take first (largest) face
        feat = {}

        # AUs
        for au in AU_COLUMNS:
            if au in result.columns:
                val = float(row_data[au])
                feat[au] = val if not np.isnan(val) else 0.0
            else:
                feat[au] = 0.0

        # Pose
        for our_col, pf_col in PYFEAT_POSE_MAP.items():
            if pf_col in result.columns:
                val = float(row_data[pf_col])
                feat[our_col] = val if not np.isnan(val) else 0.0
            else:
                feat[our_col] = 0.0

        # Gaze (py-feat may not output gaze depending on config)
        gaze_cols = [c for c in result.columns if "gaze" in c.lower()]
        if len(gaze_cols) >= 2:
            feat["gaze_x"] = float(row_data[gaze_cols[0]])
            feat["gaze_y"] = float(row_data[gaze_cols[1]])
        else:
            feat["gaze_x"] = 0.0
            feat["gaze_y"] = 0.0

        all_features.append(feat)

    if not all_features:
        return None

    # Mean-pool across frames
    mean_features = {}
    for col in ALL_FEATURE_COLUMNS:
        vals = [f.get(col, 0.0) for f in all_features]
        mean_features[col] = float(np.mean(vals))

    return mean_features


def process_meeting(meeting_id, detector, camera_map_df):
    """Process all segments for a single meeting."""
    segments_path = os.path.join(SEGMENTS_DIR, f"{meeting_id}_segments.csv")
    if not os.path.exists(segments_path):
        print(f"ERROR: {segments_path} not found", file=sys.stderr)
        sys.exit(1)

    segments_df = pd.read_csv(segments_path)
    meeting_cameras = camera_map_df[camera_map_df["meeting_id"] == meeting_id]

    if len(meeting_cameras) == 0:
        print(f"ERROR: No camera mapping for {meeting_id}", file=sys.stderr)
        sys.exit(1)

    speaker_to_camera = dict(
        zip(meeting_cameras["speaker_id"], meeting_cameras["camera_id"])
    )

    video_dir = os.path.join(VIDEO_BASE, meeting_id, "video")
    warnings_log = []
    rows = []

    total = len(segments_df)
    print(f"Processing {meeting_id}: {total} segments")

    with tempfile.TemporaryDirectory() as tmp_dir:
        for idx, seg in segments_df.iterrows():
            segment_id = int(seg["segment_id"])
            speaker_id = seg["speaker_id"]
            start = float(seg["start"])
            end = float(seg["end"])

            camera_id = speaker_to_camera.get(speaker_id, 1)
            video_path = os.path.join(video_dir, f"{meeting_id}.Closeup{camera_id}.avi")

            if not os.path.exists(video_path):
                warnings_log.append(f"segment_id={segment_id}: video not found")
                row = {"meeting_id": meeting_id, "segment_id": segment_id}
                for col in ALL_FEATURE_COLUMNS:
                    row[col] = 0.0
                rows.append(row)
                continue

            # Extract frames and save as temp files
            frame_paths = extract_and_save_frames(video_path, start, end, tmp_dir)

            if not frame_paths:
                warnings_log.append(f"segment_id={segment_id}: no frames (duration={end-start:.2f}s)")
                row = {"meeting_id": meeting_id, "segment_id": segment_id}
                for col in ALL_FEATURE_COLUMNS:
                    row[col] = 0.0
                rows.append(row)
                continue

            # Run py-feat
            features = extract_features_from_files(detector, frame_paths)

            # Clean up temp files
            for fp in frame_paths:
                try:
                    os.unlink(fp)
                except OSError:
                    pass

            if features is None:
                warnings_log.append(f"segment_id={segment_id}: no face detected")
                row = {"meeting_id": meeting_id, "segment_id": segment_id}
                for col in ALL_FEATURE_COLUMNS:
                    row[col] = 0.0
                rows.append(row)
                continue

            row = {"meeting_id": meeting_id, "segment_id": segment_id}
            row.update(features)
            rows.append(row)

            if (idx + 1) % 50 == 0:
                print(f"  {idx + 1}/{total} segments processed")

    # Build output DataFrame
    result_df = pd.DataFrame(rows)
    result_df["segment_id"] = result_df["segment_id"].astype(int)

    # Ensure column order
    col_order = ["meeting_id", "segment_id"] + ALL_FEATURE_COLUMNS
    result_df = result_df[col_order]

    # Write warnings log
    if warnings_log:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        warn_path = os.path.join(OUTPUT_DIR, "warnings.log")
        with open(warn_path, "a") as f:
            f.write(f"\n--- {meeting_id} ---\n")
            for w in warnings_log:
                f.write(f"  {w}\n")
        print(f"  {len(warnings_log)} warnings logged to {warn_path}")

    return result_df


def main():
    parser = argparse.ArgumentParser(description="Extract video features per segment")
    parser.add_argument("--meeting", type=str, required=True, help="Meeting ID (e.g. ES2002a)")
    args = parser.parse_args()

    if not os.path.exists(CAMERA_MAP_PATH):
        print(f"ERROR: {CAMERA_MAP_PATH} not found. Run map_speaker_camera.py first.", file=sys.stderr)
        sys.exit(1)

    camera_map_df = pd.read_csv(CAMERA_MAP_PATH)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{args.meeting}_video_features.csv")
    if os.path.exists(output_path):
        print(f"WARNING: {output_path} already exists. Overwriting.")

    print("Initializing py-feat Detector...")
    detector = init_detector()
    print("Detector ready.")

    result_df = process_meeting(args.meeting, detector, camera_map_df)

    # Validate: no NaN
    nan_count = result_df.isna().sum().sum()
    if nan_count > 0:
        print(f"ERROR: {nan_count} NaN values found in output", file=sys.stderr)
        sys.exit(1)

    # Validate: no duplicate segment_ids
    dupes = result_df.duplicated(subset=["meeting_id", "segment_id"], keep=False)
    if dupes.any():
        print(f"ERROR: Duplicate (meeting_id, segment_id) pairs found", file=sys.stderr)
        sys.exit(1)

    result_df.to_csv(output_path, index=False)
    print(f"WROTE {output_path} {len(result_df)} rows")


if __name__ == "__main__":
    main()
