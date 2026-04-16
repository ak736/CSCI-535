#!/usr/bin/env python3
"""
map_speaker_camera.py — Step 0.5: Speaker-Camera Mapping

Determines which Closeup{1-4}.avi corresponds to which SPEAKER_XX
for each meeting by correlating diarization timestamps with lip
movement detection (mouth-region pixel variance).

For each speaker, finds a segment where they are the sole active speaker,
then measures mouth-region pixel variance in each Closeup video during
that timestamp window. The camera with the highest variance is assigned
to that speaker.

Uses OpenCV Haar cascade for face detection and lower-face ROI for
mouth region variance estimation.

Usage:
    # Single meeting:
    python scripts/map_speaker_camera.py --meeting ES2002a

    # All meetings:
    python scripts/map_speaker_camera.py --all

Output:
    outputs/speaker_camera_map.csv
    Columns: meeting_id, speaker_id, camera_id
"""

import argparse
import os
import sys
import csv

import cv2
import numpy as np
import pandas as pd

# Fixed seed
np.random.seed(42)

MEETINGS = [
    "ES2002a", "ES2002b", "ES2002c", "ES2002d",
    "ES2003a", "ES2003b", "ES2003c", "ES2003d",
    "ES2004a", "ES2004b", "ES2004c", "ES2004d",
]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEGMENTS_DIR = os.path.join(BASE_DIR, "outputs", "segments")
VIDEO_BASE = os.path.join(BASE_DIR, "Datasets", "amicorpus")
OUTPUT_PATH = os.path.join(BASE_DIR, "outputs", "speaker_camera_map.csv")


def get_sole_speaker_segments(segments_df, speaker_id, min_duration=3.0):
    """Find segments where this speaker is the sole active speaker.

    Returns segments sorted by duration descending, filtered to those
    where no other speaker has an overlapping segment.
    """
    speaker_segs = segments_df[segments_df["speaker_id"] == speaker_id].copy()
    speaker_segs["duration"] = speaker_segs["end"] - speaker_segs["start"]
    speaker_segs = speaker_segs[speaker_segs["duration"] >= min_duration]
    speaker_segs = speaker_segs.sort_values("duration", ascending=False)

    sole_segments = []
    for _, seg in speaker_segs.iterrows():
        # Check if any other speaker has a segment overlapping this one
        others = segments_df[segments_df["speaker_id"] != speaker_id]
        overlaps = others[
            (others["start"] < seg["end"]) & (others["end"] > seg["start"])
        ]
        if len(overlaps) == 0:
            sole_segments.append(seg)
        if len(sole_segments) >= 3:
            break

    # If no sole segments found, relax and just use longest segments
    if not sole_segments:
        sole_segments = [row for _, row in speaker_segs.head(3).iterrows()]

    return sole_segments


def measure_mouth_variance(video_path, start_time, end_time):
    """Measure pixel variance in the mouth region during a time window.

    Opens the video, seeks to start_time, extracts frames until end_time,
    detects faces, crops lower-face (mouth) region, and computes mean
    pixel variance across frames.

    Returns mean variance (higher = more lip movement = likely the speaker).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0  # AMI default

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    variances = []
    prev_mouth = None
    frame_idx = start_frame

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # Sample every 3rd frame to speed up
        if (frame_idx - start_frame) % 3 != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))

        if len(faces) == 0:
            continue

        # Use the largest face
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

        # Mouth region: lower 40% of face bounding box
        mouth_y = y + int(h * 0.6)
        mouth_h = int(h * 0.4)
        mouth_region = gray[mouth_y : mouth_y + mouth_h, x : x + w]

        if mouth_region.size == 0:
            continue

        # Compute temporal difference if we have a previous frame
        if prev_mouth is not None and prev_mouth.shape == mouth_region.shape:
            diff = cv2.absdiff(mouth_region, prev_mouth)
            variances.append(float(np.var(diff)))

        prev_mouth = mouth_region.copy()

    cap.release()

    if not variances:
        return 0.0
    return float(np.mean(variances))


def map_speakers_for_meeting(meeting_id):
    """Determine speaker-camera mapping for a single meeting.

    Returns list of dicts: [{"meeting_id", "speaker_id", "camera_id"}, ...]
    """
    segments_path = os.path.join(SEGMENTS_DIR, f"{meeting_id}_segments.csv")
    if not os.path.exists(segments_path):
        print(f"ERROR: {segments_path} not found", file=sys.stderr)
        return []

    segments_df = pd.read_csv(segments_path)
    speakers = sorted(segments_df["speaker_id"].unique())
    cameras = [1, 2, 3, 4]

    video_dir = os.path.join(VIDEO_BASE, meeting_id, "video")
    if not os.path.isdir(video_dir):
        print(f"ERROR: {video_dir} not found", file=sys.stderr)
        return []

    # Verify all closeup videos exist
    for cam in cameras:
        vpath = os.path.join(video_dir, f"{meeting_id}.Closeup{cam}.avi")
        if not os.path.exists(vpath):
            print(f"ERROR: {vpath} not found", file=sys.stderr)
            return []

    print(f"\n{'='*60}")
    print(f"Mapping speakers for {meeting_id}")
    print(f"Speakers: {speakers}")
    print(f"{'='*60}")

    # Build variance matrix: speakers x cameras
    variance_matrix = np.zeros((len(speakers), len(cameras)))

    for si, speaker in enumerate(speakers):
        sole_segs = get_sole_speaker_segments(segments_df, speaker)
        if not sole_segs:
            print(f"  WARNING: No usable segments for {speaker} in {meeting_id}")
            continue

        print(f"  {speaker}: testing {len(sole_segs)} segment(s)")

        for seg in sole_segs:
            for ci, cam in enumerate(cameras):
                vpath = os.path.join(video_dir, f"{meeting_id}.Closeup{cam}.avi")
                var = measure_mouth_variance(vpath, seg["start"], seg["end"])
                variance_matrix[si, ci] += var

        # Average across segments
        variance_matrix[si] /= len(sole_segs)
        print(f"    Variance scores: {dict(zip(cameras, variance_matrix[si]))}")

    # Assign cameras using greedy best-match (Hungarian-like)
    # Each speaker gets the camera with highest variance, no duplicates
    assignments = {}
    used_cameras = set()
    used_speakers = set()

    # Sort by max variance (most confident assignment first)
    speaker_order = []
    for si in range(len(speakers)):
        max_var = np.max(variance_matrix[si])
        speaker_order.append((si, max_var))
    speaker_order.sort(key=lambda x: -x[1])

    for si, _ in speaker_order:
        best_cam_idx = None
        best_var = -1
        for ci in range(len(cameras)):
            if ci not in used_cameras and variance_matrix[si, ci] > best_var:
                best_var = variance_matrix[si, ci]
                best_cam_idx = ci
        if best_cam_idx is not None:
            assignments[speakers[si]] = cameras[best_cam_idx]
            used_cameras.add(best_cam_idx)
            used_speakers.add(si)

    # Handle any unassigned speakers (shouldn't happen with 4 speakers + 4 cameras)
    for si, speaker in enumerate(speakers):
        if speaker not in assignments:
            remaining = [ci for ci in range(len(cameras)) if ci not in used_cameras]
            if remaining:
                assignments[speaker] = cameras[remaining[0]]
                used_cameras.add(remaining[0])
                print(f"  WARNING: {speaker} assigned by fallback to Closeup{cameras[remaining[0]]}")

    print(f"\n  Final mapping for {meeting_id}:")
    results = []
    for speaker in speakers:
        cam = assignments.get(speaker, 1)
        print(f"    {speaker} -> Closeup{cam}")
        results.append({
            "meeting_id": meeting_id,
            "speaker_id": speaker,
            "camera_id": cam,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Map speakers to closeup cameras")
    parser.add_argument("--meeting", type=str, help="Single meeting ID (e.g. ES2002a)")
    parser.add_argument("--all", action="store_true", help="Process all 12 meetings")
    args = parser.parse_args()

    if not args.meeting and not args.all:
        parser.error("Specify --meeting ESXXXX or --all")

    meetings = MEETINGS if args.all else [args.meeting]

    # If output file exists and we're doing a single meeting, preserve other mappings
    existing = []
    if os.path.exists(OUTPUT_PATH) and args.meeting:
        existing_df = pd.read_csv(OUTPUT_PATH)
        existing = existing_df[existing_df["meeting_id"] != args.meeting].to_dict("records")

    all_results = list(existing)

    for mid in meetings:
        results = map_speakers_for_meeting(mid)
        if not results:
            print(f"FAILED: {mid} — skipping", file=sys.stderr)
            continue
        all_results.extend(results)

    # Write output
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df = pd.DataFrame(all_results)
    df = df.sort_values(["meeting_id", "speaker_id"]).reset_index(drop=True)

    # Validate: no duplicate (meeting_id, speaker_id) pairs
    dupes = df.duplicated(subset=["meeting_id", "speaker_id"], keep=False)
    if dupes.any():
        print(f"ERROR: Duplicate mappings found:\n{df[dupes]}", file=sys.stderr)
        sys.exit(1)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nWROTE {OUTPUT_PATH} {len(df)} rows")


if __name__ == "__main__":
    main()
