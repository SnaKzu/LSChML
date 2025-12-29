from __future__ import annotations

import math
from typing import Iterable, Optional

import numpy as np

# MediaPipe Hands landmark indices
WRIST = 0
INDEX_MCP = 5
MIDDLE_MCP = 9


def _to_points_xyz(hand_landmarks) -> np.ndarray:
    """Return (21, 3) float32 array from a MediaPipe hand_landmarks object."""
    return np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark], dtype=np.float32)


def normalize_hand_points(
    points_xyz: np.ndarray,
    *,
    rotate_palm: bool = True,
    eps: float = 1e-6,
) -> np.ndarray:
    """Normalize a single hand (21,3) landmarks.

    Steps:
    - Translate so WRIST is at origin.
    - Scale by palm size (distance wrist->middle_mcp; fallback wrist->index_mcp).
    - Optionally rotate in XY so wrist->index_mcp aligns with +X.

    Notes:
    - Operates on MediaPipe normalized coordinates (x,y in [0,1], z is relative).
    - Rotation is done in XY plane (around Z axis); Z is kept as-is (but scaled).
    """
    if points_xyz.shape != (21, 3):
        raise ValueError(f"Expected points shape (21,3), got {points_xyz.shape}")

    points = points_xyz.astype(np.float32, copy=True)

    # Translate to wrist origin
    wrist = points[WRIST].copy()
    points -= wrist

    # Scale (use palm size)
    scale = float(np.linalg.norm(points[MIDDLE_MCP, :2]))
    if scale < eps:
        scale = float(np.linalg.norm(points[INDEX_MCP, :2]))
    if scale >= eps:
        points /= scale

    # Rotate in XY plane to reduce pose variation
    if rotate_palm:
        v = points[INDEX_MCP, :2]
        v_norm = float(np.linalg.norm(v))
        if v_norm >= eps:
            theta = math.atan2(float(v[1]), float(v[0]))
            c = math.cos(-theta)
            s = math.sin(-theta)
            rot = np.array([[c, -s], [s, c]], dtype=np.float32)
            points[:, :2] = points[:, :2] @ rot.T

    return points


def frame_from_mediapipe_results(
    results,
    *,
    max_hands: int = 2,
    rotate_palm: bool = True,
) -> list[float]:
    """Build a 126-length frame vector from MediaPipe results, normalized.

    Output layout: [hand0(21*3), hand1(21*3)]
    Missing hands are left as zeros.
    """
    frame = np.zeros((max_hands, 21, 3), dtype=np.float32)

    hand_landmarks_list = getattr(results, "multi_hand_landmarks", None)
    if not hand_landmarks_list:
        return frame.reshape(-1).astype(np.float32).tolist()

    handedness_list = getattr(results, "multi_handedness", None)

    # Prefer stable ordering by handedness: LEFT -> slot 0, RIGHT -> slot 1
    if handedness_list and len(handedness_list) == len(hand_landmarks_list) and max_hands >= 2:
        filled = set()
        for hand_landmarks, handedness in zip(hand_landmarks_list, handedness_list):
            label = None
            try:
                label = handedness.classification[0].label
            except Exception:
                label = None

            if label == "Left":
                slot = 0
            elif label == "Right":
                slot = 1
            else:
                # unknown: use first free slot
                slot = 0 if 0 not in filled else 1

            if slot in filled:
                continue
            pts = _to_points_xyz(hand_landmarks)
            pts = normalize_hand_points(pts, rotate_palm=rotate_palm)
            frame[slot] = pts
            filled.add(slot)
            if len(filled) >= max_hands:
                break
    else:
        # Fallback: keep MediaPipe order
        for hand_idx, hand_landmarks in enumerate(hand_landmarks_list[:max_hands]):
            pts = _to_points_xyz(hand_landmarks)
            pts = normalize_hand_points(pts, rotate_palm=rotate_palm)
            frame[hand_idx] = pts

    return frame.reshape(-1).astype(np.float32).tolist()


def normalize_frame_vector(
    frame_126: np.ndarray,
    *,
    rotate_palm: bool = True,
    eps: float = 1e-6,
) -> np.ndarray:
    """Normalize a single frame stored as 126 values.

    Useful for training-time normalization when loading existing .npy files.
    If a hand is all zeros, it is left as zeros.
    """
    frame = np.asarray(frame_126, dtype=np.float32).reshape(2, 21, 3)

    for hand_idx in range(2):
        hand = frame[hand_idx]
        if float(np.linalg.norm(hand)) < eps:
            continue
        frame[hand_idx] = normalize_hand_points(hand, rotate_palm=rotate_palm, eps=eps)

    return frame.reshape(126)


def maybe_normalize_sequence(
    sequence: np.ndarray,
    *,
    rotate_palm: bool = True,
    eps: float = 1e-6,
) -> np.ndarray:
    """Normalize a sequence (T,126) only if it looks un-normalized.

    Heuristic: if the median |wrist_xy| over non-empty hands is large (>~0.05),
    we assume raw MediaPipe coords and normalize.
    """
    seq = np.asarray(sequence, dtype=np.float32)
    if seq.ndim != 2 or seq.shape[1] != 126:
        raise ValueError(f"Expected sequence shape (T,126), got {seq.shape}")

    reshaped = seq.reshape(seq.shape[0], 2, 21, 3)

    wrist_xy = reshaped[:, :, WRIST, :2]  # (T,2,2)
    hand_norm = np.linalg.norm(reshaped.reshape(seq.shape[0], 2, -1), axis=-1)  # (T,2)
    mask = hand_norm > eps

    if not np.any(mask):
        return seq

    wrist_abs = np.abs(wrist_xy)[mask]
    median_abs = float(np.median(wrist_abs))

    if median_abs < 1e-3:
        # already centered at origin (likely normalized)
        return seq

    out = np.empty_like(seq)
    for i in range(seq.shape[0]):
        out[i] = normalize_frame_vector(seq[i], rotate_palm=rotate_palm, eps=eps)
    return out
