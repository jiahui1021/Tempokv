"""Robust global affine flow and a residual local-motion proxy."""
from __future__ import annotations
from typing import Any


def camera_proxy(flow: Any, config: dict[str, Any]) -> dict[str, Any]:
    import numpy as np
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("camera-motion estimation requires opencv-python") from exc
    height, width = flow.shape[:2]
    step = int(config.get("grid_step", 16))
    if step <= 0:
        raise ValueError("camera.grid_step must be positive")
    yy, xx = np.mgrid[0:height:step, 0:width:step]
    source = np.stack([xx, yy], axis=-1).reshape(-1, 2).astype(np.float32)
    sampled = flow[yy, xx].reshape(-1, 2)
    valid = np.isfinite(sampled).all(axis=1)
    source, sampled = source[valid], sampled[valid]
    if len(source) < int(config.get("minimum_points", 12)):
        return {"global_motion": None, "residual_motion": None, "inlier_ratio": None, "camera_valid": False, "camera_failure": "insufficient_points", "camera_points": len(source)}
    cv2.setRNGSeed(int(config.get("seed", 0)))
    matrix, inliers = cv2.estimateAffine2D(source, source + sampled, method=cv2.RANSAC,
                                         ransacReprojThreshold=float(config.get("ransac_threshold", 3.0)),
                                         maxIters=int(config.get("max_iterations", 2000)), confidence=0.99,
                                         refineIters=10)
    ratio = float(inliers.mean()) if inliers is not None else None
    if matrix is None or ratio is None or ratio < float(config.get("min_inlier_ratio", 0.3)):
        return {"global_motion": None, "residual_motion": None, "inlier_ratio": ratio, "camera_valid": False,
                "camera_failure": "affine_failed_or_low_inlier_ratio", "camera_points": len(source)}
    yy, xx = np.mgrid[0:height, 0:width]
    coordinates = np.stack([xx, yy], axis=-1).astype(np.float32)
    global_flow = coordinates @ matrix[:, :2].T + matrix[:, 2] - coordinates
    finite = np.isfinite(flow).all(axis=-1)
    return {"global_motion": float(np.linalg.norm(global_flow[finite], axis=-1).mean()),
            "residual_motion": float(np.linalg.norm((flow - global_flow)[finite], axis=-1).mean()),
            "inlier_ratio": ratio, "camera_valid": True, "camera_failure": None,
            "camera_points": len(source), "affine_matrix": matrix.tolist()}


def forward_backward_proxy(forward: Any, backward: Any, *, absolute: float = 0.5, relative: float = 0.01) -> dict[str, Any]:
    """Flow inconsistency is an occlusion candidate signal, not ground truth."""
    import numpy as np
    import cv2
    height, width = forward.shape[:2]
    yy, xx = np.mgrid[0:height, 0:width]
    dest_x, dest_y = (xx + forward[..., 0]).astype(np.float32), (yy + forward[..., 1]).astype(np.float32)
    inside = (dest_x >= 0) & (dest_x < width - 1) & (dest_y >= 0) & (dest_y < height - 1)
    warped = cv2.remap(backward, dest_x, dest_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    valid = inside & np.isfinite(forward).all(axis=-1) & np.isfinite(warped).all(axis=-1)
    if not valid.any():
        return {"fb_inconsistent_fraction": None, "fb_valid_fraction": 0.0}
    magnitude = np.linalg.norm(forward, axis=-1) + np.linalg.norm(warped, axis=-1)
    error = np.linalg.norm(forward + warped, axis=-1)
    return {"fb_inconsistent_fraction": float((error[valid] > absolute + relative * magnitude[valid]).mean()),
            "fb_valid_fraction": float(valid.mean())}
