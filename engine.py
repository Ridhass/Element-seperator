"""
engine.py
Core logic for the Element Separator app.

Loads an image once, then lets you click points on it to get a precise
mask for whatever object/element is under that point (using Meta's
Segment Anything Model). Each accepted element is exported as its own
PNG at the ORIGINAL image's exact size and pixel values — nothing is
recolored, cropped, or altered. Only the alpha channel changes: opaque
where that element is, transparent everywhere else. Stack all exported
PNGs (plus the leftover background PNG) at position 0,0 in After Effects
and you get back the exact original image, just split into layers.
"""

import os
import sys
import numpy as np
from PIL import Image


def get_checkpoint_path() -> str:
    """Looks for the SAM checkpoint file next to the app (bundled by the
    build), falling back to a local dev path."""
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(base, "sam_vit_b_01ec64.pth")
    if os.path.isfile(candidate):
        return candidate
    return candidate  # still return it; caller reports a clear error if missing


class SegmentEngine:
    """
    Wraps a SAM predictor plus bookkeeping for which pixels of the
    currently-loaded image have already been claimed by an accepted
    element, so new selections don't re-use them and the background
    layer is exactly "whatever's left".
    """

    def __init__(self, checkpoint_path: str = None, device: str = "cpu"):
        self.checkpoint_path = checkpoint_path or get_checkpoint_path()
        self.device = device
        self._predictor = None  # lazy-loaded (slow import + weight load)
        self.image_rgb = None   # HxWx3 uint8, original pixels, untouched
        self.remaining_mask = None  # HxW bool, True = not yet claimed
        self.height = 0
        self.width = 0

    # ------------------------------------------------------------------
    def load_model(self, progress_cb=None):
        if self._predictor is not None:
            return
        if progress_cb:
            progress_cb("Loading segmentation model (first time may take a moment)...")
        from segment_anything import sam_model_registry, SamPredictor

        if not os.path.isfile(self.checkpoint_path):
            raise FileNotFoundError(
                f"Could not find the model file at:\n{self.checkpoint_path}\n"
                "It should sit in the same folder as the app."
            )

        sam = sam_model_registry["vit_b"](checkpoint=self.checkpoint_path)
        sam.to(device=self.device)
        self._predictor = SamPredictor(sam)

    # ------------------------------------------------------------------
    def load_image(self, path: str, progress_cb=None):
        img = Image.open(path).convert("RGB")
        self.image_rgb = np.array(img)  # untouched original pixel data
        self.height, self.width = self.image_rgb.shape[:2]
        self.remaining_mask = np.ones((self.height, self.width), dtype=bool)

        if progress_cb:
            progress_cb("Analyzing image (one-time per image)...")
        self.load_model(progress_cb)
        self._predictor.set_image(self.image_rgb)

    # ------------------------------------------------------------------
    def predict_mask(self, points, labels):
        """
        points: list of (x, y) in ORIGINAL image pixel coordinates.
        labels: list of 1 (include) / 0 (exclude), same length as points.
        Returns the best mask (HxW bool), restricted to pixels not
        already claimed by a previously accepted element.
        """
        if self._predictor is None or self.image_rgb is None:
            raise RuntimeError("No image loaded yet.")

        coords = np.array(points, dtype=np.float32)
        lbls = np.array(labels, dtype=np.int32)

        masks, scores, _ = self._predictor.predict(
            point_coords=coords,
            point_labels=lbls,
            multimask_output=True,
        )
        best = masks[int(np.argmax(scores))]
        return best & self.remaining_mask

    # ------------------------------------------------------------------
    def _export_masked_png(self, mask: np.ndarray, out_path: str):
        h, w = mask.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., :3] = self.image_rgb
        rgba[..., 3] = np.where(mask, 255, 0).astype(np.uint8)
        Image.fromarray(rgba, mode="RGBA").save(out_path)

    def accept_element(self, mask: np.ndarray, output_dir: str, index: int) -> str:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"element_{index:02d}.png")
        self._export_masked_png(mask, out_path)
        self.remaining_mask = self.remaining_mask & (~mask)
        return out_path

    def export_background(self, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "background.png")
        self._export_masked_png(self.remaining_mask, out_path)
        return out_path

    def remaining_pixel_count(self) -> int:
        return int(self.remaining_mask.sum()) if self.remaining_mask is not None else 0
