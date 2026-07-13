"""
gui_app.py
Click-to-select desktop app: load a scene image, click on each element you
want isolated, accept it, repeat, then export the leftover background.
Every exported PNG is the same size as your original image with the
original pixels untouched — just different alpha masks — so dropping
them all into After Effects at position 0,0 reconstructs the scene
exactly, as separate layers.

Run with:  python gui_app.py
"""

import os
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk
import numpy as np

from engine import SegmentEngine

MAX_CANVAS_W = 900
MAX_CANVAS_H = 650

POS_COLOR = "#00e0a0"
NEG_COLOR = "#ff4d4d"
MASK_TINT = (0, 224, 160)  # overlay tint for the live mask preview


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Element Separator")
        self.geometry("980x780")
        self.resizable(False, False)

        self.engine = SegmentEngine()
        self.image_path = None
        self.output_dir = tk.StringVar()

        self.scale = 1.0
        self.points = []   # list of (x, y) in ORIGINAL image coords
        self.labels = []   # 1 = include, 0 = exclude
        self.current_mask = None
        self.element_count = 0

        self.tk_img = None
        self.base_pil_img = None  # scaled preview PIL image (no overlay)

        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self):
        top = ttk.Frame(self)
        top.pack(fill="x", padx=10, pady=8)
        ttk.Button(top, text="1. Open image...", command=self.open_image).pack(side="left")
        ttk.Button(top, text="2. Choose output folder...", command=self.choose_output).pack(side="left", padx=8)
        self.lbl_output = ttk.Label(top, text="(no output folder chosen)")
        self.lbl_output.pack(side="left", padx=8)

        self.canvas = tk.Canvas(self, width=MAX_CANVAS_W, height=MAX_CANVAS_H, bg="#222")
        self.canvas.pack(padx=10, pady=6)
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<Button-3>", self.on_right_click)

        hint = ttk.Label(
            self,
            text="Left-click: mark part of the element you want.  "
                 "Right-click: mark a spot to EXCLUDE from the current selection.",
        )
        hint.pack(pady=(0, 4))

        row = ttk.Frame(self)
        row.pack(pady=4)
        ttk.Button(row, text="Clear points", command=self.clear_points).pack(side="left", padx=4)
        ttk.Button(row, text="Accept element \u2192 export PNG", command=self.accept_element).pack(side="left", padx=4)
        ttk.Button(row, text="Undo last element", command=self.undo_last).pack(side="left", padx=4)
        ttk.Button(row, text="Export background & finish", command=self.export_background).pack(side="left", padx=4)

        self.status = tk.Text(self, height=8)
        self.status.pack(fill="both", expand=True, padx=10, pady=8)

    # ------------------------------------------------------------------
    def log(self, msg):
        self.status.insert("end", msg + "\n")
        self.status.see("end")
        self.update_idletasks()

    # ------------------------------------------------------------------
    def open_image(self):
        path = filedialog.askopenfilename(
            title="Choose a scene image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp"), ("All files", "*.*")],
        )
        if not path:
            return
        self.image_path = path
        self.element_count = 0
        self.points, self.labels, self.current_mask = [], [], None
        self.log(f"Loading {os.path.basename(path)} ...")

        def work():
            try:
                self.engine.load_image(path, progress_cb=lambda m: self.after(0, self.log, m))
                self.after(0, self._on_image_loaded)
            except Exception as e:
                err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
                self.after(0, self.log, "ERROR:\n" + err)

        threading.Thread(target=work, daemon=True).start()

    def _on_image_loaded(self):
        h, w = self.engine.height, self.engine.width
        self.scale = min(MAX_CANVAS_W / w, MAX_CANVAS_H / h, 1.0)
        disp_w, disp_h = int(w * self.scale), int(h * self.scale)
        self.base_pil_img = Image.fromarray(self.engine.image_rgb).resize((disp_w, disp_h))
        self.canvas.config(width=disp_w, height=disp_h)
        self._redraw()
        self.log("Ready. Click on an element to select it.")

    def choose_output(self):
        d = filedialog.askdirectory(title="Choose output folder for the PNG layers")
        if d:
            self.output_dir.set(d)
            self.lbl_output.config(text=d)

    # ------------------------------------------------------------------
    def _redraw(self, mask_overlay=None):
        if self.base_pil_img is None:
            return
        img = self.base_pil_img.copy()

        if mask_overlay is not None:
            small_mask = Image.fromarray((mask_overlay * 255).astype(np.uint8)).resize(img.size)
            overlay = Image.new("RGB", img.size, MASK_TINT)
            mask_arr = np.array(small_mask) > 127
            base_arr = np.array(img)
            tinted = base_arr.copy()
            tinted[mask_arr] = (base_arr[mask_arr] * 0.4 + np.array(MASK_TINT) * 0.6).astype(np.uint8)
            img = Image.fromarray(tinted)

        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

        for (x, y), lbl in zip(self.points, self.labels):
            cx, cy = x * self.scale, y * self.scale
            color = POS_COLOR if lbl == 1 else NEG_COLOR
            r = 5
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, fill=color, outline="white")

    # ------------------------------------------------------------------
    def _predict_and_show(self):
        if not self.points:
            self._redraw()
            self.current_mask = None
            return
        try:
            self.current_mask = self.engine.predict_mask(self.points, self.labels)
            self._redraw(mask_overlay=self.current_mask)
        except Exception as e:
            err = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            self.log("ERROR:\n" + err)

    def on_left_click(self, event):
        if self.engine.image_rgb is None:
            return
        x, y = event.x / self.scale, event.y / self.scale
        self.points.append((x, y))
        self.labels.append(1)
        self._predict_and_show()

    def on_right_click(self, event):
        if self.engine.image_rgb is None or not self.points:
            return
        x, y = event.x / self.scale, event.y / self.scale
        self.points.append((x, y))
        self.labels.append(0)
        self._predict_and_show()

    def clear_points(self):
        self.points, self.labels, self.current_mask = [], [], None
        self._redraw()

    # ------------------------------------------------------------------
    def accept_element(self):
        if self.current_mask is None or not self.current_mask.any():
            messagebox.showinfo("Nothing selected", "Click on an element first.")
            return
        if not self.output_dir.get():
            messagebox.showwarning("No output folder", "Choose an output folder first.")
            return
        self.element_count += 1
        path = self.engine.accept_element(self.current_mask, self.output_dir.get(), self.element_count)
        self.log(f"Exported {os.path.basename(path)}")
        self.points, self.labels, self.current_mask = [], [], None
        self._redraw()

    def undo_last(self):
        if self.element_count == 0:
            return
        path = os.path.join(self.output_dir.get(), f"element_{self.element_count:02d}.png")
        messagebox.showinfo(
            "Manual undo needed",
            f"Delete this file yourself to fully undo: {path}\n\n"
            "(The 'claimed pixels' bookkeeping for the current session can't be "
            "un-claimed automatically — for a clean undo, reload the image.)",
        )

    def export_background(self):
        if not self.output_dir.get():
            messagebox.showwarning("No output folder", "Choose an output folder first.")
            return
        path = self.engine.export_background(self.output_dir.get())
        self.log(f"Exported {os.path.basename(path)} (whatever pixels were left over)")
        messagebox.showinfo(
            "Done",
            f"Exported {self.element_count} element(s) plus the background layer to:\n"
            f"{self.output_dir.get()}\n\n"
            "Drag all of them into After Effects at position (0,0) and stack in "
            "the same order to reconstruct your original image as separate layers.",
        )


if __name__ == "__main__":
    app = App()
    app.mainloop()
