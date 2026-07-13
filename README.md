# Element Separator

Load a scene image (illustration or photo), click on each element you want
isolated, and get it exported as its own PNG — same canvas size as the
original, original pixels untouched, just a different alpha mask. Drop every
exported PNG into After Effects at position (0,0), stack them in the same
order they were made, and you get back the exact original image, now split
into layers you can animate independently.

Uses Meta's Segment Anything Model (SAM) under the hood so a single click
can precisely grab whatever object is at that point — works reasonably on
both flat/vector-style graphics and real photos.

## How to use it

1. **Open image** — pick your scene file.
2. **Choose output folder** — where the PNG layers will be saved.
3. **Left-click** on part of an element to select it. A colored preview
   overlay shows what's currently selected.
   - Left-click again on another part of the *same* element (e.g. if the
     first click only grabbed part of it) to expand the selection.
   - **Right-click** to mark a point that should be *excluded* — useful if
     the selection is bleeding into a neighboring object.
4. **Accept element → export PNG** once the preview overlay matches what you
   want. This saves `element_01.png`, `element_02.png`, etc. and "claims"
   those pixels so later clicks won't reuse them.
5. Repeat for every element.
6. **Export background & finish** — saves whatever pixels are left over
   (everything you didn't click) as `background.png`.
7. In After Effects: import all the PNGs, drop them onto the timeline,
   make sure each one sits at position (0,0) at 100% scale (they already
   are, since every export is full-canvas size) — done.

## Get a ready-made .exe with nothing to install (recommended)

This repo builds itself into a real Windows app on GitHub's free cloud
Windows machines — you don't need Python, PyTorch, or the SAM model
installed locally.

1. Create a new **public** GitHub repo.
2. Upload `engine.py`, `gui_app.py`, `requirements.txt`, and this
   `README.md` directly into the repo root (select the files themselves
   when uploading, not a parent folder).
3. Go to the **Actions** tab → click **"set up a workflow yourself"** →
   rename the file to `build-exe.yml` (keep it inside `.github/workflows/`)
   → paste in the workflow YAML (see `.github/workflows/build-exe.yml` in
   this project) → commit.
4. Wait for the run to go green — this one takes longer than a typical
   build (~10-15 min) because it downloads PyTorch and a ~375MB model file.
5. Download the **ElementSeparator-Windows** artifact from the finished
   run. Unzip it — you'll get `ElementSeparator.exe` and
   `sam_vit_b_01ec64.pth`. **Keep them in the same folder.**
6. Double-click `ElementSeparator.exe` to run it.

## Things worth knowing

- **First image load is the slow part.** Loading the model and analyzing a
  new image can take anywhere from a few seconds to ~30+ seconds on a
  normal CPU (there's no GPU requirement, but no GPU means no acceleration
  either). Every click *after* that on the same image is fast — the slow
  part only happens once per image.
- **The app and model file are large** (several hundred MB total) because
  they bundle PyTorch and the segmentation model so nothing needs to be
  installed separately. That's the tradeoff for "no install, no internet
  needed to run it."
- If a click grabs too much or too little, right-click to exclude an area
  or left-click again to add to the selection, then re-accept.
- **Undo** for an accepted element currently just tells you which file to
  delete manually — full pixel "un-claiming" isn't implemented, so for a
  clean redo of a whole image, just reload it.

## Files

- `gui_app.py` — the desktop window and click handling.
- `engine.py` — the SAM wrapper, mask math, and PNG export logic. Can be
  imported and used from your own scripts too.
- `requirements.txt` — Python dependencies (CPU-only PyTorch).
- `.github/workflows/build-exe.yml` — builds the Windows exe automatically.
