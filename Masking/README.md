# Segmentation & Inpainting API — MobileSAM + LaMa

**Version:** 0.1.0

This repository provides a FastAPI-based backend that uses **MobileSAM** for segmentation (point-selection) and **LaMa** for image inpainting. It accepts an uploaded image, performs segmentation for provided point coordinates, crops objects, inpaints removed areas, and serves downloadable artifacts.

---

## Features

* Upload images via `/upload`
* Start segmentation + inpainting with `/segment` (point-selection)
* Check job status via `/status/{job_id}`
* Download artifacts with `/download/{job_id}/{filename}`
* Cleanup job files with `/cleanup/{job_id}`
* Health check at `/health`

---

## Quick Start (Development)

> These instructions assume you're using a Debian/Ubuntu-like environment (Codespaces, Ubuntu, Debian). If you use Docker, see the Docker section below.

### 1. Clone the repo

```bash
git clone <your-repo-url>
cd Relighting_Backend/Masking
```

### 2. Python environment

Use Python 3.10 or 3.11 if possible. Python 3.12 can cause build issues for some packages.

Create a virtual environment and activate it:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. System dependencies (required for Pillow, OpenCV, headless GL, etc.)

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  libjpeg-dev zlib1g-dev libpng-dev libtiff5-dev libfreetype6-dev \
  libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1
```

> If you are in an environment without `sudo` (some containers/dev containers), run these in a privileged container or add to your Dockerfile.

### 4. Install Python dependencies

A `requirements.txt` is provided. Install with:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> If you need GPU-accelerated PyTorch, install torch/torchvision/torchaudio wheels from the official PyTorch index for your CUDA version instead of the generic `torch` line in `requirements.txt`.

### 5. Place MobileSAM checkpoint

Download the MobileSAM checkpoint (`mobile_sam.pt`) and place it in the project root (same folder as `runserver.py` / `main.py`).

Recommended source: [https://github.com/ChaoningZhang/MobileSAM](https://github.com/ChaoningZhang/MobileSAM)

### 6. Run the server

```bash
python runserver.py
```

Or run via uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Open the Swagger UI at `http://localhost:8000/docs` (or the forwarded URL in Codespaces).

---


## API Endpoints

All endpoints are under the root server (e.g., `http://localhost:8000`). Examples below use `BASE_URL` as placeholder.

### `POST /upload`

Upload an image for processing. Returns a `job_id` you will use for later calls.

**Request**

* Content-Type: `multipart/form-data`
* Form field: `file` (binary file)

**Response** (200)

```json
{
  "job_id": "<job_id>",
  "status": "queued",
  "message": "Image uploaded. Send segmentation request to start processing."
}
```

**cURL Example**

```bash
curl -X POST "${BASE_URL}/upload" -F "file=@path/to/image.png"
```

---

### `POST /segment?job_id=<job_id>`

Start segmentation + inpainting for the image uploaded in `/upload`.

**Query parameter**

* `job_id` (string) — the job id returned by `/upload`

**Request body**

* JSON array of point coordinates: `[[x1, y1], [x2, y2], ...]`

**Response** (200)

```json
{ "job_id": "<job_id>", "status": "processing" }
```

**cURL Example**

```bash
curl -X POST "${BASE_URL}/segment?job_id=<job_id>" \
  -H "Content-Type: application/json" \
  -d '[ [200, 350], [400, 120] ]'
```

---

### `GET /status/{job_id}`

Get job processing status and progress.

**Response** (200)

```json
{
  "status": "ready",            # queued | processing | ready | error
  "progress": 100,
  "original_size": [width, height],
  "job_dir": "/path/to/jobs/<job_id>",
  "num_objects": 1,
  "successful_segmentations": 1
}
```

---

### `GET /download/{job_id}/{filename}`

Download processed artifacts from the job directory.

**Common filenames**

* `original.png`
* `mask_0.png`, `mask_1.png`, ...
* `cropped_0.png`, `cropped_1.png`, ...
* `inpainted_0.png`, `inpainted_1.png`, ...
* `full_inpainted.png`

**cURL Example**

```bash
curl -o out.png "${BASE_URL}/download/<job_id>/inpainted_0.png"
```

---

### `DELETE /cleanup/{job_id}`

Delete job folder and artifacts.

**Response**

```json
{ "status": "deleted", "job_id": "<job_id>" }
```

---

### `GET /health`

Returns health info about the server and whether models are loaded.

**Response**

```json
{ "status": "healthy", "device": "cpu|cuda", "models_loaded": true }
```

---

## Typical filenames produced

The pipeline generally writes files under `jobs/<job_id>/`. Typical names include:

```
original.png
mask_0.png
cropped_0.png
inpainted_0.png
full_inpainted.png
overlay_0.png
```

If you want the API to list files, consider adding a small endpoint to return `os.listdir(job_dir)`.

---

## Troubleshooting / Common Errors

### `ImportError: libGL.so.1: cannot open shared object file`

Install system packages:

```bash
sudo apt-get update
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender1
```

Or use `opencv-python-headless` instead of GUI-enabled OpenCV if you don't need GUI features:

```bash
pip uninstall -y opencv-python
pip install opencv-python-headless
```

### `ModuleNotFoundError: No module named 'timm'`

Install `timm`:

```bash
pip install timm
```

### `ModuleNotFoundError: No module named 'simple_lama_inpainting'`

Install LaMa wrapper:

```bash
pip install simple-lama-inpainting
```

If Pillow build fails during install (missing `jpeg` headers), install system deps before pip:

```bash
sudo apt-get install -y libjpeg-dev zlib1g-dev libpng-dev libtiff5-dev libfreetype6-dev
pip install --force-reinstall pillow
```

### Pillow / Python 3.12 build issues

If you see build errors on Python 3.12 consider switching to Python 3.10 or 3.11 for compatibility with some wheels.

### CUDA / PyTorch

If you want GPU support, install PyTorch with the wheel matching your CUDA version from the official PyTorch instructions. Do **not** rely on `pip install torch` in GPU environments if you need a specific CUDA build.

---

## Developer tips & improvements

* Add a `/files/{job_id}` endpoint to list available artifacts.
* Add authentication if you plan to expose the API publicly.
* Expand segmentation input to accept polygons or bounding boxes for more precise masks.
* Add queueing/backpressure for high traffic (e.g., Redis/RQ, Celery, or FastAPI BackgroundTasks with worker pool).

---

## License

Add your project license here.

---

## Contact
