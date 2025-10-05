# R-Farm – Cycles render farm prototype

This repository contains a proof-of-concept implementation of a Blender add-on and a GPU-enabled Google Cloud Run service that together form a minimal render farm for Cycles. The first iteration focuses on rendering a single frame from Blender on demand.

## Project structure

- `blender_addon/` – Blender add-on that exposes a panel inside the **Render Properties** tab. The button submits the current frame together with the existing Cycles settings to the remote worker.
- `cloud_run/` – Source code and container definition for the Cloud Run GPU worker executing Blender in headless mode.

## Blender add-on

1. Zip the folder:
   ```bash
   cd blender_addon
   zip -r ../rfarm_blender_addon.zip .
   ```
2. Inside Blender open **Edit → Preferences → Add-ons** and install the generated archive.
3. Configure the add-on in **Edit → Preferences → Add-ons → R-Farm Cycles Render**:
   - **Cloud Run endpoint** – URL of the deployed worker (for example `https://render-worker-xxxxx.a.run.app`).
   - **Auth token** – Optional bearer token when the worker is protected (for example by Cloud Endpoints or IAP).
   - The add-on uses the `requests` Python library, which ships with recent Blender releases. If you are running a custom Python build, install it via `pip install requests --target "<blender>/scripts/modules"`.
4. Switch the render engine to **Cycles** and configure all render parameters as usual (resolution, samples, output path, etc.).
5. Open the **Render Properties** tab and use the **Render Current Frame on R-Farm** button. The resulting frame is written to the configured output path.

The add-on serialises the current `.blend` file to a temporary location, sends it to the worker, waits for completion, and stores the returned image on disk. Any errors from the worker are surfaced inside Blender.

## Cloud Run GPU worker

The worker is a FastAPI application that receives a `.blend` file (base64-encoded) together with render metadata. It starts Blender in background mode, enforces GPU rendering, and returns the rendered image as a base64 payload.

### Building the container image

```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/rfarm-render cloud_run
```

The Docker image installs Blender 4.5.0 on top of an NVIDIA CUDA 12 runtime image and uses `uvicorn` to expose the FastAPI service.

### Deploying to Cloud Run (GPU)

```bash
gcloud run deploy rfarm-render \
  --image gcr.io/PROJECT_ID/rfarm-render \
  --platform managed \
  --region REGION \
  --gpu=1 \
  --no-gpu-zonal-redundancy \
  --memory 16Gi \
  --cpu 4 \
  --timeout 15m \
  --max-instances 1 \
  --min-instances 0 \
  --allow-unauthenticated
```

Cloud Run currently provisions NVIDIA L4 GPUs for managed GPU services. Adjust the region depending on availability. The worker only runs while a render job is executing, allowing you to pay only for active render time. Consider restricting access (e.g. with a service account and Cloud Endpoints) for production deployments. The worker prefers the Cycles **OPTIX** compute backend (best suited for L4) and automatically falls back to CUDA when OPTIX is unavailable.

### API contract

`POST /render`

```json
{
  "frame": 1,
  "blend_file": "base64-encoded data",
  "render_settings": {
    "samples": 256,
    "resolution_x": 1920,
    "resolution_y": 1080,
    "resolution_percentage": 100,
    "file_format": "PNG",
    "color_mode": "RGB",
    "color_depth": "8"
  },
  "device": "GPU",
  "compute_device_type": "OPTIX"
}
```

Successful responses return the rendered image encoded as base64 together with a job identifier.

### Running locally

The service can be tested without deploying to Cloud Run:

```bash
cd cloud_run
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8080
```

Then point the Blender add-on endpoint to `http://127.0.0.1:8080` (requires Blender to have access to the same Docker daemon or Blender installation to execute renders locally). When deployed to Cloud Run the worker logs include the detected GPU via `nvidia-smi`, which should report `NVIDIA L4`.

## Security and scaling considerations

- Protect the `/render` endpoint with authentication to avoid exposing your render farm to the public Internet.
- Use Google Cloud Storage for large outputs or animations in future iterations to avoid transferring large payloads back to Blender synchronously.
- To scale to multiple frames or animations, extend the API to enqueue tasks (for example via Pub/Sub or Cloud Tasks) and stream results asynchronously.

This repository provides the minimal end-to-end scaffolding required to experiment with GPU rendering on demand using Blender and Google Cloud Run.
