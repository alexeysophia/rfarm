"""FastAPI application powering the R-Farm Cloud Run render worker."""

from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import tempfile
import uuid
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.error import URLError
from urllib.request import Request as UrlRequest, urlopen

from fastapi import FastAPI, HTTPException
from google.api_core import exceptions as gcs_exceptions
from google.cloud import storage
from pydantic import BaseModel, Field

import google.auth
from google.auth import exceptions as google_auth_exceptions

APP_ROOT = Path(__file__).resolve().parent
BLENDER_EXECUTABLE = os.environ.get("BLENDER_EXECUTABLE", "blender")
RENDER_SCRIPT = APP_ROOT / "render_worker.py"
GCS_BUCKET = os.environ.get("RFARM_GCS_BUCKET")
SIGNED_URL_TTL_MINUTES = int(os.environ.get("RFARM_SIGNED_URL_TTL_MINUTES", "15"))

_storage_client: Optional[storage.Client] = None
_service_account_email: Optional[str] = None

app = FastAPI(title="R-Farm Render Worker", version="0.1.0")


class RenderSettings(BaseModel):
    resolution_x: Optional[int] = None
    resolution_y: Optional[int] = None
    resolution_percentage: Optional[int] = Field(default=None, ge=1, le=100)
    samples: Optional[int] = Field(default=None, ge=1)
    preview_samples: Optional[int] = Field(default=None, ge=1)
    use_adaptive_sampling: Optional[bool] = None
    file_format: Optional[str] = None
    color_mode: Optional[str] = None
    color_depth: Optional[str] = None
    use_file_extension: Optional[bool] = True
    filepath: Optional[str] = None


class RenderRequest(BaseModel):
    frame: int = Field(description="Frame number to render")
    blend_file: Optional[str] = Field(
        default=None, description="Base64 encoded .blend content (legacy mode)",
    )
    blend_gcs_uri: Optional[str] = Field(
        default=None,
        description="gs:// URI pointing at the .blend file uploaded to Cloud Storage",
    )
    render_settings: RenderSettings = Field(default_factory=RenderSettings)
    device: str = Field(default="GPU", description="Either GPU or CPU")
    compute_device_type: str = Field(default="OPTIX", description="Cycles compute device")


class RenderResponse(BaseModel):
    job_id: str
    image_base64: str
    output_format: str
    output_path: str
    logs: Optional[str] = None


class BlendUploadRequest(BaseModel):
    filename: str = Field(
        default="scene.blend",
        description="Name of the .blend file used to compose the Cloud Storage object",
    )


class BlendUploadResponse(BaseModel):
    upload_url: str
    blob_name: str
    gcs_uri: str
    expires_in: int = Field(description="Validity of the signed URL in seconds")


@app.get("/healthz")
def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


def _get_storage_client() -> storage.Client:
    global _storage_client
    if _storage_client is None:
        _storage_client = storage.Client()
    return _storage_client


def _get_service_account_email() -> Optional[str]:
    global _service_account_email
    if _service_account_email:
        return _service_account_email

    env_email = os.environ.get("RFARM_SERVICE_ACCOUNT_EMAIL") or os.environ.get(
        "GOOGLE_SERVICE_ACCOUNT_EMAIL"
    )
    if env_email:
        _service_account_email = env_email
        return _service_account_email

    try:
        credentials, _ = google.auth.default()
    except google_auth_exceptions.DefaultCredentialsError:
        credentials = None

    if credentials:
        email = getattr(credentials, "service_account_email", None)
        if email:
            _service_account_email = email
            return _service_account_email

    try:
        metadata_request = UrlRequest(
            "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email",
            headers={"Metadata-Flavor": "Google"},
        )
        with urlopen(metadata_request, timeout=2) as response:
            email = response.read().decode("utf-8").strip()
    except (URLError, OSError, TimeoutError):
        email = None

    _service_account_email = email
    return _service_account_email


def _write_blend_file(tmp_dir: Path, blend_data: str) -> Path:
    blend_path = tmp_dir / "scene.blend"
    with open(blend_path, "wb") as handle:
        handle.write(base64.b64decode(blend_data))
    return blend_path


def _parse_gcs_uri(gcs_uri: str) -> Tuple[str, str]:
    if not gcs_uri.startswith("gs://"):
        raise HTTPException(status_code=400, detail="Invalid GCS URI")
    path = gcs_uri[5:]
    try:
        bucket_name, blob_name = path.split("/", 1)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid GCS URI") from exc
    if not bucket_name or not blob_name:
        raise HTTPException(status_code=400, detail="Invalid GCS URI")
    return bucket_name, blob_name


def _download_blend_from_gcs(tmp_dir: Path, gcs_uri: str) -> Path:
    bucket_name, blob_name = _parse_gcs_uri(gcs_uri)
    try:
        client = _get_storage_client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        destination = tmp_dir / "scene.blend"
        blob.download_to_filename(destination)
    except gcs_exceptions.GoogleAPIError as exc:
        raise HTTPException(status_code=502, detail=f"Failed to download blend file: {exc}") from exc
    if not destination.exists():
        raise HTTPException(status_code=500, detail="Blend download failed")
    return destination


def _run_blender(blend_path: Path, request: RenderRequest) -> Dict[str, Any]:
    output_dir = blend_path.parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_basename = "frame"
    script_args = [
        "--frame",
        str(request.frame),
        "--output-dir",
        str(output_dir),
        "--output-basename",
        output_basename,
        "--device",
        request.device,
        "--compute-device",
        request.compute_device_type,
    ]

    settings = request.render_settings
    if settings.samples is not None:
        script_args.extend(["--samples", str(settings.samples)])
    if settings.resolution_x is not None:
        script_args.extend(["--resolution-x", str(settings.resolution_x)])
    if settings.resolution_y is not None:
        script_args.extend(["--resolution-y", str(settings.resolution_y)])
    if settings.resolution_percentage is not None:
        script_args.extend(["--resolution-percentage", str(settings.resolution_percentage)])
    if settings.file_format is not None:
        script_args.extend(["--file-format", settings.file_format])
    if settings.color_mode is not None:
        script_args.extend(["--color-mode", settings.color_mode])
    if settings.color_depth is not None:
        script_args.extend(["--color-depth", settings.color_depth])
    if settings.use_adaptive_sampling is not None:
        script_args.extend(["--use-adaptive-sampling", str(int(settings.use_adaptive_sampling))])

    command = [
        BLENDER_EXECUTABLE,
        "--background",
        str(blend_path),
        "--python",
        str(RENDER_SCRIPT),
        "--",
    ] + script_args

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    logs = result.stdout
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Blender failed: {logs[-2000:]}")

    output_path = None
    for line in logs.splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "output_path" in payload:
            output_path = Path(payload["output_path"])
            break

    if not output_path or not output_path.exists():
        raise HTTPException(status_code=500, detail="Render worker did not produce output")

    encoded = base64.b64encode(output_path.read_bytes()).decode("ascii")
    return {
        "output_path": str(output_path),
        "image_base64": encoded,
        "logs": logs,
    }


@app.post("/render", response_model=RenderResponse)
async def render_endpoint(request: RenderRequest) -> RenderResponse:
    job_id = str(uuid.uuid4())
    tmp_dir = Path(tempfile.mkdtemp(prefix="rfarm_"))
    blend_path: Optional[Path] = None
    try:
        if request.blend_gcs_uri:
            blend_path = _download_blend_from_gcs(tmp_dir, request.blend_gcs_uri)
        elif request.blend_file:
            blend_path = _write_blend_file(tmp_dir, request.blend_file)
        else:
            raise HTTPException(status_code=400, detail="Missing blend file reference")
        result = _run_blender(blend_path, request)
        output_path = Path(result["output_path"])
        output_format = output_path.suffix.lstrip(".").upper() or request.render_settings.file_format or "PNG"
        return RenderResponse(
            job_id=job_id,
            image_base64=result["image_base64"],
            output_format=output_format,
            output_path=str(output_path),
            logs=result["logs"],
        )
    finally:
        if blend_path and blend_path.exists():
            try:
                blend_path.unlink()
            except OSError:
                pass
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.post("/upload-url", response_model=BlendUploadResponse)
async def create_upload_url(request: BlendUploadRequest) -> BlendUploadResponse:
    if not GCS_BUCKET:
        raise HTTPException(status_code=500, detail="RFARM_GCS_BUCKET is not configured")

    object_name = f"uploads/{uuid.uuid4()}/{request.filename}"
    expiration = timedelta(minutes=SIGNED_URL_TTL_MINUTES)

    try:
        client = _get_storage_client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(object_name)
        service_account_email = _get_service_account_email()
        signing_kwargs = {}
        if service_account_email:
            signing_kwargs["service_account_email"] = service_account_email
        upload_url = blob.generate_signed_url(
            version="v4",
            expiration=expiration,
            method="PUT",
            content_type="application/octet-stream",
            **signing_kwargs,
        )
    except gcs_exceptions.GoogleAPIError as exc:
        raise HTTPException(status_code=502, detail=f"Failed to generate upload URL: {exc}") from exc

    gcs_uri = f"gs://{GCS_BUCKET}/{object_name}"
    return BlendUploadResponse(
        upload_url=upload_url,
        blob_name=object_name,
        gcs_uri=gcs_uri,
        expires_in=int(expiration.total_seconds()),
    )


@app.post("/upload", response_model=BlendUploadResponse, include_in_schema=False)
async def create_upload_url_legacy(request: BlendUploadRequest) -> BlendUploadResponse:
    """Compatibility shim for older Blender add-on versions."""

    return await create_upload_url(request)
