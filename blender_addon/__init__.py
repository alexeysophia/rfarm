"""Blender add-on for triggering R-Farm remote renders."""

bl_info = {
    "name": "R-Farm Cycles Render",
    "author": "R-Farm",
    "version": (0, 1, 0),
    "blender": (3, 4, 0),
    "location": "Properties > Render",
    "description": "Submit the active frame to an R-Farm Cloud Run render worker",
    "warning": "",
    "category": "Render",
}

import base64
import json
import queue
import shutil
import tempfile
import threading
import time
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Optional, Tuple

import bpy
from bpy.props import (
    BoolProperty,
    CollectionProperty,
    FloatProperty,
    IntProperty,
    PointerProperty,
    StringProperty,
)
from bpy.types import AddonPreferences, Operator, Panel, PropertyGroup, UIList
from bpy.utils import register_class, unregister_class

try:
    import requests
except ImportError:
    requests = None


class RFarmAddonPreferences(AddonPreferences):
    bl_idname = __name__

    endpoint: StringProperty(
        name="Cloud Run endpoint",
        description="Base URL of the R-Farm Cloud Run render worker",
        default="",
    )
    auth_token: StringProperty(
        name="Auth token",
        description=(
            "Optional bearer token passed to the Cloud Run service. "
            "Leave empty if authentication is disabled."
        ),
        default="",
        subtype="PASSWORD",
    )

    def draw(self, _context):
        layout = self.layout
        layout.prop(self, "endpoint")
        layout.prop(self, "auth_token")


class RFarmLogEntry(PropertyGroup):
    timestamp: FloatProperty(
        name="Timestamp",
        description="Unix timestamp of the log entry",
        default=0.0,
    )
    message: StringProperty(
        name="Message",
        description="Log message emitted during the remote render",
        default="",
    )


class RFarmStatus(PropertyGroup):
    last_job_id: StringProperty(
        name="Last Job ID",
        description="Identifier of the last submitted remote job",
        default="",
    )
    last_output_path: StringProperty(
        name="Last Output Path",
        description="Absolute path to the file produced by the remote render",
        default="",
        subtype="FILE_PATH",
    )
    is_rendering: BoolProperty(
        name="Is Rendering",
        description="True while Blender waits for a response from the remote worker",
        default=False,
    )
    last_error: StringProperty(
        name="Last Error",
        description="Error message returned during the previous remote render",
        default="",
    )
    log_entries: CollectionProperty(
        name="Log Entries",
        type=RFarmLogEntry,
        description="Log messages produced during the active remote render",
    )
    log_index: IntProperty(
        name="Active Log Index",
        description="Helper index used by the log UI list",
        default=0,
    )
    start_timestamp: FloatProperty(
        name="Start Timestamp",
        description="Unix timestamp for when the current remote render began",
        default=0.0,
    )
    finish_timestamp: FloatProperty(
        name="Finish Timestamp",
        description="Unix timestamp for when the remote render completed",
        default=0.0,
    )


class RFarm_UL_status_log(UIList):
    bl_idname = "RFarm_UL_status_log"

    def draw_item(self, _context, layout, _data, item, _icon, _active_data, _active_propname, _index):
        if self.layout_type in {"DEFAULT", "COMPACT"}:
            timestamp = datetime.fromtimestamp(item.timestamp) if item.timestamp else None
            if timestamp:
                time_label = timestamp.strftime("%H:%M:%S")
            else:
                time_label = "--:--:--"
            row = layout.row()
            row.label(text=time_label, icon="TIME")
            row.label(text=item.message)
        elif self.layout_type == "GRID":
            layout.alignment = "CENTER"
            layout.label(text="")


class RFarm_PT_panel(Panel):
    bl_label = "R-Farm Remote Render"
    bl_space_type = "PROPERTIES"
    bl_region_type = "WINDOW"
    bl_context = "render"

    @classmethod
    def poll(cls, context):
        return context.scene.render.engine == "CYCLES"

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        status = scene.rfarm_status

        layout.label(text="Submit the active frame to the configured R-Farm backend.")
        if status.is_rendering:
            layout.label(text="Status: waiting for response", icon="TIME")
        elif status.last_error:
            layout.label(text=f"Error: {status.last_error}", icon="ERROR")
        elif status.last_job_id:
            layout.label(text=f"Last job: {status.last_job_id}", icon="INFO")
            if status.last_output_path:
                layout.label(text=f"Saved to: {status.last_output_path}", icon="FILE")

        layout.separator()
        row = layout.row()
        row.enabled = not status.is_rendering
        row.operator(RFarm_OT_render_frame.bl_idname, icon="RENDER_STILL")


def _ensure_output_directory(path: Path) -> None:
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)


def _collect_render_metadata(scene) -> dict:
    render = scene.render
    cycles = scene.cycles
    metadata = {
        "resolution_x": render.resolution_x,
        "resolution_y": render.resolution_y,
        "resolution_percentage": render.resolution_percentage,
        "samples": getattr(cycles, "samples", None),
        "preview_samples": getattr(cycles, "preview_samples", None),
        "use_adaptive_sampling": getattr(cycles, "use_adaptive_sampling", None),
        "file_format": render.image_settings.file_format,
        "color_mode": render.image_settings.color_mode,
        "color_depth": render.image_settings.color_depth,
        "use_file_extension": render.use_file_extension,
        "filepath": render.filepath,
    }
    return metadata


def _prepare_base_payload(scene) -> dict:
    payload = {
        "frame": scene.frame_current,
        "render_settings": _collect_render_metadata(scene),
    }

    cycles = scene.cycles
    payload["device"] = getattr(cycles, "device", "CPU")

    compute_device = "CUDA"
    if getattr(cycles, "device", "CPU") == "GPU":
        prefs = bpy.context.preferences.addons.get("cycles")
        if prefs:
            compute_device = getattr(prefs.preferences, "compute_device_type", "CUDA")
    payload["compute_device_type"] = compute_device

    return payload


class UploadURLNotSupported(RuntimeError):
    """Raised when the remote worker does not expose the /upload-url endpoint."""


def _build_payload(
    base_payload: dict,
    blend_gcs_uri: Optional[str],
    blend_inline: Optional[str],
) -> dict:
    payload = dict(base_payload)

    if blend_gcs_uri:
        payload["blend_gcs_uri"] = blend_gcs_uri
    if blend_inline:
        payload["blend_file"] = blend_inline

    return payload


def _request_upload_url(endpoint: str, auth_token: str, blend_path: Path) -> Tuple[str, str]:
    url = endpoint.rstrip("/") + "/upload-url"
    headers = {"Content-Type": "application/json"}
    if auth_token:
        headers["Authorization"] = f"Bearer {auth_token}"

    request_payload = {"filename": blend_path.name}

    response = requests.post(url, headers=headers, json=request_payload, timeout=60)
    if response.status_code == 404:
        raise UploadURLNotSupported(
            "Remote worker does not expose /upload-url (legacy deployment)"
        )
    if response.status_code >= 400:
        raise RuntimeError(f"Failed to request upload URL: HTTP {response.status_code} {response.text}")

    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError("Upload URL response was not valid JSON") from exc

    upload_url = payload.get("upload_url")
    gcs_uri = payload.get("gcs_uri")
    if not upload_url or not gcs_uri:
        raise RuntimeError("Upload URL response missing required fields")

    return upload_url, gcs_uri


def _upload_blend_to_gcs(upload_url: str, blend_path: Path) -> None:
    headers = {"Content-Type": "application/octet-stream", "Content-Length": str(blend_path.stat().st_size)}
    with open(blend_path, "rb") as handle:
        response = requests.put(upload_url, data=handle, headers=headers, timeout=300)
    if response.status_code not in (200, 201):
        raise RuntimeError(f"Upload to Cloud Storage failed: HTTP {response.status_code} {response.text}")


ACTIVE_RENDER_JOBS = {}


def _format_elapsed(elapsed_seconds: float) -> str:
    elapsed_seconds = max(0, int(elapsed_seconds))
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _append_log_entry(status: RFarmStatus, message: str, timestamp: Optional[float] = None) -> None:
    entry = status.log_entries.add()
    entry.timestamp = timestamp or time.time()
    entry.message = message
    status.log_index = max(0, len(status.log_entries) - 1)


def _run_remote_render_job(job_args: dict) -> None:
    job_queue: queue.Queue = job_args["queue"]
    endpoint: str = job_args["endpoint"]
    auth_token: str = job_args["auth_token"]
    blend_path = Path(job_args["blend_path"])
    tmpdir = Path(job_args["tmpdir"])
    base_payload = job_args["base_payload"]
    output_path_str: str = job_args["output_path"]
    fallback_dir = Path(job_args["fallback_dir"])
    requests_module = job_args["requests_module"]

    def log(message: str) -> None:
        job_queue.put(("log", time.time(), message))

    try:
        if requests_module is None:
            raise RuntimeError(
                "Python module 'requests' is required. Install it in Blender's Python environment."
            )

        log("Requesting upload URL from R-Farm service")
        gcs_uri: Optional[str] = None
        blend_inline: Optional[str] = None

        try:
            upload_url, gcs_uri = _request_upload_url(endpoint, auth_token, blend_path)
            log("Upload URL received, uploading .blend file")
            _upload_blend_to_gcs(upload_url, blend_path)
            log("Blend file uploaded successfully")
        except UploadURLNotSupported:
            log("Service does not support upload URLs, embedding .blend file in request")
            with open(blend_path, "rb") as handle:
                blend_inline = base64.b64encode(handle.read()).decode("ascii")

        payload = _build_payload(base_payload, gcs_uri, blend_inline)
        url = endpoint.rstrip("/") + "/render"

        headers = {"Content-Type": "application/json"}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        log("Submitting render request to R-Farm service")
        response = requests_module.post(url, headers=headers, data=json.dumps(payload), timeout=300)
        log(f"Service responded with HTTP {response.status_code}")

        if response.status_code >= 400:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text}")

        try:
            payload_response = response.json()
        except ValueError as exc:
            raise RuntimeError("Invalid JSON response") from exc

        job_id = payload_response.get("job_id", "")
        image_base64 = payload_response.get("image_base64")
        remote_format = payload_response.get("output_format", "PNG")

        if not image_base64:
            raise RuntimeError("Response missing image data")

        image_bytes = base64.b64decode(image_base64)
        final_path: Optional[Path] = None

        if output_path_str:
            final_path = Path(output_path_str)
            _ensure_output_directory(final_path)
            with open(final_path, "wb") as out_file:
                out_file.write(image_bytes)
        else:
            fallback_dir.mkdir(parents=True, exist_ok=True)
            fallback_name = f"rfarm_frame.{remote_format.lower()}"
            final_path = fallback_dir / fallback_name
            with open(final_path, "wb") as out_file:
                out_file.write(image_bytes)

        folder_text = str(final_path.parent)
        log(f"Result saved to {folder_text} ({final_path.name})")

        job_queue.put(
            (
                "result",
                time.time(),
                {
                    "status": "success",
                    "job_id": job_id,
                    "output_path": str(final_path),
                },
            )
        )
    except Exception as exc:  # noqa: BLE001
        log(f"Error: {exc}")
        job_queue.put(
            (
                "result",
                time.time(),
                {
                    "status": "error",
                    "error": str(exc),
                },
            )
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _process_remote_job_queue(scene) -> bool:
    status = scene.rfarm_status
    scene_key = scene.as_pointer()
    job_info = ACTIVE_RENDER_JOBS.get(scene_key)
    if not job_info:
        return False

    job_queue: queue.Queue = job_info["queue"]
    updated = False

    while True:
        try:
            kind, timestamp, payload = job_queue.get_nowait()
        except queue.Empty:
            break

        if kind == "log":
            _append_log_entry(status, payload, timestamp)
            updated = True
        elif kind == "result":
            status.finish_timestamp = timestamp
            if payload.get("status") == "success":
                status.last_error = ""
                status.last_job_id = payload.get("job_id", "")
                status.last_output_path = payload.get("output_path", "")
            else:
                status.last_job_id = ""
                status.last_output_path = ""
                status.last_error = payload.get("error", "Unknown error")
            status.is_rendering = False
            updated = True
            job_info["done"] = True

    if job_info.get("done") and job_queue.empty():
        ACTIVE_RENDER_JOBS.pop(scene_key, None)

    return updated


def _scene_from_pointer(pointer: int):
    for scene in bpy.data.scenes:
        if scene.as_pointer() == pointer:
            return scene
    return None


def _remote_job_timer_callback(scene_key: int):
    scene = _scene_from_pointer(scene_key)
    if scene is None:
        ACTIVE_RENDER_JOBS.pop(scene_key, None)
        return None

    _process_remote_job_queue(scene)
    job_info = ACTIVE_RENDER_JOBS.get(scene_key)

    if not job_info:
        return None

    if job_info.get("done") and job_info["queue"].empty():
        return None

    return 0.5


class RFarm_OT_render_frame(Operator):
    bl_idname = "rfarm.render_frame"
    bl_label = "Render Current Frame on R-Farm"
    bl_description = "Upload the current .blend and render the active frame on Cloud Run"

    def _resolve_output_path(self, scene) -> Optional[Path]:
        render = scene.render
        frame = scene.frame_current
        filepath = bpy.path.abspath(render.frame_path(frame=frame))
        if not filepath:
            return None
        return Path(filepath)

    def execute(self, context):
        scene = context.scene
        status = scene.rfarm_status
        prefs = context.preferences.addons[__name__].preferences

        if status.is_rendering:
            self.report({"WARNING"}, "A remote render is already running.")
            return {"CANCELLED"}

        if requests is None:
            message = "Python module 'requests' is required. Install it in Blender's Python environment."
            status.last_error = message
            self.report({"ERROR"}, message)
            return {"CANCELLED"}

        if not prefs.endpoint:
            self.report({"ERROR"}, "Configure the Cloud Run endpoint in the add-on preferences.")
            status.last_error = "Cloud Run endpoint not configured"
            return {"CANCELLED"}

        if scene.render.engine != "CYCLES":
            self.report({"ERROR"}, "Switch the render engine to Cycles before submitting.")
            status.last_error = "Render engine must be Cycles"
            return {"CANCELLED"}

        tmpdir = Path(tempfile.mkdtemp(prefix="rfarm_"))
        blend_path = tmpdir / "scene.blend"
        try:
            bpy.ops.wm.save_as_mainfile(filepath=str(blend_path), copy=True)
        except RuntimeError as ex:
            shutil.rmtree(tmpdir, ignore_errors=True)
            status.last_error = str(ex)
            self.report({"ERROR"}, f"Unable to export .blend: {ex}")
            return {"CANCELLED"}

        output_path = self._resolve_output_path(scene)
        fallback_dir = Path(bpy.app.tempdir or tempfile.gettempdir())
        base_payload = _prepare_base_payload(scene)

        status.is_rendering = True
        status.last_error = ""
        status.last_output_path = ""
        status.last_job_id = ""
        status.start_timestamp = time.time()
        status.finish_timestamp = 0.0
        status.log_entries.clear()
        status.log_index = 0
        _append_log_entry(status, "Remote render job initialised")

        job_queue: queue.Queue = queue.Queue()
        scene_key = scene.as_pointer()
        ACTIVE_RENDER_JOBS.pop(scene_key, None)
        ACTIVE_RENDER_JOBS[scene_key] = {"queue": job_queue, "done": False}

        job_args = {
            "queue": job_queue,
            "endpoint": prefs.endpoint,
            "auth_token": prefs.auth_token,
            "blend_path": str(blend_path),
            "tmpdir": str(tmpdir),
            "base_payload": base_payload,
            "output_path": str(output_path) if output_path else "",
            "fallback_dir": str(fallback_dir),
            "requests_module": requests,
        }

        worker = threading.Thread(target=_run_remote_render_job, args=(job_args,), daemon=True)
        job_info = ACTIVE_RENDER_JOBS[scene_key]
        job_info["thread"] = worker
        timer_callback = partial(_remote_job_timer_callback, scene_key)
        job_info["timer_callback"] = timer_callback
        bpy.app.timers.register(timer_callback, first_interval=0.5)
        worker.start()

        self.report({"INFO"}, "Remote render job submitted to R-Farm")
        try:
            bpy.ops.rfarm.render_status_popup("INVOKE_DEFAULT")
        except RuntimeError:
            # Popup may fail if invoked from non-main areas; ignore but continue.
            pass

        return {"FINISHED"}


class RFarm_OT_render_status_popup(Operator):
    bl_idname = "rfarm.render_status_popup"
    bl_label = "R-Farm Render Status"

    _timer = None

    def invoke(self, context, _event):
        scene = context.scene
        if scene is None or not hasattr(scene, "rfarm_status"):
            return {"CANCELLED"}

        wm = context.window_manager
        wm.invoke_popup(self, width=520)
        self._timer = wm.event_timer_add(0.5, window=context.window)
        wm.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type == "TIMER":
            _process_remote_job_queue(context.scene)
            if context.area:
                context.area.tag_redraw()

            scene_key = context.scene.as_pointer()
            status = context.scene.rfarm_status
            job_active = scene_key in ACTIVE_RENDER_JOBS and ACTIVE_RENDER_JOBS[scene_key].get("done") is not True

            if not job_active and not status.is_rendering:
                self._remove_timer(context)
                return {"FINISHED"}

            return {"RUNNING_MODAL"}

        if event.type in {"ESC", "RIGHTMOUSE"}:
            self._remove_timer(context)
            return {"CANCELLED"}

        return {"PASS_THROUGH"}

    def cancel(self, context):
        self._remove_timer(context)

    def _remove_timer(self, context):
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None

    def draw(self, context):
        status = context.scene.rfarm_status
        layout = self.layout

        if status.start_timestamp:
            start_dt = datetime.fromtimestamp(status.start_timestamp)
            layout.label(text=f"Service started: {start_dt.strftime('%Y-%m-%d %H:%M:%S')}")
            end_time = status.finish_timestamp if status.finish_timestamp else time.time()
            elapsed = max(0.0, end_time - status.start_timestamp)
        else:
            layout.label(text="Service not started")
            elapsed = 0.0

        layout.label(text=f"Elapsed time: {_format_elapsed(elapsed)}")
        layout.separator()
        layout.template_list(
            "RFarm_UL_status_log",
            "",
            status,
            "log_entries",
            status,
            "log_index",
            rows=8,
        )


classes = (
    RFarmAddonPreferences,
    RFarmLogEntry,
    RFarmStatus,
    RFarm_UL_status_log,
    RFarm_PT_panel,
    RFarm_OT_render_frame,
    RFarm_OT_render_status_popup,
)


def register():
    ACTIVE_RENDER_JOBS.clear()
    for cls in classes:
        register_class(cls)
    bpy.types.Scene.rfarm_status = PointerProperty(type=RFarmStatus)


def unregister():
    for job in list(ACTIVE_RENDER_JOBS.values()):
        callback = job.get("timer_callback")
        if callback:
            try:
                bpy.app.timers.unregister(callback)
            except ValueError:
                pass
    ACTIVE_RENDER_JOBS.clear()
    for cls in reversed(classes):
        unregister_class(cls)
    if hasattr(bpy.types.Scene, "rfarm_status"):
        del bpy.types.Scene.rfarm_status


if __name__ == "__main__":
    register()
