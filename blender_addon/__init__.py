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
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Tuple

import bpy
from bpy.props import BoolProperty, PointerProperty, StringProperty
from bpy.types import AddonPreferences, Operator, Panel, PropertyGroup
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
        layout.operator(RFarm_OT_render_frame.bl_idname, icon="RENDER_STILL")


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


class UploadURLNotSupported(RuntimeError):
    """Raised when the remote worker does not expose the /upload-url endpoint."""


def _build_payload(
    scene,
    blend_gcs_uri: Optional[str],
    blend_inline: Optional[str],
    preferences: RFarmAddonPreferences,
) -> dict:
    payload = {
        "frame": scene.frame_current,
        "render_settings": _collect_render_metadata(scene),
    }

    if blend_gcs_uri:
        payload["blend_gcs_uri"] = blend_gcs_uri
    if blend_inline:
        payload["blend_file"] = blend_inline

    compute_device = "CUDA"
    cycles = scene.cycles
    if getattr(cycles, "device", "CPU") == "GPU":
        prefs = bpy.context.preferences.addons.get("cycles")
        if prefs:
            compute_device = prefs.preferences.compute_device_type
        else:
            compute_device = "CUDA"
    payload["device"] = cycles.device
    payload["compute_device_type"] = compute_device

    return payload


def _request_upload_url(
    preferences: RFarmAddonPreferences, blend_path: Path
) -> Tuple[str, str]:
    url = preferences.endpoint.rstrip("/") + "/upload-url"
    headers = {"Content-Type": "application/json"}
    if preferences.auth_token:
        headers["Authorization"] = f"Bearer {preferences.auth_token}"

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

        status.is_rendering = True
        status.last_error = ""
        status.last_output_path = ""
        self.report({"INFO"}, "Submitting remote render job...")

        try:
            gcs_uri: Optional[str] = None
            blend_inline: Optional[str] = None
            try:
                upload_url, gcs_uri = _request_upload_url(prefs, blend_path)
            except UploadURLNotSupported:
                with open(blend_path, "rb") as handle:
                    blend_inline = base64.b64encode(handle.read()).decode("ascii")
                self.report(
                    {"INFO"},
                    "Remote worker is outdated, falling back to inline upload.",
                )
            else:
                _upload_blend_to_gcs(upload_url, blend_path)
        except (requests.RequestException, RuntimeError) as exc:
            status.is_rendering = False
            status.last_error = str(exc)
            self.report({"ERROR"}, f"Failed to upload blend file: {exc}")
            shutil.rmtree(tmpdir, ignore_errors=True)
            return {"CANCELLED"}

        payload = _build_payload(scene, gcs_uri, blend_inline, prefs)
        url = prefs.endpoint.rstrip("/") + "/render"

        headers = {"Content-Type": "application/json"}
        if prefs.auth_token:
            headers["Authorization"] = f"Bearer {prefs.auth_token}"

        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=300)
        except requests.RequestException as exc:
            status.is_rendering = False
            status.last_error = str(exc)
            self.report({"ERROR"}, f"Request failed: {exc}")
            shutil.rmtree(tmpdir, ignore_errors=True)
            return {"CANCELLED"}

        if response.status_code >= 400:
            status.is_rendering = False
            status.last_error = f"HTTP {response.status_code}: {response.text}"
            self.report({"ERROR"}, status.last_error)
            shutil.rmtree(tmpdir, ignore_errors=True)
            return {"CANCELLED"}

        try:
            payload_response = response.json()
        except ValueError:
            status.is_rendering = False
            status.last_error = "Invalid JSON response"
            self.report({"ERROR"}, status.last_error)
            shutil.rmtree(tmpdir, ignore_errors=True)
            return {"CANCELLED"}

        status.last_job_id = payload_response.get("job_id", "")
        image_base64 = payload_response.get("image_base64")
        remote_format = payload_response.get("output_format", "PNG")
        output_path = self._resolve_output_path(scene)

        if image_base64 and output_path:
            _ensure_output_directory(output_path)
            try:
                with open(output_path, "wb") as out_file:
                    out_file.write(base64.b64decode(image_base64))
                status.last_output_path = str(output_path)
                self.report({"INFO"}, f"Remote render complete: {output_path}")
            except OSError as exc:
                status.last_error = str(exc)
                self.report({"ERROR"}, f"Failed to write output: {exc}")
        elif image_base64:
            # Fall back to a temporary directory inside Blender's session.
            temp_dir = Path(bpy.app.tempdir or tempfile.gettempdir())
            temp_dir.mkdir(parents=True, exist_ok=True)
            fallback_name = f"rfarm_frame.{remote_format.lower()}"
            temp_path = temp_dir / fallback_name
            try:
                with open(temp_path, "wb") as out_file:
                    out_file.write(base64.b64decode(image_base64))
                status.last_output_path = str(temp_path)
                self.report({"INFO"}, f"Saved remote render to temporary path: {temp_path}")
            except OSError as exc:
                status.last_error = str(exc)
                self.report({"ERROR"}, f"Failed to write temporary output: {exc}")
        elif not image_base64:
            status.last_error = "Response missing image data"
            self.report({"WARNING"}, "Remote worker did not return image bytes.")

        status.is_rendering = False
        shutil.rmtree(tmpdir, ignore_errors=True)
        return {"FINISHED"}


classes = (
    RFarmAddonPreferences,
    RFarmStatus,
    RFarm_PT_panel,
    RFarm_OT_render_frame,
)


def register():
    for cls in classes:
        register_class(cls)
    bpy.types.Scene.rfarm_status = PointerProperty(type=RFarmStatus)


def unregister():
    for cls in reversed(classes):
        unregister_class(cls)
    if hasattr(bpy.types.Scene, "rfarm_status"):
        del bpy.types.Scene.rfarm_status


if __name__ == "__main__":
    register()
