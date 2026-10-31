"""vidAI companion node.

Adds a small REST API next to ComfyUI's own routes so the vidAI web UI can
manage saved workflows on the server instead of only downloading them:

    GET    /vidai/workflows              list saved workflows
    POST   /vidai/workflows              save a workflow {name, meta, prompt}
    DELETE /vidai/workflows/{name}       delete a workflow
    GET    /vidai/outputs                list generated files on disk
    GET    /vidai/history                prompt history from disk
    GET    /vidai/progress               live progress of the running job
"""

import asyncio
import json
import os
import re
import shutil
import time
from pathlib import Path

from aiohttp import web
from server import PromptServer

PROJECT_DIR = Path(__file__).resolve().parents[3]
WORKFLOW_DIR = Path(os.environ.get("VIDAI_WORKFLOWS", PROJECT_DIR / "workflows")).expanduser()
COMFY_INPUT = Path(__file__).resolve().parents[2] / "input"
UI_CONVERTER = PROJECT_DIR / "scripts" / "workflow_to_ui.py"
HISTORY_LOG = PROJECT_DIR / "prompts" / "history.jsonl"


def slugify(text, limit=48):
    slug = re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")
    return slug[:limit].rstrip("-") or "workflow"


def workflow_files():
    return [p for p in sorted(WORKFLOW_DIR.glob("*.json"))
            if p.name != "index.json" and not p.name.endswith(".ui.json")]


def read_workflow(path):
    payload = json.loads(path.read_text())
    return payload.get("prompt", payload), payload.get("meta", {})


def write_index():
    entries = []
    for path in workflow_files():
        try:
            _, meta = read_workflow(path)
        except Exception:
            continue
        entries.append({
            "name": path.stem,
            "created": meta.get("created", ""),
            "prompt": meta.get("prompt", ""),
            "negative": meta.get("negative", ""),
            "model": meta.get("model", ""),
            "lora": meta.get("lora"),
            "steps": meta.get("steps"),
            "cfg": meta.get("cfg"),
            "size": meta.get("size"),
            "seed": meta.get("seed"),
            "sampler": meta.get("sampler"),
            "scheduler": meta.get("scheduler"),
            "has_ui": path.with_name(path.stem + ".ui.json").exists(),
        })
    WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)
    (WORKFLOW_DIR / "index.json").write_text(json.dumps({"workflows": entries}, indent=2))
    return entries


def build_ui_workflow(api_graph):
    """Convert the API graph into the canvas format ComfyUI's frontend loads.

    Runs in a worker thread: the converter fetches /object_info over HTTP from
    this same server, which would deadlock the event loop if called inline.
    """
    return asyncio.get_running_loop().run_in_executor(
        None, _build_ui_workflow_sync, api_graph)


def _build_ui_workflow_sync(api_graph):
    import importlib.util

    spec = importlib.util.spec_from_file_location("vidai_workflow_to_ui", UI_CONVERTER)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module.convert(api_graph, module.object_info())
    except Exception:
        return None


async def save_workflow(name, meta, api_graph):
    WORKFLOW_DIR.mkdir(parents=True, exist_ok=True)
    path = WORKFLOW_DIR / f"{name}.json"
    path.write_text(json.dumps({"meta": meta, "prompt": api_graph}, indent=2))

    ui = await build_ui_workflow(api_graph)
    if ui:
        ui_path = path.with_name(path.stem + ".ui.json")
        ui_path.write_text(json.dumps(ui, indent=2))
        COMFY_INPUT.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ui_path, COMFY_INPUT / ui_path.name)

    write_index()
    return path


@PromptServer.instance.routes.get("/vidai/workflows")
async def list_workflows(request):
    return web.json_response({"workflows": write_index()})


@PromptServer.instance.routes.post("/vidai/workflows")
async def create_workflow(request):
    data = await request.json()
    meta = data.get("meta", {})
    name = slugify(data.get("name") or meta.get("prompt", "workflow"))
    meta.setdefault("created", time.strftime("%Y-%m-%d %H:%M:%S"))
    path = await save_workflow(name, meta, data.get("prompt", {}))
    return web.json_response({"name": path.stem, "path": str(path)})


@PromptServer.instance.routes.delete("/vidai/workflows/{name}")
async def delete_workflow(request):
    name = request.match_info["name"]
    for suffix in (".json", ".ui.json"):
        path = WORKFLOW_DIR / f"{name}{suffix}"
        if path.exists():
            path.unlink()
    write_index()
    return web.json_response({"deleted": name})


@PromptServer.instance.routes.get("/vidai/history")
async def list_history(request):
    """Prompt history from the on-disk log — survives a ComfyUI restart."""
    entries = []
    if HISTORY_LOG.exists():
        for line in reversed(HISTORY_LOG.read_text().splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if len(entries) >= 100:
                break
    return web.json_response({"history": entries})


@PromptServer.instance.routes.post("/vidai/history")
async def append_history(request):
    """Append one entry to the prompt history log."""
    entry = await request.json()
    entry.setdefault("at", time.strftime("%Y-%m-%d %H:%M:%S"))
    HISTORY_LOG.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_LOG.open("a") as fh:
        fh.write(json.dumps(entry) + "\n")
    return web.json_response({"logged": True})


@PromptServer.instance.routes.get("/vidai/outputs")
async def list_outputs(request):
    """List generated files on disk — survives a ComfyUI restart, unlike /history."""
    import folder_paths

    output_dir = Path(folder_paths.get_output_directory())
    items = []
    for path in sorted(output_dir.rglob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
        if not path.is_file() or path.name.startswith("_"):
            continue
        stat = path.stat()
        items.append({
            "filename": path.name,
            "subfolder": str(path.parent.relative_to(output_dir)) if path.parent != output_dir else "",
            "type": "output",
            "size": stat.st_size,
            "mtime": int(stat.st_mtime * 1000),
        })
    return web.json_response({"outputs": items[:200]})


@PromptServer.instance.routes.get("/vidai/progress")
async def get_progress(request):
    """Live progress of the running job, read from ComfyUI's progress registry."""
    from comfy_execution.progress import get_progress_state

    registry = get_progress_state()
    nodes = []
    for node_id, entry in registry.nodes.items():
        value = entry.get("value", 0) or 0
        maximum = entry.get("max", 1) or 1
        nodes.append({
            "node": node_id,
            "state": entry.get("state").value if entry.get("state") else "pending",
            "value": value,
            "max": maximum,
            "percent": round(min(100.0, value / maximum * 100), 1) if maximum else 0,
        })
    nodes.sort(key=lambda n: n["node"])

    running = PromptServer.instance.prompt_queue.get_current_queue_volatile()[0]
    running_id = running[0][1] if running else None
    remaining = PromptServer.instance.prompt_queue.get_tasks_remaining()

    return web.json_response({
        "prompt_id": registry.prompt_id or running_id,
        "running_prompt_id": running_id,
        "queue_remaining": remaining,
        "nodes": nodes,
        "last_node_id": PromptServer.instance.last_node_id,
    })


WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
