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
import sys
import time
from pathlib import Path

from aiohttp import web
from server import PromptServer

PROJECT_DIR = Path(__file__).resolve().parents[3]
WORKFLOW_DIR = Path(os.environ.get("VIDAI_WORKFLOWS", PROJECT_DIR / "workflows")).expanduser()
COMFY_INPUT = Path(__file__).resolve().parents[2] / "input"
UI_CONVERTER = PROJECT_DIR / "scripts" / "workflow_to_ui.py"
HISTORY_LOG = PROJECT_DIR / "prompts" / "history.jsonl"

# The cloud package lives next to this repo, not inside ComfyUI.
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


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


# ---------------------------------------------------------------------------
# Cloud execution tier
#
#   GET    /vidai/cloud/settings        read the admin settings
#   POST   /vidai/cloud/settings        update them
#   GET    /vidai/cloud/providers        which providers are usable right now
#   POST   /vidai/cloud/draft           draft a workflow with the LLM
#   POST   /vidai/cloud/submit          queue a job to be rendered on a GPU
#   GET    /vidai/cloud/queue           queue + rental state
#   GET    /vidai/cloud/costs           spend summary, jobs and rentals
#   POST   /vidai/cloud/score           score a rendered image with the LLM
#   GET    /vidai/cloud/offers          live prices from the providers
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parents[3]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def _cloud():
    """Import the cloud package lazily so ComfyUI still starts without it."""
    from cloud import config as cloud_config
    return cloud_config


@PromptServer.instance.routes.get("/vidai/cloud/settings")
async def cloud_get_settings(request):
    from cloud import config as cloud_config
    from cloud import registry
    settings = cloud_config.get_config().get()
    return web.json_response({
        "settings": settings,
        "available_providers": registry.available_providers(),
        "configured": {
            "vast": bool(cloud_config.provider_key("vast")),
            "runpod": bool(cloud_config.provider_key("runpod")),
        },
    })


@PromptServer.instance.routes.post("/vidai/cloud/settings")
async def cloud_update_settings(request):
    from cloud import config as cloud_config
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    patch = payload.get("settings") if isinstance(payload, dict) else None
    if not isinstance(patch, dict):
        return web.json_response({"error": "expected {\"settings\": {...}}"}, status=400)
    try:
        updated = cloud_config.get_config().update(patch)
    except ValueError as exc:
        return web.json_response({"error": str(exc)}, status=400)
    return web.json_response({"settings": updated})


@PromptServer.instance.routes.get("/vidai/cloud/providers")
async def cloud_providers(request):
    from cloud import registry
    from cloud import config as cloud_config
    return web.json_response({
        "available": registry.available_providers(),
        "configured": {
            "vast": bool(cloud_config.provider_key("vast")),
            "runpod": bool(cloud_config.provider_key("runpod")),
        },
        "default_provider": cloud_config.get_config().get().get("default_provider"),
        "routing_mode": cloud_config.get_config().get().get("routing_mode"),
    })


@PromptServer.instance.routes.post("/vidai/cloud/draft")
async def cloud_draft(request):
    from cloud import drafter
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    user_request = (payload or {}).get("request") or ""
    if not user_request:
        return web.json_response({"error": "expected {\"request\": \"...\"}"}, status=400)
    try:
        params = drafter.draft_workflow(
            user_request,
            model=(payload or {}).get("model"),
            candidates=(payload or {}).get("candidates"),
            llm_model=(payload or {}).get("llm_model"),
        )
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=502)
    return web.json_response(params)


@PromptServer.instance.routes.post("/vidai/cloud/submit")
async def cloud_submit(request):
    from cloud import orchestrator
    from cloud.base import JobRequest
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    workflow = (payload or {}).get("workflow")
    if not isinstance(workflow, dict) or not workflow:
        return web.json_response({"error": "expected {\"workflow\": {...}}"}, status=400)
    kind = (payload or {}).get("kind") or "image"
    request_obj = JobRequest(
        workflow=workflow,
        kind=kind,
        timeout_seconds=int((payload or {}).get("timeout_seconds") or 3600),
        label=(payload or {}).get("label") or "cloud",
    )
    job_id = orchestrator.get_orchestrator().submit(
        request_obj,
        prompt=(payload or {}).get("prompt") or "",
        label=request_obj.label,
    )
    return web.json_response({"job_id": job_id, "queued": True})


@PromptServer.instance.routes.get("/vidai/cloud/queue")
async def cloud_queue(request):
    from cloud import orchestrator
    return web.json_response(orchestrator.get_orchestrator().queue_state())


@PromptServer.instance.routes.get("/vidai/cloud/costs")
async def cloud_costs(request):
    from cloud import ledger
    jobs = ledger.list_jobs(limit=int(request.query.get("limit", 50)))
    for job in jobs:
        job["amortised_cost_usd"] = round(
            ledger.amortised_cost(job.get("rental_id")), 8)
    return web.json_response({
        "summary": ledger.job_summary(),
        "jobs": jobs,
        "rentals": ledger.list_rentals(limit=int(request.query.get("limit", 50))),
    })


@PromptServer.instance.routes.post("/vidai/cloud/score")
async def cloud_score(request):
    from cloud import drafter
    try:
        payload = await request.json()
    except Exception:
        payload = {}
    path = (payload or {}).get("path") or ""
    prompt = (payload or {}).get("prompt") or ""
    if not path or not Path(path).exists():
        return web.json_response({"error": "expected {\"path\": \"...\"}"}, status=400)
    try:
        result = drafter.score_image(path, prompt,
                                     llm_model=(payload or {}).get("llm_model"))
    except Exception as exc:
        return web.json_response({"error": str(exc)}, status=502)
    return web.json_response(result)


@PromptServer.instance.routes.get("/vidai/cloud/offers")
async def cloud_offers(request):
    """Live prices from whichever provider the admin asked for."""
    from cloud import registry
    from cloud import config as cloud_config
    provider = request.query.get("provider") or \
        cloud_config.get_config().get().get("default_provider") or "vast"
    if provider == "local":
        return web.json_response({"provider": provider, "offers": []})
    try:
        instance_provider = registry.build_provider(provider)
        offers = instance_provider.search_offers()
    except Exception as exc:
        return web.json_response({"provider": provider, "error": str(exc),
                                  "offers": []}, status=200)
    slim = []
    for offer in offers[:20]:
        slim.append({
            "id": offer.get("id"),
            "gpu": offer.get("gpu_name") or offer.get("gpu_type_id"),
            "dph": offer.get("dph_total") or offer.get("community_price"),
            "vram_mb": offer.get("gpu_totalram"),
            "disk_gb": offer.get("disk_space"),
            "location": offer.get("geolocation") or offer.get("country_code"),
        })
    return web.json_response({"provider": provider, "offers": slim})


def _start_cloud():
    """Kick off the queue worker and create the cost ledger."""
    try:
        from cloud import ledger, orchestrator
        ledger.init_db()
        orchestrator.get_orchestrator().start()
    except Exception as exc:      # never block ComfyUI startup
        print(f"[vidai_ui] cloud tier not started: {exc}")


_start_cloud()


WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
