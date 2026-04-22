import base64
import hashlib
import io
import json
import os
import time
from pathlib import Path

try:
    import openai
except ImportError:
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])
    import openai

try:
    import numpy as np
    from PIL import Image
except ImportError:
    np = None
    Image = None

try:
    from aiohttp import web
    from server import PromptServer
except Exception:
    web = None
    PromptServer = None


BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-5.3-chat-latest"
REASONING_EFFORT_OPTS = ["default", "minimal", "low", "medium", "high", "xhigh"]

NODE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = NODE_DIR / "uploads"
KB_CACHE_PATH = NODE_DIR / "kb_cache.json"


# ---------------------------------------------------------------------------
# Image tensor conversion
# ---------------------------------------------------------------------------

def _tensor_to_pngs(image_tensor):
    if image_tensor is None or np is None or Image is None:
        return []
    try:
        arr = image_tensor.detach().cpu().numpy() if hasattr(image_tensor, "detach") else np.asarray(image_tensor)
    except Exception:
        arr = np.asarray(image_tensor)

    if arr.ndim == 3:
        arr = arr[None, ...]

    out = []
    for frame in arr:
        frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
        mode = "RGBA" if frame.shape[-1] == 4 else "RGB"
        pil = Image.fromarray(frame, mode=mode)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        out.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    return out


def tensor_to_responses_image_parts(image_tensor, image_detail="auto"):
    parts = []
    for b64 in _tensor_to_pngs(image_tensor):
        parts.append(
            {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{b64}",
                "detail": image_detail,
            }
        )
    return parts


# ---------------------------------------------------------------------------
# Knowledge upload + cache
# ---------------------------------------------------------------------------

def _sanitize_filename(name: str) -> str:
    # Keep original filename for better UX (including CJK),
    # only strip path/control-risky characters.
    base = os.path.basename(name or "").strip().replace("\x00", "")
    base = base.replace("/", "_").replace("\\", "_")
    if base in ("", ".", ".."):
        return "knowledge.md"
    return base


def _ensure_dirs():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def _read_cache():
    if not KB_CACHE_PATH.exists():
        return {"stores": {}}
    try:
        return json.loads(KB_CACHE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"stores": {}}


def _write_cache(cache_obj):
    KB_CACHE_PATH.write_text(json.dumps(cache_obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _parse_knowledge_files(raw_text: str):
    text = (raw_text or "").strip()
    if not text:
        return []

    # UI writes one filename per line.
    items = [x.strip() for x in text.splitlines() if x.strip()]

    resolved = []
    for item in items:
        p = Path(item)
        if p.is_absolute():
            fp = p
        else:
            fp = UPLOAD_DIR / item
        if fp.suffix.lower() != ".md":
            continue
        if fp.exists() and fp.is_file():
            resolved.append(fp)

    unique = []
    seen = set()
    for p in resolved:
        s = str(p)
        if s not in seen:
            seen.add(s)
            unique.append(p)
    return unique


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _kb_fingerprint(paths):
    parts = []
    for p in sorted(paths, key=lambda x: str(x).lower()):
        st = p.stat()
        parts.append(
            {
                "name": p.name,
                "path": str(p),
                "size": st.st_size,
                "mtime": int(st.st_mtime),
                "sha256": _file_sha256(p),
            }
        )
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), parts


def _poll_vs_file_ready(client, vector_store_id: str, vector_store_file_id: str, timeout_sec=300):
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        item = client.vector_stores.files.retrieve(
            vector_store_id=vector_store_id,
            file_id=vector_store_file_id,
        )
        status = getattr(item, "status", None)
        if status == "completed":
            return
        if status in ("failed", "cancelled"):
            raise RuntimeError(f"Vector store file indexing failed: status={status}")
        time.sleep(1.5)
    raise TimeoutError("Vector store file indexing timeout")


def _create_or_reuse_vector_store(client, file_paths):
    fingerprint, file_meta = _kb_fingerprint(file_paths)
    cache_obj = _read_cache()
    stores = cache_obj.setdefault("stores", {})

    cached = stores.get(fingerprint)
    if cached and cached.get("vector_store_id"):
        return cached["vector_store_id"], "reused", file_meta

    vs = client.vector_stores.create(name="ComfyUI GPT5 Knowledge")
    vs_id = getattr(vs, "id", None)
    if not vs_id:
        raise RuntimeError("Failed to create vector store")

    create_and_poll = getattr(getattr(client.vector_stores, "files", object()), "create_and_poll", None)

    for path in file_paths:
        with path.open("rb") as fh:
            f = client.files.create(file=fh, purpose="assistants")
        file_id = getattr(f, "id", None)
        if not file_id:
            raise RuntimeError(f"Failed to upload file: {path.name}")

        if callable(create_and_poll):
            create_and_poll(vector_store_id=vs_id, file_id=file_id)
        else:
            attached = client.vector_stores.files.create(vector_store_id=vs_id, file_id=file_id)
            vs_file_id = getattr(attached, "id", None)
            if not vs_file_id:
                raise RuntimeError(f"Failed to attach file to vector store: {path.name}")
            _poll_vs_file_ready(client, vs_id, vs_file_id)

    stores[fingerprint] = {
        "vector_store_id": vs_id,
        "updated_at": int(time.time()),
        "files": file_meta,
    }
    _write_cache(cache_obj)
    return vs_id, "created", file_meta


# ---------------------------------------------------------------------------
# Responses API
# ---------------------------------------------------------------------------

def _extract_citations(resp):
    citations = []
    output_items = getattr(resp, "output", None) or []

    for item in output_items:
        if getattr(item, "type", None) != "message":
            continue
        content = getattr(item, "content", None) or []
        for c in content:
            if getattr(c, "type", None) != "output_text":
                continue
            anns = getattr(c, "annotations", None) or []
            for ann in anns:
                if getattr(ann, "type", None) != "file_citation":
                    continue
                citations.append(
                    {
                        "file_id": getattr(ann, "file_id", ""),
                        "filename": getattr(ann, "filename", ""),
                    }
                )

    uniq = []
    seen = set()
    for x in citations:
        key = (x.get("file_id", ""), x.get("filename", ""))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(x)
    return uniq


def _extract_file_search_calls(resp):
    calls = []
    for item in (getattr(resp, "output", None) or []):
        if getattr(item, "type", None) != "file_search_call":
            continue
        results = getattr(item, "results", None)
        if results is None:
            results = getattr(item, "search_results", None)
        hit_filenames = []
        hit_file_ids = []
        for r in (results or []):
            if isinstance(r, dict):
                fname = r.get("filename") or r.get("file_name")
                fid = r.get("file_id")
            else:
                fname = getattr(r, "filename", None) or getattr(r, "file_name", None)
                fid = getattr(r, "file_id", None)
            if fname:
                hit_filenames.append(str(fname))
            if fid:
                hit_file_ids.append(str(fid))
        # Keep stable order and uniqueness
        hit_filenames = list(dict.fromkeys(hit_filenames))
        hit_file_ids = list(dict.fromkeys(hit_file_ids))
        calls.append(
            {
                "id": getattr(item, "id", ""),
                "status": getattr(item, "status", ""),
                "queries": list(getattr(item, "queries", None) or []),
                "results_count": len(results or []),
                "hit_filenames": hit_filenames,
                "hit_file_ids": hit_file_ids,
            }
        )
    return calls


def _extract_first_json_object(text: str):
    if not text:
        return None
    s = text.strip()
    try:
        return json.loads(s)
    except Exception:
        pass

    # Fallback: extract first top-level {...} block
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(s)):
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = s[start:i + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    return None
    return None


def build_scene_brief_instructions(system_content: str):
    return (
        (system_content or "")
        + "\n\nYou are in retrieval-and-structuring mode."
        + " Do not produce final creative prose."
        + " Return only strict JSON with keys:"
        + " core_mood, visual_anchors, interactions, scene_layers, lighting_style, constraints."
    )


def build_scene_brief_user_text(prompt: str):
    return (
        "Task: Build a structured scene brief from the prompt and available references.\n"
        "Output JSON only.\n"
        f"User prompt: {prompt}"
    )


def build_final_generation_instructions(system_content: str):
    return (
        (system_content or "")
        + "\n\nGeneration mode: produce the final fused prompt only."
        + " Use the provided structured_scene_brief as mandatory grounding."
    )


def call_responses(
    client,
    model,
    instructions,
    input_items,
    max_tokens,
    reasoning_effort,
    vector_store_id=None,
    force_file_search=False,
    retries=3,
):
    kwargs = {
        "model": model,
        "input": input_items,
    }
    if instructions:
        kwargs["instructions"] = instructions
    if max_tokens and max_tokens > 0:
        kwargs["max_output_tokens"] = max_tokens
    if reasoning_effort and reasoning_effort != "default":
        kwargs["reasoning"] = {"effort": reasoning_effort}
    if vector_store_id:
        kwargs["tools"] = [{"type": "file_search", "vector_store_ids": [vector_store_id], "max_num_results": 8}]
        kwargs["include"] = ["file_search_call.results"]
        if force_file_search:
            kwargs["tool_choice"] = "required"

    last_err = None
    for attempt in range(retries):
        try:
            resp = client.responses.create(**kwargs)
            text = getattr(resp, "output_text", "") or ""
            status = getattr(resp, "status", "completed")
            return text, status, resp, kwargs
        except openai.AuthenticationError:
            raise
        except openai.BadRequestError as ex:
            msg = str(ex).lower()
            # Auto-heal unsupported reasoning options.
            if "reasoning" in msg or "effort" in msg:
                kwargs.pop("reasoning", None)
                resp = client.responses.create(**kwargs)
                return (getattr(resp, "output_text", "") or ""), getattr(resp, "status", "completed"), resp, kwargs
            # Some gateways may reject tool_choice/include. Degrade gracefully.
            if "tool_choice" in msg or "include" in msg:
                kwargs.pop("tool_choice", None)
                kwargs.pop("include", None)
                resp = client.responses.create(**kwargs)
                return (getattr(resp, "output_text", "") or ""), getattr(resp, "status", "completed"), resp, kwargs
            raise
        except openai.OpenAIError as ex:
            last_err = ex
            if attempt >= retries - 1:
                raise
            time.sleep(2 * (attempt + 1))

    if last_err:
        raise last_err


def _summarize_input_items(input_items):
    summary = []
    for item in input_items:
        role = item.get("role")
        content = item.get("content", [])
        cc = []
        for part in content:
            if part.get("type") == "input_image":
                url = part.get("image_url", "")
                cc.append({"type": "input_image", "detail": part.get("detail"), "image_url": f"{url[:40]}...({len(url)} bytes)"})
            else:
                cc.append(part)
        summary.append({"role": role, "content": cc})
    return summary


# ---------------------------------------------------------------------------
# Upload API route for node frontend
# ---------------------------------------------------------------------------

def _register_upload_route():
    if PromptServer is None or web is None:
        return

    _ensure_dirs()

    @PromptServer.instance.routes.post("/gpt5/upload_md")
    async def gpt5_upload_md(request):
        reader = await request.multipart()
        uploaded = []

        while True:
            part = await reader.next()
            if part is None:
                break
            if part.name != "files":
                continue

            raw_name = part.filename or "knowledge.md"
            name = _sanitize_filename(raw_name)
            if not name.lower().endswith(".md"):
                continue

            target = UPLOAD_DIR / name

            with target.open("wb") as f:
                while True:
                    chunk = await part.read_chunk()
                    if not chunk:
                        break
                    f.write(chunk)

            uploaded.append({"name": target.name})

        return web.json_response({"files": uploaded})


_register_upload_route()


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class GPT5ChatNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": False}),
                "system_content": (
                    "STRING",
                    {
                        "default": "You are a helpful assistant. Answer concisely.",
                        "multiline": True,
                        "dynamicPrompts": False,
                    },
                ),
                "model_name": ("STRING", {"default": DEFAULT_MODEL, "multiline": False}),
                "max_output_tokens": ("INT", {"default": 16384, "min": 0, "max": 131072, "step": 64}),
                "reasoning_effort": (REASONING_EFFORT_OPTS, {"default": "default"}),
                "image_detail": (["auto", "low", "high"], {"default": "high"}),
                "knowledge_files": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "dynamicPrompts": False,
                        "placeholder": "Upload .md files using the node button. One filename per line.",
                    },
                ),
            },
            "optional": {
                "images": ("IMAGE",),
                "api_key": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("text", "request_debug", "knowledge_info", "finish_reason", "citations")
    FUNCTION = "run"
    CATEGORY = "GPT5/Chat"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return time.time()

    def run(
        self,
        prompt,
        system_content,
        model_name,
        max_output_tokens,
        reasoning_effort,
        image_detail,
        knowledge_files,
        images=None,
        api_key=None,
    ):
        if not api_key:
            raise ValueError("api_key is required. Please connect a STRING input with your OpenAI API key.")

        client = openai.OpenAI(api_key=api_key, base_url=BASE_URL)

        image_tensors = [images] if images is not None else []
        user_content = [{"type": "input_text", "text": prompt}]
        for t in image_tensors:
            user_content.extend(tensor_to_responses_image_parts(t, image_detail))
        input_items = [{"role": "user", "content": user_content}]

        kb_files = _parse_knowledge_files(knowledge_files)
        vector_store_id = None
        kb_status = "off"
        kb_file_names = []

        if kb_files:
            vector_store_id, kb_status, _ = _create_or_reuse_vector_store(client, kb_files)
            kb_file_names = [p.name for p in kb_files]

        two_stage_used = False
        scene_brief = None
        scene_brief_raw = ""
        stage1_file_search_calls = []
        stage1_kwargs = {}

        try:
            # Stage 1: retrieve + structure (JSON brief)
            stage1_content = [{"type": "input_text", "text": build_scene_brief_user_text(prompt)}]
            for t in image_tensors:
                stage1_content.extend(tensor_to_responses_image_parts(t, image_detail))
            stage1_input_items = [{"role": "user", "content": stage1_content}]

            stage1_text, _, stage1_resp, stage1_kwargs = call_responses(
                client=client,
                model=model_name,
                instructions=build_scene_brief_instructions(system_content),
                input_items=stage1_input_items,
                max_tokens=min(max_output_tokens, 4096),
                reasoning_effort=reasoning_effort,
                vector_store_id=vector_store_id,
                force_file_search=bool(vector_store_id),
            )
            scene_brief_raw = stage1_text or ""
            scene_brief = _extract_first_json_object(scene_brief_raw)
            stage1_file_search_calls = _extract_file_search_calls(stage1_resp)

            # Stage 2: final generation grounded by structured brief
            brief_payload = scene_brief if isinstance(scene_brief, dict) else {"raw_brief": scene_brief_raw}
            stage2_content = [
                {"type": "input_text", "text": prompt},
                {
                    "type": "input_text",
                    "text": "structured_scene_brief:\n" + json.dumps(brief_payload, ensure_ascii=False),
                },
            ]
            for t in image_tensors:
                stage2_content.extend(tensor_to_responses_image_parts(t, image_detail))
            stage2_input_items = [{"role": "user", "content": stage2_content}]

            text, finish_reason, raw_resp, sent_kwargs = call_responses(
                client=client,
                model=model_name,
                instructions=build_final_generation_instructions(system_content),
                input_items=stage2_input_items,
                max_tokens=max_output_tokens,
                reasoning_effort=reasoning_effort,
                vector_store_id=None,
                force_file_search=False,
            )
            two_stage_used = True
        except Exception:
            # Fallback to single-shot mode to keep node robust.
            text, finish_reason, raw_resp, sent_kwargs = call_responses(
                client=client,
                model=model_name,
                instructions=system_content,
                input_items=input_items,
                max_tokens=max_output_tokens,
                reasoning_effort=reasoning_effort,
                vector_store_id=vector_store_id,
                force_file_search=bool(vector_store_id),
            )

        citations = _extract_citations(raw_resp)
        file_search_calls = _extract_file_search_calls(raw_resp)
        if stage1_file_search_calls:
            file_search_calls = stage1_file_search_calls + file_search_calls

        debug = json.dumps(
            {
                "api": "responses",
                "base_url": BASE_URL,
                "model": model_name,
                "has_knowledge": bool(vector_store_id),
                "request": {
                    "instructions_len": len(system_content or ""),
                    "input_items": _summarize_input_items(input_items),
                    "max_output_tokens": sent_kwargs.get("max_output_tokens"),
                    "reasoning": sent_kwargs.get("reasoning"),
                    "tools": sent_kwargs.get("tools", []),
                    "tool_choice": sent_kwargs.get("tool_choice"),
                    "include": sent_kwargs.get("include", []),
                },
                "two_stage_used": two_stage_used,
                "stage1_request": {
                    "tools": stage1_kwargs.get("tools", []),
                    "tool_choice": stage1_kwargs.get("tool_choice"),
                    "include": stage1_kwargs.get("include", []),
                    "scene_brief_parsed": isinstance(scene_brief, dict),
                },
            },
            ensure_ascii=False,
            indent=2,
        )

        knowledge_info = json.dumps(
            {
                "status": kb_status,
                "vector_store_id": vector_store_id,
                "files": kb_file_names,
                "file_search_calls": file_search_calls,
                "retrieval_used": len(file_search_calls) > 0,
                "two_stage_used": two_stage_used,
            },
            ensure_ascii=False,
            indent=2,
        )

        return (
            text,
            debug,
            knowledge_info,
            str(finish_reason),
            json.dumps(citations, ensure_ascii=False, indent=2),
        )


NODE_CLASS_MAPPINGS = {
    "GPT5ChatNode": GPT5ChatNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GPT5ChatNode": "GPT-5 Vision + Knowledge",
}
