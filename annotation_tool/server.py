#!/usr/bin/env python3
import base64
import hashlib
import io
import json
import mimetypes
import os
import re
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, unquote, urlparse
from urllib.request import Request, urlopen


class PathResolver:
    def __init__(self):
        cwd = Path.cwd().resolve()
        home = Path.home().resolve()
        env_roots = [p for p in os.environ.get("ANNOTATION_DATA_ROOTS", "").split(os.pathsep) if p.strip()]

        defaults = [
            cwd,
            cwd.parent,
            home / "Documents",
            home / "Downloads",
        ]

        self.roots = []
        for raw in [*env_roots, *defaults]:
            path = Path(raw).expanduser().resolve()
            if path.exists() and path not in self.roots:
                self.roots.append(path)

        self.cache = {}

    @staticmethod
    def _norm(path_str: str) -> str:
        return str(path_str).replace("\\", "/").strip()

    def resolve(self, incoming_path: str):
        target = self._norm(unquote(incoming_path or ""))
        if not target:
            return None, []

        if target in self.cache:
            return self.cache[target], []

        attempted = []
        path_obj = Path(target).expanduser()

        if path_obj.is_absolute() and path_obj.exists() and path_obj.is_file():
            resolved = path_obj.resolve()
            self.cache[target] = resolved
            return resolved, attempted

        relative = target.lstrip("/")
        for root in self.roots:
            candidate = (root / relative).resolve()
            attempted.append(str(candidate))
            if candidate.exists() and candidate.is_file():
                self.cache[target] = candidate
                return candidate, attempted

        # Fallback: basename search with suffix match
        wanted_base = Path(relative).name
        wanted_suffix = self._norm(relative)
        if wanted_base:
            for root in self.roots:
                try:
                    for candidate in root.rglob(wanted_base):
                        if not candidate.is_file():
                            continue
                        cand_norm = self._norm(candidate)
                        if cand_norm.endswith(wanted_suffix):
                            resolved = candidate.resolve()
                            self.cache[target] = resolved
                            return resolved, attempted
                except PermissionError:
                    continue

        return None, attempted


RESOLVER = PathResolver()
HF_TOKEN = os.environ.get("HF_TOKEN", "").strip()
HF_SAM2_MODEL = os.environ.get("HF_SAM2_MODEL", "facebook/sam2-hiera-large").strip()
HF_VLM_MODEL = os.environ.get("HF_VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()
HF_TIMEOUT_SECONDS = int(os.environ.get("HF_TIMEOUT_SECONDS", "90"))
HF_SAM2_INFERENCE_URL = os.environ.get("HF_SAM2_INFERENCE_URL", "").strip()
GOOGLE_AI_STUDIO_API_KEY = os.environ.get("GOOGLE_AI_STUDIO_API_KEY", "").strip()  # Placeholder for live Gemini calls
SAM_BACKEND = os.environ.get("SAM_BACKEND", os.environ.get("SAM2_BACKEND", "local")).strip().lower()  # local | hf | sam1
SAM2_LOCAL_CONFIG = os.environ.get("SAM2_LOCAL_CONFIG", "").strip()
SAM2_LOCAL_CHECKPOINT = os.environ.get("SAM2_LOCAL_CHECKPOINT", "").strip()
SAM2_LOCAL_DEVICE = os.environ.get("SAM2_LOCAL_DEVICE", "cpu").strip()
SAM2_LOCAL_MAX_BOXES = int(os.environ.get("SAM2_LOCAL_MAX_BOXES", "8"))
SAM1_CHECKPOINT = os.environ.get("SAM1_CHECKPOINT", "").strip()
SAM1_MODEL_TYPE = os.environ.get("SAM1_MODEL_TYPE", "vit_h").strip()  # vit_h | vit_l | vit_b
SAM1_DEVICE = os.environ.get("SAM1_DEVICE", "cpu").strip()
SAM1_MAX_BOXES = int(os.environ.get("SAM1_MAX_BOXES", "8"))

_SAM2_LOCAL_GENERATOR = None
_SAM2_LOCAL_LOCK = threading.Lock()
_SAM1_GENERATOR = None
_SAM1_LOCK = threading.Lock()


def _find_dataset_root() -> Path:
    candidates = [
        Path.cwd() / "Diagram_Attribution_Dataset",
        Path.cwd().parent / "Diagram_Attribution_Dataset",
    ]
    for root in RESOLVER.roots:
        candidates.append(root / "Diagram_Attribution_Dataset")
    for c in candidates:
        p = c.expanduser().resolve()
        if p.exists() and p.is_dir():
            return p
    # fallback (may not exist; callers handle errors)
    return (Path.cwd() / "Diagram_Attribution_Dataset").resolve()


DATASET_ROOT = _find_dataset_root()
PREDICTIONS_ROOT = (DATASET_ROOT.parent / "Predictions" / "gemini-pred").resolve()
PRED_ID_RE = re.compile(r"([^,\s/]+)/")
_PRED_INDEX = None
_PRED_LOCK = threading.Lock()

DEFAULT_GEMINI_QA_PROMPT = (
    "1 · Task Overview\n"
    "You are given a diagram image with rich visual and textual structure.\n"
    "Your task is to generate a high-quality question-answer pair that is answerable from the image alone.\n"
    "The QA should reflect visual reasoning over meaningful diagram content (objects, labels, legends, axes, arrows, relationships, quantities, or structure).\n"
    "\n"
    "2 · Response Format — STRICT\n"
    "Return strict JSON only with exactly these keys:\n"
    "{\n"
    "  \"question\": \"<string>\",\n"
    "  \"answer\": \"<string>\",\n"
    "  \"choices\": [\"<string>\", \"<string>\", ...]\n"
    "}\n"
    "Rules:\n"
    "- No markdown, no code fences, no additional keys.\n"
    "- If the question is free-response, choices must be [] (empty array).\n"
    "- If choices are provided, exactly one choice must match the answer string verbatim.\n"
    "\n"
    "3 · Question Quality Requirements\n"
    "The question must be:\n"
    "- Unambiguous and specific.\n"
    "- Grounded in visible evidence from the image.\n"
    "- Useful for diagram understanding (not trivial yes/no unless strongly justified by content).\n"
    "- Natural-sounding and concise.\n"
    "\n"
    "Prefer question types such as:\n"
    "- Identification (component/region/label)\n"
    "- Comparison (greater/less/equal, before/after, connected/not connected)\n"
    "- Retrieval (value, label, category, state, node, symbol)\n"
    "- Relationship/flow (next step, dependency, direction, linkage)\n"
    "\n"
    "4 · Answer Requirements\n"
    "- Must be directly supported by visual/textual evidence in the image.\n"
    "- Must be concise (typically 1–6 words unless numeric/phrase requires more).\n"
    "- Must not include explanation text.\n"
    "\n"
    "5 · Multiple-Choice Guidance\n"
    "If you provide choices:\n"
    "- Provide 3–5 options.\n"
    "- Include one correct option equal to \"answer\" exactly.\n"
    "- Distractors should be plausible and image-related, not random.\n"
    "- Avoid obviously wrong distractors.\n"
    "\n"
    "6 · Safety / Validity Constraints\n"
    "- Do not hallucinate content not present in the image.\n"
    "- Do not rely on outside/world knowledge.\n"
    "- Do not generate subjective/opinion-based questions.\n"
    "- If the image is unreadable or insufficient, still output valid JSON with:\n"
    "  question set to a simple observable question,\n"
    "  answer set to \"unclear\",\n"
    "  choices as [].\n"
)

DEFAULT_GEMINI_BBOX_PROMPT = (
    "1 · Task Overview\n"
    "You are given:\n"
    "(a) a diagram image,\n"
    "(b) a multiple-choice or free-text question and its correct answer,\n"
    "(c) rich region annotations, including both text and non-text elements.\n"
    "Your task is to identify all annotated visual regions a human would use to identify, verify, or reason through the correct answer. These include:\n"
    "\n"
    "Answer object: visual region directly depicting the answer.\n"
    "Answer label: text naming or labeling the answer object (if available).\n"
    "Supporting evidence: any other region visually needed to reason through the answer:\n"
    "legend swatches and legend text,\n"
    "axis ticks or scales,\n"
    "comparison or neighboring regions,\n"
    "arrows or flow connectors,\n"
    "associated text labels or visual structures.\n"
    "\n"
    "2 · Response Format — STRICT\n"
    "You must return exactly two lines:\n"
    "Answer: <ID1>/x,y,w,h,<ID2>/x,y,w,h,<ID3>/x,y,w,h\n"
    "Explanation: <Step-by-step, multi-sentence reasoning. Mention every ID with its role and how it contributed.>\n"
    "No extra spaces, no line breaks.\n"
    "No JSON, no additional formatting.\n"
    "Coordinates must follow the x,y,w,h format as in the annotations.\n"
    "In the Explanation:\n"
    "Follow Chain-of-Thought (CoT): walk through your reasoning step by step.\n"
    "Explain how the answer was derived using the selected regions.\n"
    "Explicitly mention each ID, and specify its role:\n"
    "Answer object, Answer label, Supporting evidence (Legend swatch, Legend text, Axis, Tick, Comparator, Neighbor, Flow Arrow, etc.)\n"
    "Justify why each region was necessary and sufficient.\n"
    "\n"
    "3 · Input\n"
    "Question_ID ............ {Question_ID_STRING}\n"
    "Question ............... {Question_String}\n"
    "Choices ................ {Choices string}\n"
    "Correct Answer ......... {Answer string}\n"
    "\n"
    "4 · Annotations\n"
    "Text Annotations: {Text_annotations}\n"
    "Polygon/BBox Annotations: {BBox_annotations}\n"
    "All Region IDs: {ALL_BBOX_ID_string}\n"
    "Optional Labels: {BBOX_LABEL_STRING}\n"
    "\n"
    "5 · Selection Criteria\n"
    "Select the minimal but sufficient regions required to answer the question correctly.\n"
    "Include:\n"
    "The answer object (e.g., map region, bar, line point, component),\n"
    "Its text label, if present (e.g., \"MS\"),\n"
    "Any legend elements used (swatch and text),\n"
    "All comparators or neighboring objects used in the reasoning,\n"
    "Axes, ticks, arrows, or connectors used visually.\n"
    "\n"
    "Do not:\n"
    "Include unreferenced or unrelated elements,\n"
    "Repeat region IDs,\n"
    "Use unannotated regions.\n"
    "\n"
    "6 · Diagram-Specific Guidance\n"
    "Maps\n"
    "Answer object: region polygon\n"
    "Label: state/country abbreviation\n"
    "Supporting: legend swatch and text, neighboring states if required\n"
    "Bar/Line Charts\n"
    "\n"
    "Answer object: bar, point, or segment\n"
    "Label: axis or legend text\n"
    "Supporting: legend, axis tick, comparison bars/points\n"
    "Flowcharts\n"
    "\n"
    "Answer object: flow node or step\n"
    "Label: text within node\n"
    "Supporting: arrows indicating flow, relevant preceding or following steps\n"
    "Infographics / Scientific Diagrams\n"
    "\n"
    "Answer object: labeled shape or visual entity\n"
    "Label: associated text\n"
    "Supporting: arrows, value indicators, visually linked regions\n"
    "Circuit Diagrams\n"
    "\n"
    "Answer object: electronic component\n"
    "Label: name, symbol, or identifier\n"
    "Supporting: wires, directional indicators, dependent or adjacent components\n"
)


def _read_json_body(handler):
    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    return json.loads(raw.decode("utf-8"))


def _extract_data_url_image(data_url: str):
    if not data_url or not data_url.startswith("data:"):
        return None, None
    try:
        header, encoded = data_url.split(",", 1)
        mime = "image/png"
        m = re.match(r"^data:([^;]+);base64$", header)
        if m:
            mime = m.group(1)
        return base64.b64decode(encoded), mime
    except Exception:
        return None, None


def _read_image_input(payload: dict):
    image_data_url = payload.get("image_data_url", "")
    image_path = payload.get("image_path", "")

    if image_data_url:
        img, mime = _extract_data_url_image(image_data_url)
        if img:
            return img, mime, "data_url"

    resolved, attempted = RESOLVER.resolve(image_path)
    if not resolved:
        return None, None, {"requested": image_path, "attempted": attempted[:10]}

    mime, _ = mimetypes.guess_type(str(resolved))
    return resolved.read_bytes(), (mime or "application/octet-stream"), str(resolved)


def _hf_request_json(url: str, payload: dict):
    if not HF_TOKEN:
        raise RuntimeError("Missing HF_TOKEN in environment.")

    req = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=HF_TIMEOUT_SECONDS) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HF HTTP {e.code}: {detail[:500]}") from e
    except URLError as e:
        raise RuntimeError(f"HF request failed: {e}") from e


def _extract_json_object(text: str):
    if not isinstance(text, str):
        return None
    block = text.strip()
    if block.startswith("```"):
        block = re.sub(r"^```(?:json)?\s*", "", block, flags=re.IGNORECASE)
        block = re.sub(r"\s*```$", "", block)
    m = re.search(r"\{[\s\S]*\}", block)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _extract_ids_from_answer_line(text: str):
    if not isinstance(text, str):
        return []
    # Accept both "Answer: ID/x,y,w,h,ID2/x,y,w,h" and free-form text containing ID/... tokens.
    line = ""
    for part in text.splitlines():
        if part.strip().lower().startswith("answer:"):
            line = part
            break
    if not line:
        return []
    ids = []
    for m in re.finditer(r"([A-Za-z0-9_.+\-]+)\s*/", line):
        ident = m.group(1).strip()
        if ident:
            ids.append(ident)
    return ids


def _gemini_generate_content(prompt: str, image_bytes: bytes | None = None, mime_type: str = "image/png", api_key: str | None = None):
    key = str(api_key or "").strip() or GOOGLE_AI_STUDIO_API_KEY
    if not key:
        raise RuntimeError("Missing Google AI Studio API key. Provide api_key in request or set GOOGLE_AI_STUDIO_API_KEY.")

    parts = [{"text": str(prompt or "").strip()}]
    if image_bytes:
        parts.append(
            {
                "inline_data": {
                    "mime_type": mime_type or "image/png",
                    "data": base64.b64encode(image_bytes).decode("utf-8"),
                }
            }
        )

    payload = {"contents": [{"role": "user", "parts": parts}]}
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent"
        f"?key={key}"
    )
    req = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=HF_TIMEOUT_SECONDS) as resp:
            out = json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Gemini HTTP {e.code}: {detail[:500]}") from e
    except URLError as e:
        raise RuntimeError(f"Gemini request failed: {e}") from e

    texts = []
    for cand in out.get("candidates", []) if isinstance(out, dict) else []:
        content = cand.get("content", {}) if isinstance(cand, dict) else {}
        for p in content.get("parts", []) if isinstance(content, dict) else []:
            if isinstance(p, dict) and isinstance(p.get("text"), str):
                texts.append(p["text"])
    if not texts:
        raise RuntimeError("Gemini response missing text output.")
    return "\n".join(texts).strip()


def _offline_qa_template():
    return {
        "question": "What is the main object or region of interest in this image?",
        "answer": "Needs annotator input",
        "choices": [],
    }


def _hf_request_binary(url: str, image_bytes: bytes, mime_type: str):
    if not HF_TOKEN:
        raise RuntimeError("Missing HF_TOKEN in environment.")

    req = Request(
        url,
        data=image_bytes,
        headers={
            "Authorization": f"Bearer {HF_TOKEN}",
            "Content-Type": mime_type,
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=HF_TIMEOUT_SECONDS) as resp:
            content = resp.read().decode("utf-8")
            return json.loads(content)
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"HF HTTP {e.code}: {detail[:500]}") from e
    except URLError as e:
        raise RuntimeError(f"HF request failed: {e}") from e


def _normalize_box_item(item: dict, idx: int):
    if not isinstance(item, dict):
        return None

    box = item.get("box") if isinstance(item.get("box"), dict) else item

    if isinstance(box.get("bbox"), (list, tuple)) and len(box.get("bbox")) >= 4:
        x, y, w, h = box["bbox"][:4]
    elif all(k in box for k in ("xmin", "ymin", "xmax", "ymax")):
        x = float(box["xmin"])
        y = float(box["ymin"])
        w = float(box["xmax"]) - x
        h = float(box["ymax"]) - y
    elif all(k in box for k in ("x", "y", "w", "h")):
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
    else:
        return None

    w = max(1.0, float(w))
    h = max(1.0, float(h))
    label = str(item.get("label") or item.get("category") or item.get("name") or "")
    score = item.get("score")
    meta = {"source": "sam2"}
    if score is not None:
        meta["score"] = score

    return {
        "id": item.get("id") or f"sam2_{idx+1}",
        "bbox": [float(x), float(y), w, h],
        "label": label,
        "meta": meta,
    }


def _extract_boxes_from_hf_output(output):
    candidates = []

    if isinstance(output, list):
        for idx, item in enumerate(output):
            norm = _normalize_box_item(item, idx)
            if norm:
                candidates.append(norm)
        return candidates

    if isinstance(output, dict):
        for key in ("boxes", "predictions", "outputs", "detections"):
            if isinstance(output.get(key), list):
                return _extract_boxes_from_hf_output(output[key])
        if "error" in output:
            raise RuntimeError(str(output["error"]))

    return []


def _load_local_sam2_generator():
    global _SAM2_LOCAL_GENERATOR
    with _SAM2_LOCAL_LOCK:
        if _SAM2_LOCAL_GENERATOR is not None:
            return _SAM2_LOCAL_GENERATOR

        if not SAM2_LOCAL_CONFIG or not SAM2_LOCAL_CHECKPOINT:
            raise RuntimeError(
                "Local SAM2 requires SAM2_LOCAL_CONFIG and SAM2_LOCAL_CHECKPOINT env vars."
            )

        config_name = SAM2_LOCAL_CONFIG.strip()
        # build_sam2 expects a Hydra config name (e.g. configs/sam2.1/sam2.1_hiera_l.yaml),
        # not an absolute filesystem path. Convert common absolute paths automatically.
        cfg_path = Path(config_name).expanduser()
        if cfg_path.is_absolute() and cfg_path.exists():
            cfg_norm = str(cfg_path).replace("\\", "/")
            marker = "/site-packages/sam2/"
            if marker in cfg_norm:
                config_name = cfg_norm.split(marker, 1)[1]
            else:
                # Last-resort mapping by filename for common shipped configs.
                cfg_base = cfg_path.name
                if cfg_base in ("sam2.1_hiera_t.yaml", "sam2.1_hiera_s.yaml", "sam2.1_hiera_b+.yaml", "sam2.1_hiera_l.yaml"):
                    config_name = f"configs/sam2.1/{cfg_base}"
                elif cfg_base in ("sam2_hiera_t.yaml", "sam2_hiera_s.yaml", "sam2_hiera_b+.yaml", "sam2_hiera_l.yaml"):
                    config_name = f"configs/sam2/{cfg_base}"

        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            from sam2.build_sam import build_sam2
        except Exception as e:
            raise RuntimeError(
                "SAM2 python package not available. Install local SAM2 dependencies first."
            ) from e

        model = build_sam2(config_name, SAM2_LOCAL_CHECKPOINT, device=SAM2_LOCAL_DEVICE)
        _SAM2_LOCAL_GENERATOR = SAM2AutomaticMaskGenerator(model)
        return _SAM2_LOCAL_GENERATOR


def suggest_boxes_via_local_sam2(image_bytes: bytes):
    try:
        import numpy as np
        from PIL import Image
    except Exception as e:
        raise RuntimeError("Local SAM2 requires Pillow and NumPy installed.") from e

    generator = _load_local_sam2_generator()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)
    masks = generator.generate(image_np)

    if not isinstance(masks, list) or not masks:
        raise RuntimeError("Local SAM2 returned no masks.")

    # Keep larger/higher-confidence candidates to avoid flooding annotators.
    def _rank_key(m):
        area = float(m.get("area") or 0.0)
        score = float(m.get("predicted_iou") or 0.0)
        return (area, score)

    ranked = sorted(masks, key=_rank_key, reverse=True)[: max(1, SAM2_LOCAL_MAX_BOXES)]
    boxes = []
    for i, m in enumerate(ranked):
        bbox = m.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue
        x, y, w, h = [float(v) for v in bbox[:4]]
        boxes.append(
            {
                "id": f"sam2_local_{i+1}",
                "bbox": [x, y, max(1.0, w), max(1.0, h)],
                "label": "",
                "meta": {
                    "source": "sam2_local",
                    "predicted_iou": float(m.get("predicted_iou") or 0.0),
                    "stability_score": float(m.get("stability_score") or 0.0),
                },
            }
        )

    if not boxes:
        raise RuntimeError("Local SAM2 masks did not include valid boxes.")
    return boxes


def suggest_boxes_via_hf_sam2(image_bytes: bytes, mime_type: str):
    candidate_urls = []
    if HF_SAM2_INFERENCE_URL:
        candidate_urls.append(HF_SAM2_INFERENCE_URL)
    candidate_urls.append(f"https://router.huggingface.co/hf-inference/models/{HF_SAM2_MODEL}")
    candidate_urls.append(f"https://api-inference.huggingface.co/models/{HF_SAM2_MODEL}")  # legacy fallback

    errors = []
    for model_url in candidate_urls:
        try:
            output = _hf_request_binary(model_url, image_bytes, mime_type)
            boxes = _extract_boxes_from_hf_output(output)
            if boxes:
                return boxes
            errors.append(f"{model_url}: no box predictions in response")
        except Exception as e:
            errors.append(f"{model_url}: {e}")

    raise RuntimeError(
        "SAM2 inference failed across available endpoints. "
        "Use a dedicated Hugging Face Inference Endpoint and set HF_SAM2_INFERENCE_URL, "
        "or choose a SAM2 model route that supports inference in your account/region. "
        f"Details: {' | '.join(errors[:3])}"
    )


def suggest_boxes_via_sam2(image_bytes: bytes, mime_type: str):
    if SAM_BACKEND == "sam1":
        return suggest_boxes_via_sam1(image_bytes)
    if SAM_BACKEND == "local":
        return suggest_boxes_via_local_sam2(image_bytes)
    return suggest_boxes_via_hf_sam2(image_bytes, mime_type)


def _load_local_sam1_generator():
    global _SAM1_GENERATOR
    with _SAM1_LOCK:
        if _SAM1_GENERATOR is not None:
            return _SAM1_GENERATOR

        if not SAM1_CHECKPOINT:
            raise RuntimeError("SAM1 requires SAM1_CHECKPOINT env var.")

        try:
            from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
        except Exception as e:
            raise RuntimeError(
                "segment-anything package not available. Install with: pip install segment-anything"
            ) from e

        if SAM1_MODEL_TYPE not in ("vit_h", "vit_l", "vit_b"):
            raise RuntimeError("SAM1_MODEL_TYPE must be one of: vit_h, vit_l, vit_b")

        sam = sam_model_registry[SAM1_MODEL_TYPE](checkpoint=SAM1_CHECKPOINT)
        try:
            sam.to(device=SAM1_DEVICE)
        except Exception:
            # Keep default if explicit device move fails.
            pass
        _SAM1_GENERATOR = SamAutomaticMaskGenerator(sam)
        return _SAM1_GENERATOR


def suggest_boxes_via_sam1(image_bytes: bytes):
    try:
        import numpy as np
        from PIL import Image
    except Exception as e:
        raise RuntimeError("SAM1 requires Pillow and NumPy installed.") from e

    generator = _load_local_sam1_generator()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(image)
    masks = generator.generate(image_np)

    if not isinstance(masks, list) or not masks:
        raise RuntimeError("SAM1 returned no masks.")

    def _rank_key(m):
        area = float(m.get("area") or 0.0)
        score = float(m.get("predicted_iou") or m.get("stability_score") or 0.0)
        return (area, score)

    ranked = sorted(masks, key=_rank_key, reverse=True)[: max(1, SAM1_MAX_BOXES)]
    boxes = []
    for i, m in enumerate(ranked):
        bbox = m.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue
        x, y, w, h = [float(v) for v in bbox[:4]]
        boxes.append(
            {
                "id": f"sam1_{i+1}",
                "bbox": [x, y, max(1.0, w), max(1.0, h)],
                "label": "",
                "meta": {
                    "source": "sam1_local",
                    "predicted_iou": float(m.get("predicted_iou") or 0.0),
                    "stability_score": float(m.get("stability_score") or 0.0),
                },
            }
        )
    if not boxes:
        raise RuntimeError("SAM1 masks did not include valid boxes.")
    return boxes


def _pick_best_box_for_point(boxes, x: float, y: float):
    if not isinstance(boxes, list) or not boxes:
        return None

    containing = []
    others = []
    for b in boxes:
        if not isinstance(b, dict):
            continue
        bbox = b.get("bbox")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) >= 4):
            continue
        bx, by, bw, bh = [float(v) for v in bbox[:4]]
        cx = bx + (bw / 2.0)
        cy = by + (bh / 2.0)
        area = max(1.0, bw * bh)
        dist2 = (cx - x) ** 2 + (cy - y) ** 2
        enriched = (dist2, area, b)
        if bx <= x <= bx + bw and by <= y <= by + bh:
            containing.append(enriched)
        else:
            others.append(enriched)

    if containing:
        # Prefer nearest center, then smaller area if tie.
        containing.sort(key=lambda t: (t[0], t[1]))
        return containing[0][2]
    if others:
        others.sort(key=lambda t: (t[0], t[1]))
        return others[0][2]
    return None


def suggest_box_for_point(image_bytes: bytes, mime_type: str, x: float, y: float):
    boxes = suggest_boxes_via_sam2(image_bytes, mime_type)
    best = _pick_best_box_for_point(boxes, float(x), float(y))
    if not best:
        raise RuntimeError("SAM2 could not find a box near the clicked point.")
    return best


def _point_source_tag() -> str:
    if SAM_BACKEND == "sam1":
        return "sam1_point"
    if SAM_BACKEND == "local":
        return "sam2_point"
    if SAM_BACKEND == "hf":
        return "sam2_hf_point"
    return "sam_point"


def _normalize_fallback_box(item: dict):
    if not isinstance(item, dict):
        return None
    ident = str(item.get("id") or item.get("bbox_id") or item.get("ann_id") or "").strip()
    if not ident:
        return None
    try:
        x = float(item.get("x"))
        y = float(item.get("y"))
        w = float(item.get("w"))
        h = float(item.get("h"))
    except Exception:
        return None
    if w <= 0 or h <= 0:
        return None
    return {
        "id": ident,
        "bbox": [x, y, w, h],
        "label": str(item.get("label") or ""),
        "meta": {"source": "fallback_catalog"},
    }


def generate_qa_via_hf(image_bytes: bytes, mime_type: str):
    data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('utf-8')}"
    prompt = (
        "Generate one concise visual question-answer pair for this image. "
        "Return strict JSON only with keys: question, answer, choices. "
        "choices must be an array (can be empty)."
    )
    payload = {
        "model": HF_VLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "max_tokens": 220,
        "temperature": 0.2,
    }
    output = _hf_request_json("https://router.huggingface.co/v1/chat/completions", payload)

    content = ""
    if isinstance(output, dict):
        choices = output.get("choices", [])
        if choices and isinstance(choices[0], dict):
            message = choices[0].get("message", {})
            if isinstance(message, dict):
                content = message.get("content", "") or ""
            elif isinstance(message, str):
                content = message

    if not content:
        raise RuntimeError("HF VLM response missing content.")

    json_match = re.search(r"\{[\s\S]*\}", content)
    parsed = None
    if json_match:
        try:
            parsed = json.loads(json_match.group(0))
        except json.JSONDecodeError:
            parsed = None

    if not isinstance(parsed, dict):
        parsed = {
            "question": "What is shown in this image?",
            "answer": content.strip()[:200],
            "choices": [],
        }

    return {
        "question": str(parsed.get("question") or "").strip(),
        "answer": str(parsed.get("answer") or "").strip(),
        "choices": parsed.get("choices") if isinstance(parsed.get("choices"), list) else [],
    }


def generate_missing_qa_via_gemini(
    image_bytes: bytes,
    mime_type: str,
    existing_qa: dict | None = None,
    prompt_override: str | None = None,
    api_key: str | None = None,
):
    existing_qa = existing_qa if isinstance(existing_qa, dict) else {}
    question = str(existing_qa.get("question") or "").strip()
    answer = str(existing_qa.get("answer") or "").strip()
    choices = existing_qa.get("choices") if isinstance(existing_qa.get("choices"), list) else []

    if question and answer:
        return {"question": question, "answer": answer, "choices": choices, "status": "unchanged"}

    prompt = str(prompt_override or DEFAULT_GEMINI_QA_PROMPT).strip()
    text = _gemini_generate_content(prompt, image_bytes=image_bytes, mime_type=mime_type, api_key=api_key)
    parsed = _extract_json_object(text)
    if not isinstance(parsed, dict):
        raise RuntimeError("Gemini QA output is not valid JSON.")

    final_q = question or str(parsed.get("question") or "").strip()
    final_a = answer or str(parsed.get("answer") or "").strip()
    final_choices = choices if choices else (parsed.get("choices") if isinstance(parsed.get("choices"), list) else [])
    return {"question": final_q, "answer": final_a, "choices": final_choices, "status": "generated"}


def generate_missing_boxes_via_gemini(image_bytes: bytes, mime_type: str, payload: dict, api_key: str | None = None):
    gt_boxes = payload.get("gt_boxes") if isinstance(payload.get("gt_boxes"), list) else []
    question = str(payload.get("question_text") or "").strip()
    answer = str(payload.get("answer_text") or "").strip()
    choices = payload.get("choices")
    if isinstance(choices, list):
        choices_text = ", ".join(str(c) for c in choices)
    else:
        choices_text = str(payload.get("choices_text") or "").strip()
    qid = str(payload.get("q_id") or "").strip()
    if not gt_boxes:
        return {"boxes": [], "status": "no_gt"}

    id_to_box = {}
    gt_lines = []
    for b in gt_boxes:
        if not isinstance(b, dict):
            continue
        ident = str(b.get("id") or "").strip()
        if not ident:
            continue
        id_to_box[_norm_key(ident)] = b
        label = str(b.get("label") or "").strip()
        kind = str(b.get("kind") or "").strip()
        gt_lines.append(f"- id: {ident}, label: {label}, kind: {kind}")

    prompt_head = str(payload.get("prompt") or DEFAULT_GEMINI_BBOX_PROMPT).strip()
    prompt = (
        f"{prompt_head}\n\n"
        f"3 · Input\n"
        f"Question_ID ............ {qid}\n"
        f"Question ............... {question}\n"
        f"Choices ................ {choices_text}\n"
        f"Correct Answer ......... {answer}\n\n"
        f"4 · Annotations\n"
        f"Polygon/BBox Annotations:\n" + "\n".join(gt_lines) + "\n"
        + f"All Region IDs: {', '.join(sorted(id_to_box.keys()))}\n"
    )
    text = _gemini_generate_content(prompt, image_bytes=image_bytes, mime_type=mime_type, api_key=api_key)
    parsed = _extract_json_object(text)
    raw_ids = []
    if isinstance(parsed, dict):
        raw_ids = parsed.get("box_ids")
        if not isinstance(raw_ids, list):
            raw_ids = parsed.get("boxes") if isinstance(parsed.get("boxes"), list) else []
    if not raw_ids:
        raw_ids = _extract_ids_from_answer_line(text)

    # Guardrail: if model returns almost all GT IDs, treat as degenerate and retry parse from strict Answer line.
    gt_count = max(1, len(id_to_box))
    if isinstance(raw_ids, list) and len(raw_ids) >= max(12, int(0.8 * gt_count)):
        answer_line_ids = _extract_ids_from_answer_line(text)
        if answer_line_ids:
            raw_ids = answer_line_ids

    out = []
    seen = set()
    for rid in raw_ids:
        ident = str(rid or "").strip()
        if not ident:
            continue
        match = id_to_box.get(_norm_key(ident))
        if not match:
            # tolerant id lookup
            for k in _id_variants(ident):
                match = id_to_box.get(_norm_key(k))
                if match:
                    break
        if not match:
            continue
        mid = str(match.get("id") or ident).strip()
        if mid in seen:
            continue
        seen.add(mid)
        out.append(
            {
                "id": mid,
                "x": float(match.get("x", 0)),
                "y": float(match.get("y", 0)),
                "w": float(match.get("w", 1)),
                "h": float(match.get("h", 1)),
                "source": "pred",
                "kind": str(match.get("kind") or "bbox"),
                "label": str(match.get("label") or ""),
            }
        )
    # Final guardrail: avoid returning near-complete GT as "prediction".
    if out and len(out) >= max(12, int(0.8 * gt_count)):
        return {
            "boxes": [],
            "status": "over_selected",
            "warning": f"Model selected too many boxes ({len(out)}/{gt_count}); treating as empty for safety.",
            "raw": parsed if isinstance(parsed, dict) else None,
            "raw_text": text,
        }

    return {"boxes": out, "status": "generated" if out else "empty", "raw": parsed if isinstance(parsed, dict) else None, "raw_text": text}


def _to_posix(path_obj: Path) -> str:
    return str(path_obj).replace("\\", "/")


def _is_in_dataset_root(path_obj: Path) -> bool:
    try:
        path_obj.resolve().relative_to(DATASET_ROOT)
        return True
    except Exception:
        return False


def _relative_dataset_path(path_obj: Path) -> str:
    if _is_in_dataset_root(path_obj):
        return _to_posix(path_obj.resolve().relative_to(DATASET_ROOT))
    return _to_posix(path_obj)


def _extract_gt_boxes(annotation_obj):
    boxes = []

    def add_box(ident, x, y, w, h, kind="bbox", label=""):
        try:
            x = float(x)
            y = float(y)
            w = max(1.0, float(w))
            h = max(1.0, float(h))
        except Exception:
            return
        boxes.append(
            {
                "id": str(ident),
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "source": "gt",
                "kind": kind,
                "label": str(label or ""),
            }
        )

    if not isinstance(annotation_obj, dict):
        return boxes

    # AI2D blobs/arrows/arrowHeads with bbox and text rectangles
    for key in ("blobs", "arrows", "arrowHeads"):
        group = annotation_obj.get(key, {})
        if isinstance(group, dict):
            for _, item in group.items():
                if not isinstance(item, dict):
                    continue
                bbox = item.get("bbox")
                if isinstance(bbox, dict):
                    ident = item.get("bbox_id") or item.get("id")
                    if ident:
                        add_box(ident, bbox.get("x"), bbox.get("y"), bbox.get("w"), bbox.get("h"), "bbox")

    for key in ("text", "texts"):
        group = annotation_obj.get(key, {})
        if isinstance(group, dict):
            for ident, item in group.items():
                if not isinstance(item, dict):
                    continue
                rect = item.get("rectangle")
                if isinstance(rect, list) and len(rect) == 2 and all(isinstance(p, list) and len(p) == 2 for p in rect):
                    x1, y1 = rect[0]
                    x2, y2 = rect[1]
                    add_box(item.get("id") or ident, x1, y1, float(x2) - float(x1), float(y2) - float(y1), "text", item.get("value") or item.get("replacementText") or "")

    # Circuit/Generic bbox arrays
    for key in ("bboxes", "boxes", "bbox", "annotations"):
        group = annotation_obj.get(key)
        if isinstance(group, list):
            for item in group:
                if not isinstance(item, dict):
                    continue
                ident = item.get("id") or item.get("bbox_id")
                bbox = item.get("bbox")
                if ident and isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    add_box(ident, bbox[0], bbox[1], bbox[2], bbox[3], "bbox", item.get("label") or "")
                elif ident and all(k in item for k in ("x", "y", "w", "h")):
                    add_box(ident, item["x"], item["y"], item["w"], item["h"], "bbox", item.get("label") or "")

    # ChartQA style models[].bboxes[]
    models = annotation_obj.get("models")
    if isinstance(models, list):
        for model in models:
            if not isinstance(model, dict):
                continue
            bboxes = model.get("bboxes")
            if isinstance(bboxes, list):
                for item in bboxes:
                    if not isinstance(item, dict):
                        continue
                    ident = item.get("bbox_id") or item.get("id")
                    if ident and all(k in item for k in ("x", "y", "w", "h")):
                        add_box(ident, item["x"], item["y"], item["w"], item["h"], "bbox")

    # MapWise/MapIQ transferred boxes: [x1,y1,x2,y2]
    tboxes = annotation_obj.get("transferred_bboxes")
    if isinstance(tboxes, list):
        for item in tboxes:
            if not isinstance(item, dict):
                continue
            ident = item.get("id")
            bbox = item.get("bbox")
            if ident and isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                x1, y1, x2, y2 = bbox[:4]
                add_box(ident, x1, y1, float(x2) - float(x1), float(y2) - float(y1), "bbox", item.get("label") or "")

    # Infographics OCR blocks with normalized geometry
    for key in ("PAGE", "LINE", "WORD"):
        blocks = annotation_obj.get(key)
        if isinstance(blocks, list):
            for item in blocks:
                if not isinstance(item, dict):
                    continue
                ident = item.get("id") or item.get("Id")
                geom = item.get("Geometry") or {}
                bb = geom.get("BoundingBox") if isinstance(geom, dict) else None
                if ident and isinstance(bb, dict):
                    add_box(ident, bb.get("Left", 0), bb.get("Top", 0), bb.get("Width", 0), bb.get("Height", 0), "text", item.get("Text") or "")

    # Deduplicate by ID while keeping first
    dedup = {}
    for b in boxes:
        dedup.setdefault(b["id"], b)
    return list(dedup.values())


def _build_records_for_question_file(dataset_name: str, q_file: Path, rel_image: str, rel_ann: str):
    records = []
    try:
        q_obj = json.loads(q_file.read_text(encoding="utf-8"))
    except Exception:
        return records

    def push(q_id, question_text, answer_text, choices=None):
        choices = choices if isinstance(choices, list) else []
        answer_text = "" if answer_text is None else str(answer_text)
        records.append(
            {
                "image_path": rel_image,
                "annotation_path": rel_ann,
                "question_path": _relative_dataset_path(q_file),
                "image_id": Path(rel_image).name,
                "q_id": str(q_id),
                "question_text": str(question_text or ""),
                "answer_text": answer_text,
                "correct_answer": answer_text,
                "ground_truth_answer": answer_text,
                "choices": [str(c) for c in choices],
            }
        )

    if dataset_name == "ai2d" and isinstance(q_obj, dict):
        questions = q_obj.get("questions", {})
        if isinstance(questions, dict):
            for question_text, info in questions.items():
                if not isinstance(info, dict):
                    continue
                choices = info.get("answerTexts") if isinstance(info.get("answerTexts"), list) else []
                idx = info.get("correctAnswer")
                answer_text = choices[idx] if isinstance(idx, int) and 0 <= idx < len(choices) else ""
                push(info.get("questionId") or f"{Path(rel_image).stem}-0", question_text, answer_text, choices)
        return records

    if dataset_name == "ChartQA" and isinstance(q_obj, dict):
        all_q = []
        for key in ("human", "augmented"):
            val = q_obj.get(key, [])
            if isinstance(val, list):
                all_q.extend(val)
        for item in all_q:
            if not isinstance(item, dict):
                continue
            push(item.get("question_id") or "", item.get("query") or "", item.get("label") or "", [])
        return records

    if dataset_name == "MapWise" and isinstance(q_obj, dict):
        for item in q_obj.get("data", []):
            if not isinstance(item, dict):
                continue
            push(item.get("q_id") or item.get("id") or "", item.get("question") or "", item.get("ground_truth") or "", [])
        return records

    if dataset_name == "mapIQ" and isinstance(q_obj, dict):
        for item in q_obj.get("qas", []):
            if not isinstance(item, dict):
                continue
            ans = item.get("ground_truth_answer")
            if ans is None:
                ans = item.get("answer")
            push(item.get("qid") or item.get("q_id") or "", item.get("question") or "", ans or "", item.get("choices") or [])
        return records

    if dataset_name == "circut-vqa" and isinstance(q_obj, dict):
        questions = q_obj.get("questions", [])
        if isinstance(questions, list):
            for item in questions:
                if not isinstance(item, dict):
                    continue
                push(item.get("q_id") or "", item.get("question") or "", item.get("answer") or "", [])
        return records

    if dataset_name == "infographicsvqa" and isinstance(q_obj, dict):
        for item in q_obj.get("data", []):
            if not isinstance(item, dict):
                continue
            answers = item.get("answers", [])
            answer_text = answers[0] if isinstance(answers, list) and answers else ""
            push(item.get("id") or item.get("questionId") or "", item.get("question") or "", answer_text, [])
        return records

    if dataset_name == "figure-vqa" and isinstance(q_obj, dict):
        for item in q_obj.get("qa_pairs", []):
            if not isinstance(item, dict):
                continue
            answer_text = item.get("answer")
            push(item.get("q_id") or item.get("question_id") or "", item.get("question_string") or "", answer_text, [])
        return records

    if dataset_name == "RealCQA" and isinstance(q_obj, list):
        for item in q_obj:
            if not isinstance(item, dict):
                continue
            push(item.get("id") or item.get("qa_id") or "", item.get("question") or "", item.get("answer") or "", [])
        return records

    return records


def _infer_question_annotation_paths(rel_image_path: str):
    rel = rel_image_path.replace("\\", "/").lstrip("/")
    parts = rel.split("/")
    if len(parts) < 3:
        return None, None, None
    dataset = parts[0]
    stem = Path(parts[-1]).stem
    suffix = Path(parts[-1]).suffix

    # dataset specific defaults
    if dataset == "ChartQA" and len(parts) >= 4:
        split = parts[1]
        q = DATASET_ROOT / "ChartQA" / split / "questions" / f"{stem}.json"
        a = DATASET_ROOT / "ChartQA" / split / "annotations" / f"{stem}.json"
        return dataset, q, a

    q = DATASET_ROOT / dataset / "questions" / f"{stem}.json"
    a = DATASET_ROOT / dataset / "annotations" / f"{stem}.json"
    return dataset, q, a


def load_records_by_image(image_path: str):
    resolved, attempted = RESOLVER.resolve(image_path)
    if not resolved:
        prefixed = f"Diagram_Attribution_Dataset/{str(image_path).lstrip('/')}"
        resolved, attempted2 = RESOLVER.resolve(prefixed)
        attempted = [*attempted, *attempted2]
    if not resolved:
        return None, {"error": "Image not found", "requested": image_path, "attempted": attempted[:10]}

    rel_image = _relative_dataset_path(resolved)
    dataset, q_file, ann_file = _infer_question_annotation_paths(rel_image)
    if not dataset:
        return None, {"error": "Could not infer dataset for image path", "image_path": rel_image}

    if not q_file or not q_file.exists():
        return None, {"error": "Question file not found for image", "image_path": rel_image, "question_path": _relative_dataset_path(q_file) if q_file else ""}

    if not ann_file or not ann_file.exists():
        return None, {"error": "Annotation file not found for image", "image_path": rel_image, "annotation_path": _relative_dataset_path(ann_file) if ann_file else ""}

    records = _build_records_for_question_file(dataset, q_file, rel_image, _relative_dataset_path(ann_file))
    if not records:
        return None, {"error": "No QA records parsed for image", "question_path": _relative_dataset_path(q_file)}

    try:
        ann_obj = json.loads(ann_file.read_text(encoding="utf-8"))
        gt_boxes = _extract_gt_boxes(ann_obj)
    except Exception:
        gt_boxes = []

    for rec in records:
        rec["bbox"] = gt_boxes

    return {
        "dataset": dataset,
        "image_path": rel_image,
        "question_path": _relative_dataset_path(q_file),
        "annotation_path": _relative_dataset_path(ann_file),
        "records": records,
    }, None


def load_records_by_uploaded_image(filename: str, image_data_url: str):
    image_bytes, _ = _extract_data_url_image(image_data_url or "")
    if not image_bytes:
        return None, {"error": "image_data_url is required and must be a valid data URL"}

    name = Path(str(filename or "")).name
    if not name:
        return None, {"error": "filename is required"}

    candidates = []
    try:
        candidates = [p for p in DATASET_ROOT.rglob(name) if p.is_file()]
    except Exception:
        candidates = []

    if not candidates:
        return None, {"error": "No dataset image found with same filename", "filename": name}

    if len(candidates) == 1:
        target = candidates[0]
    else:
        target = None
        uploaded_hash = hashlib.sha256(image_bytes).hexdigest()
        for c in candidates:
            try:
                if hashlib.sha256(c.read_bytes()).hexdigest() == uploaded_hash:
                    target = c
                    break
            except Exception:
                continue
        if target is None:
            return None, {
                "error": "Multiple dataset images matched filename; none matched file content hash",
                "filename": name,
                "candidate_count": len(candidates),
            }

    return load_records_by_image(_relative_dataset_path(target))


def load_next_image_records(current_image_path: str):
    resolved, attempted = RESOLVER.resolve(current_image_path)
    if not resolved:
        prefixed = f"Diagram_Attribution_Dataset/{str(current_image_path).lstrip('/')}"
        resolved, attempted2 = RESOLVER.resolve(prefixed)
        attempted = [*attempted, *attempted2]
    if not resolved:
        return None, {"error": "Current image not found", "requested": current_image_path, "attempted": attempted[:10]}

    rel_current = _relative_dataset_path(resolved)
    parts = rel_current.split("/")
    if len(parts) < 3:
        return None, {"error": "Could not infer dataset for current image", "image_path": rel_current}

    dataset = parts[0]
    current_name = Path(rel_current).name

    if dataset == "ChartQA" and len(parts) >= 4:
        split = parts[1]
        image_dir = DATASET_ROOT / "ChartQA" / split / "png"
    else:
        image_dir = DATASET_ROOT / dataset / "images"

    if not image_dir.exists():
        return None, {"error": "Image directory not found for dataset", "dataset": dataset, "image_dir": _to_posix(image_dir)}

    image_files = sorted([p for p in image_dir.iterdir() if p.is_file()], key=lambda p: p.name.lower())
    if not image_files:
        return None, {"error": "No images found in dataset image directory", "image_dir": _to_posix(image_dir)}

    start_idx = -1
    for i, p in enumerate(image_files):
        if p.name == current_name:
            start_idx = i
            break
    if start_idx < 0:
        return None, {"error": "Current image not in dataset image list", "image_path": rel_current}

    for next_file in image_files[start_idx + 1:]:
        rel_next = _relative_dataset_path(next_file)
        result, err = load_records_by_image(rel_next)
        if not err and result and result.get("records"):
            return result, None

    return None, {"error": "No next image with valid records found", "dataset": dataset, "current_image": rel_current}


def load_previous_image_records(current_image_path: str):
    resolved, attempted = RESOLVER.resolve(current_image_path)
    if not resolved:
        prefixed = f"Diagram_Attribution_Dataset/{str(current_image_path).lstrip('/')}"
        resolved, attempted2 = RESOLVER.resolve(prefixed)
        attempted = [*attempted, *attempted2]
    if not resolved:
        return None, {"error": "Current image not found", "requested": current_image_path, "attempted": attempted[:10]}

    rel_current = _relative_dataset_path(resolved)
    parts = rel_current.split("/")
    if len(parts) < 3:
        return None, {"error": "Could not infer dataset for current image", "image_path": rel_current}

    dataset = parts[0]
    current_name = Path(rel_current).name

    if dataset == "ChartQA" and len(parts) >= 4:
        split = parts[1]
        image_dir = DATASET_ROOT / "ChartQA" / split / "png"
    else:
        image_dir = DATASET_ROOT / dataset / "images"

    if not image_dir.exists():
        return None, {"error": "Image directory not found for dataset", "dataset": dataset, "image_dir": _to_posix(image_dir)}

    image_files = sorted([p for p in image_dir.iterdir() if p.is_file()], key=lambda p: p.name.lower())
    if not image_files:
        return None, {"error": "No images found in dataset image directory", "image_dir": _to_posix(image_dir)}

    start_idx = -1
    for i, p in enumerate(image_files):
        if p.name == current_name:
            start_idx = i
            break
    if start_idx < 0:
        return None, {"error": "Current image not in dataset image list", "image_path": rel_current}

    for prev_file in reversed(image_files[:start_idx]):
        rel_prev = _relative_dataset_path(prev_file)
        result, err = load_records_by_image(rel_prev)
        if not err and result and result.get("records"):
            return result, None

    return None, {"error": "No previous image with valid records found", "dataset": dataset, "current_image": rel_current}


def _norm_key(v: str) -> str:
    return str(v or "").strip().lower()


def _parse_prediction_ids(answer_text: str):
    if not isinstance(answer_text, str):
        return []
    ids = []
    for m in PRED_ID_RE.finditer(answer_text):
        ident = m.group(1).strip()
        if ident:
            ids.append(ident)
    return ids


def _id_variants(ident: str):
    base = str(ident or "").strip()
    if not base:
        return []
    out = [base]
    low = base.lower()
    if low != base:
        out.append(low)
    if low.endswith("_val"):
        stripped = base[:-4]
        out.append(stripped)
        if stripped.lower() != stripped:
            out.append(stripped.lower())
    return out


def _build_prediction_index():
    index = {}
    by_qid = {}
    if not PREDICTIONS_ROOT.exists():
        return {"by_q_ann": index, "by_qid": by_qid}

    for fp in sorted(PREDICTIONS_ROOT.glob("*.json")):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            continue
        items = data.values() if isinstance(data, dict) else data if isinstance(data, list) else []
        for item in items:
            if not isinstance(item, dict):
                continue
            q_id = str(item.get("q_id") or "").strip()
            ann = str(item.get("annotation") or item.get("annotation_path") or "").strip()
            img = str(item.get("image") or item.get("image_path") or "").strip()
            answer = str(item.get("answer") or "")
            rec = {
                "q_id": q_id,
                "annotation_path": ann,
                "image_path": img,
                "answer": answer,
                "pred_ids": _parse_prediction_ids(answer),
            }
            if q_id:
                key = (_norm_key(q_id), _norm_key(ann))
                index.setdefault(key, rec)
                by_qid.setdefault(_norm_key(q_id), []).append(rec)
    return {"by_q_ann": index, "by_qid": by_qid}


def _get_prediction_index():
    global _PRED_INDEX
    with _PRED_LOCK:
        if _PRED_INDEX is None:
            _PRED_INDEX = _build_prediction_index()
        return _PRED_INDEX


def predict_boxes_from_saved_outputs(payload: dict):
    q_id = str(payload.get("q_id") or "").strip()
    ann = str(payload.get("annotation_path") or "").strip()
    img = str(payload.get("image_path") or "").strip()
    gt_boxes = payload.get("gt_boxes") if isinstance(payload.get("gt_boxes"), list) else []

    pred_index = _get_prediction_index()
    rec = pred_index["by_q_ann"].get((_norm_key(q_id), _norm_key(ann)))
    if rec is None:
        candidates = pred_index["by_qid"].get(_norm_key(q_id), [])
        if len(candidates) == 1:
            rec = candidates[0]
        elif candidates:
            img_norm = _norm_key(img)
            rec = next((c for c in candidates if _norm_key(c.get("image_path")) == img_norm), candidates[0])

    if not rec:
        return {"boxes": [], "source": "saved_predictions", "status": "not_found"}

    gt_by_id = {}
    for b in gt_boxes:
        if not isinstance(b, dict):
            continue
        ident = str(b.get("id") or "").strip()
        if ident:
            for key in _id_variants(ident):
                gt_by_id[key] = b

    predicted = []
    seen = set()
    for ident in rec.get("pred_ids", []):
        source_gt = None
        for key in _id_variants(ident):
            if key in gt_by_id:
                source_gt = gt_by_id[key]
                break
        if source_gt is None:
            continue
        out_id = str(source_gt.get("id") or ident)
        if out_id in seen:
            continue
        seen.add(out_id)
        predicted.append(
            {
                "id": out_id,
                "x": float(source_gt.get("x", 0)),
                "y": float(source_gt.get("y", 0)),
                "w": float(source_gt.get("w", 1)),
                "h": float(source_gt.get("h", 1)),
                "source": "pred",
                "kind": source_gt.get("kind", "bbox"),
                "label": str(source_gt.get("label", "")),
            }
        )

    return {
        "boxes": predicted,
        "source": "saved_predictions",
        "status": "ok" if predicted else "empty",
        "key": {
            "q_id": q_id,
            "annotation_path": ann,
            "image_path": img,
        },
        "note": "Dummy mode: loaded from Predictions/gemini-pred. Switch to live Gemini by implementing AI Studio call with GOOGLE_AI_STUDIO_API_KEY.",
    }


class Handler(SimpleHTTPRequestHandler):
    def _write_json(self, status_code: int, payload: dict):
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/health":
            return self._write_json(
                200,
                {
                    "ok": True,
                    "roots": [str(r) for r in RESOLVER.roots],
                    "capabilities": {
                        "sam_backend": SAM_BACKEND,
                        "sam2_backend": SAM_BACKEND,  # backward-compatible alias
                        "sam2_local_ready": bool(SAM2_LOCAL_CONFIG and SAM2_LOCAL_CHECKPOINT),
                        "sam1_ready": bool(SAM1_CHECKPOINT),
                        "hf_token_configured": bool(HF_TOKEN),
                        "google_ai_studio_key_configured": bool(GOOGLE_AI_STUDIO_API_KEY),
                    },
                },
            )

        if parsed.path == "/api/image":
            params = parse_qs(parsed.query)
            raw_path = params.get("path", [""])[0]
            resolved, attempted = RESOLVER.resolve(raw_path)
            if not resolved:
                return self._write_json(
                    404,
                    {
                        "error": "Image not found",
                        "requested": raw_path,
                        "attempted": attempted[:10],
                    },
                )

            ctype, _ = mimetypes.guess_type(str(resolved))
            ctype = ctype or "application/octet-stream"
            data = resolved.read_bytes()

            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)
            return

        return super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        try:
            payload = _read_json_body(self)
        except json.JSONDecodeError:
            return self._write_json(400, {"error": "Invalid JSON body"})

        if parsed.path == "/api/suggest-boxes":
            image_bytes, mime_type, source_info = _read_image_input(payload)
            if not image_bytes:
                return self._write_json(404, {"error": "Image not found", "details": source_info})

            try:
                boxes = suggest_boxes_via_sam2(image_bytes, mime_type)
                if SAM_BACKEND == "sam1":
                    model_name = f"sam1_{SAM1_MODEL_TYPE}"
                else:
                    model_name = "sam2_local" if SAM_BACKEND == "local" else HF_SAM2_MODEL
                return self._write_json(200, {"boxes": boxes, "source": source_info, "model": model_name, "backend": SAM_BACKEND})
            except Exception as e:
                return self._write_json(
                    502,
                    {
                        "error": str(e),
                        "model": HF_SAM2_MODEL if SAM_BACKEND == "hf" else ("sam2_local" if SAM_BACKEND == "local" else f"sam1_{SAM1_MODEL_TYPE}"),
                        "hint": "For hf backend, set HF_SAM2_INFERENCE_URL to a working SAM2 endpoint URL.",
                    },
                )

        if parsed.path == "/api/sam-point-box":
            image_bytes, mime_type, source_info = _read_image_input(payload)
            if not image_bytes:
                return self._write_json(404, {"error": "Image not found", "details": source_info})

            try:
                x = float(payload.get("x"))
                y = float(payload.get("y"))
            except Exception:
                return self._write_json(400, {"error": "x and y are required numeric coordinates in image space"})

            try:
                box = suggest_box_for_point(image_bytes, mime_type, x, y)
                box["meta"] = {**(box.get("meta") or {}), "source": _point_source_tag()}
                return self._write_json(
                    200,
                    {
                        "box": box,
                        "point": {"x": x, "y": y},
                        "source": source_info,
                        "backend": SAM_BACKEND,
                        "model": "sam2_local" if SAM_BACKEND == "local" else (f"sam1_{SAM1_MODEL_TYPE}" if SAM_BACKEND == "sam1" else HF_SAM2_MODEL),
                    },
                )
            except Exception as e:
                # Fallback path for local testing when SAM backend is unavailable:
                # choose nearest box from client-provided catalog (GT/predicted list).
                fallback_catalog = payload.get("gt_boxes")
                fallback_norm = []
                if isinstance(fallback_catalog, list):
                    for item in fallback_catalog:
                        nb = _normalize_fallback_box(item)
                        if nb:
                            fallback_norm.append(nb)
                fb_best = _pick_best_box_for_point(fallback_norm, x, y) if fallback_norm else None
                if fb_best:
                    fb_best["meta"] = {**(fb_best.get("meta") or {}), "source": "sam2_point_fallback"}
                    return self._write_json(
                        200,
                        {
                            "box": fb_best,
                            "point": {"x": x, "y": y},
                            "source": source_info,
                            "backend": "fallback_catalog",
                            "model": "none",
                            "warning": str(e),
                        },
                    )
                return self._write_json(502, {"error": str(e), "backend": SAM_BACKEND, "model": HF_SAM2_MODEL})

        if parsed.path == "/api/generate-qa":
            image_bytes, mime_type, source_info = _read_image_input(payload)
            if not image_bytes:
                return self._write_json(404, {"error": "Image not found", "details": source_info})

            if not HF_TOKEN:
                return self._write_json(
                    200,
                    {
                        "qa": _offline_qa_template(),
                        "source": source_info,
                        "model": "offline_template",
                        "warning": "HF_TOKEN not set; returned offline QA template.",
                    },
                )

            try:
                qa = generate_qa_via_hf(image_bytes, mime_type)
                return self._write_json(200, {"qa": qa, "source": source_info, "model": HF_VLM_MODEL})
            except Exception as e:
                return self._write_json(500, {"error": str(e), "model": HF_VLM_MODEL})

        if parsed.path == "/api/generate-missing-qa":
            image_bytes, mime_type, source_info = _read_image_input(payload)
            if not image_bytes:
                return self._write_json(404, {"error": "Image not found", "details": source_info})

            try:
                qa = generate_missing_qa_via_gemini(
                    image_bytes=image_bytes,
                    mime_type=mime_type,
                    existing_qa=payload.get("existing_qa") if isinstance(payload.get("existing_qa"), dict) else {},
                    prompt_override=str(payload.get("prompt") or "").strip() or None,
                    api_key=str(payload.get("api_key") or "").strip() or None,
                )
                return self._write_json(200, {"qa": qa, "source": source_info, "model": GEMINI_MODEL, "provider": "google_ai_studio"})
            except Exception as e:
                return self._write_json(500, {"error": str(e), "model": GEMINI_MODEL, "provider": "google_ai_studio"})

        if parsed.path == "/api/records-by-image":
            image_path = str(payload.get("image_path") or "").strip()
            if not image_path:
                return self._write_json(400, {"error": "image_path is required"})

            result, err = load_records_by_image(image_path)
            if err:
                return self._write_json(404, err)
            return self._write_json(200, result)

        if parsed.path == "/api/records-by-upload-image":
            filename = str(payload.get("filename") or "").strip()
            image_data_url = str(payload.get("image_data_url") or "")
            result, err = load_records_by_uploaded_image(filename, image_data_url)
            if err:
                return self._write_json(404, err)
            return self._write_json(200, result)

        if parsed.path == "/api/predict-boxes":
            try:
                result = predict_boxes_from_saved_outputs(payload)
                return self._write_json(200, result)
            except Exception as e:
                return self._write_json(500, {"error": str(e)})

        if parsed.path == "/api/generate-missing-bboxes":
            image_bytes, mime_type, source_info = _read_image_input(payload)
            if not image_bytes:
                return self._write_json(404, {"error": "Image not found", "details": source_info})
            try:
                result = generate_missing_boxes_via_gemini(
                    image_bytes=image_bytes,
                    mime_type=mime_type,
                    payload=payload,
                    api_key=str(payload.get("api_key") or "").strip() or None,
                )
                result["model"] = GEMINI_MODEL
                result["provider"] = "google_ai_studio"
                result["source"] = source_info
                return self._write_json(200, result)
            except Exception as e:
                return self._write_json(500, {"error": str(e), "model": GEMINI_MODEL, "provider": "google_ai_studio"})

        if parsed.path == "/api/next-image-records":
            image_path = str(payload.get("image_path") or "").strip()
            if not image_path:
                return self._write_json(400, {"error": "image_path is required"})
            result, err = load_next_image_records(image_path)
            if err:
                return self._write_json(404, err)
            return self._write_json(200, result)

        if parsed.path == "/api/previous-image-records":
            image_path = str(payload.get("image_path") or "").strip()
            if not image_path:
                return self._write_json(400, {"error": "image_path is required"})
            result, err = load_previous_image_records(image_path)
            if err:
                return self._write_json(404, err)
            return self._write_json(200, result)

        return self._write_json(404, {"error": "Unknown endpoint"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"Serving on http://127.0.0.1:{port}/")
    print("API health: http://127.0.0.1:%d/api/health" % port)
    server.serve_forever()
