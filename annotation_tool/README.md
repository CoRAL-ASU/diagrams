# Universal Annotation Tool (Phase 1 + Phase 2)

Dataset-agnostic browser tool for:
- Uploading JSON records (single object or array)
- Auto-detecting dataset shape
- Normalizing into one universal schema
- Viewing/editing QA fields
- Drawing/editing bounding boxes
- Optional AI assist:
  - Generate missing bounding boxes (SAM2 via Hugging Face API)
  - Generate missing QA pair (VLM via Hugging Face API)
- Exporting unified JSON per record

## Project Structure

- `frontend/index.html`
- `frontend/main.js`
- `frontend/datasetDetector.js`
- `frontend/schemaNormalizer.js`
- `frontend/renderer.js`
- `frontend/annotationEngine.js`
- `frontend/styles.css`

## Run Locally

Run the local server (serves frontend + resolves image paths from JSON):

```bash
cd /Users/tampu/Documents/Diagram_Attribution/annotation_tool
python3 server.py
```

Then open:

`http://127.0.0.1:8000/frontend/`

## Notes

- JSON `image` / `image_path` values are resolved through `/api/image`.
- For dataset paths outside common roots, set `ANNOTATION_DATA_ROOTS` before launch, for example:

```bash
export ANNOTATION_DATA_ROOTS="/Users/tampu/Documents/Reviewed/Reviewed"
python3 server.py
```
- Exports are browser downloads: `*_annotated.json` and `*_annotated.png` are saved to your browser's default Downloads folder.
- Phase 2 requires a Hugging Face token:

```bash
export HF_TOKEN="hf_..."
```

- Optional model overrides:

```bash
export HF_SAM2_MODEL="facebook/sam2-hiera-large"
export HF_SAM2_INFERENCE_URL="https://<your-sam2-inference-endpoint>"
export HF_VLM_MODEL="Qwen/Qwen2.5-VL-7B-Instruct"
export HF_TIMEOUT_SECONDS="90"
```

- Important for SAM2:
  - Serverless router access for SAM2 may return `404/410` depending on model/provider availability.
  - For reliable SAM2 integration, deploy/use a dedicated Hugging Face Inference Endpoint and set `HF_SAM2_INFERENCE_URL`.

- Local SAM2 (no endpoint charges):
  - Set `SAM2_BACKEND=local`
  - Install dependencies and SAM2 package in your Python env:

```bash
pip install numpy pillow torch torchvision
pip install git+https://github.com/facebookresearch/sam2.git
```

  - Download SAM2 checkpoint/config locally and set:

```bash
export SAM2_BACKEND="local"
export SAM2_LOCAL_CONFIG="/absolute/path/to/sam2_hiera_l.yaml"
export SAM2_LOCAL_CHECKPOINT="/absolute/path/to/sam2_hiera_large.pt"
export SAM2_LOCAL_DEVICE="cuda"   # or cpu
export SAM2_LOCAL_MAX_BOXES="8"
```

- Quick free setup (recommended):

```bash
./scripts/setup_local_free.sh
export SAM2_LOCAL_CONFIG="/absolute/path/to/sam2_config.yaml"
export SAM2_LOCAL_CHECKPOINT="/absolute/path/to/sam2_checkpoint.pt"
./scripts/run_local_free.sh
```

- In fully free mode, if `HF_TOKEN` is not set:
  - Box suggestion still uses local SAM2.
  - QA generation returns an offline template placeholder (annotator can edit).

- In UI:
  - `Generate Missing Boxes` is enabled only when no boxes exist.
  - `Generate Missing QA` is enabled only when question or answer is missing.
  - `AI Status` panel shows endpoint, model, success/error, and response summary.
- The universal output format is:

```json
{
  "dataset_type": "...",
  "image": "...",
  "qa": {
    "question": "",
    "answer": "",
    "choices": []
  },
  "annotations": [
    {
      "id": "",
      "bbox": [0, 0, 0, 0],
      "label": "",
      "meta": {}
    }
  ],
  "metadata": {}
}
```

## Current Scope

Includes:
- Phase 1 universal core (dataset-agnostic normalization + annotation/edit/export)
- Phase 2 optional AI assist (SAM2 missing-box suggestion + HF QA generation)

## Phase 2 Test Flow

1. Export your token and start server:

```bash
export HF_TOKEN="hf_..."
export ANNOTATION_DATA_ROOTS="/Users/tampu/Documents/Reviewed"
python3 server.py
```

2. Open `http://127.0.0.1:8000/frontend/`
3. Upload a JSON with missing boxes or missing QA.
4. Click `Load Record`.
5. If boxes are missing, click `Generate Missing Boxes`.
6. If question/answer is missing, click `Generate Missing QA`.
7. Inspect `AI Status` for success/error details.
8. Export JSON/image to validate final outputs.
