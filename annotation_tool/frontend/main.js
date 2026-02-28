import { detectDatasetType } from "./datasetDetector.js";
import { normalizeToUniversal } from "./schemaNormalizer.js";
import {
  loadImageFromSource,
  setRawJsonView,
  populateQaFields,
  readQaFields,
  exportJson,
  exportDataUrl
} from "./renderer.js";
import { AnnotationEngine } from "./annotationEngine.js";

const els = {
  imageUpload: document.getElementById("imageUpload"),
  geminiApiKey: document.getElementById("geminiApiKey"),
  prevBtn: document.getElementById("prevBtn"),
  nextBtn: document.getElementById("nextBtn"),
  recordInfo: document.getElementById("recordInfo"),
  rawJsonView: document.getElementById("rawJsonView"),
  aiStatusView: document.getElementById("aiStatusView"),
  gtBoxList: document.getElementById("gtBoxList"),
  datasetType: document.getElementById("datasetType"),
  questionField: document.getElementById("questionField"),
  answerField: document.getElementById("answerField"),
  choicesField: document.getElementById("choicesField"),
  qaActionBtn: document.getElementById("qaActionBtn"),
  bboxActionBtn: document.getElementById("bboxActionBtn"),
  samSelectPointsBtn: document.getElementById("samSelectPointsBtn"),
  samGenerateBtn: document.getElementById("samGenerateBtn"),
  exportAllBtn: document.getElementById("exportAllBtn"),
  drawModeBtn: document.getElementById("drawModeBtn"),
  undoBtn: document.getElementById("undoBtn"),
  deleteBoxBtn: document.getElementById("deleteBoxBtn"),
  clearBoxesBtn: document.getElementById("clearBoxesBtn"),
  canvas: document.getElementById("annotationCanvas"),
  canvasHint: document.getElementById("canvasHint"),
  labelModal: document.getElementById("labelModal"),
  labelModalBoxId: document.getElementById("labelModalBoxId"),
  newBoxLabelInput: document.getElementById("newBoxLabelInput"),
  newBoxLabelSaveBtn: document.getElementById("newBoxLabelSaveBtn"),
  newBoxLabelCancelBtn: document.getElementById("newBoxLabelCancelBtn")
};

const state = {
  rawData: null,
  records: [],
  currentIndex: 0,
  currentUniversal: null,
  uploadedImageDataUrl: "",
  uploadedImageName: "",
  drawMode: false,
  history: [],
  historyIndex: -1,
  suppressHistory: false,
  health: null,
  pendingNewBoxId: "",
  sessionGeminiApiKey: "",
  samPoints: [],
  samPointSelectionArmed: false
};

const MAX_GENERATE_IMAGE_DIM = 1400;
const MAX_BASE64_CHARS = 2_000_000;

const canvasContainer = document.getElementById("canvasContainer");
const engine = new AnnotationEngine(
  els.canvas,
  document.getElementById("canvasContainer"),
  (boxes) => {
    if (!state.currentUniversal) return;
    state.currentUniversal.annotations = boxes;
    renderGtBoxList();
    pushHistory();
  },
  ({ boxId, currentLabel }) => {
    if (!els.labelModal || !els.newBoxLabelInput) return;
    state.pendingNewBoxId = String(boxId || "").trim();
    els.labelModal.hidden = false;
    els.labelModalBoxId.textContent = `Box: ${state.pendingNewBoxId}`;
    els.newBoxLabelInput.value = String(currentLabel || "");
    els.newBoxLabelInput.focus();
    els.newBoxLabelInput.select();
  },
  async ({ x, y }) => {
    if (!state.currentUniversal || state.drawMode) return;
    if (!state.samPointSelectionArmed) return;
    state.samPoints.push({ x, y });
    setAiStatus("sam_point_added", { count: state.samPoints.length, last_point: { x, y } });
    els.recordInfo.textContent = `SAM point captured (${state.samPoints.length}). Click Generate BBoxes to run SAM.`;
  }
);

function closeLabelModal() {
  if (!els.labelModal) return;
  els.labelModal.hidden = true;
  state.pendingNewBoxId = "";
}

function getGeminiApiKey() {
  const live = String(els.geminiApiKey?.value || "").trim();
  if (live) return live;
  return String(state.sessionGeminiApiKey || "").trim();
}

function stashGeminiApiKeyFromInput() {
  const key = String(els.geminiApiKey?.value || "").trim();
  if (!key) return;
  state.sessionGeminiApiKey = key;
  if (els.geminiApiKey) {
    els.geminiApiKey.value = "";
    els.geminiApiKey.placeholder = "Key saved for this session";
  }
}

function addBoxToCanvas(box, metaSource = "added") {
  if (!state.currentUniversal) return null;
  if (!box || !Array.isArray(box.bbox) || box.bbox.length < 4) return null;
  const [bx, by, bw, bh] = box.bbox.map((n) => Number(n) || 0);
  const current = state.currentUniversal.annotations || [];
  const usedIds = new Set(current.map((b) => String(b?.id || "").trim().toLowerCase()));
  let id = String(box.id || `sam_${Date.now()}`).trim();

  const nextAddedId = () => {
    let i = 1;
    while (usedIds.has(`a_${i}`)) i += 1;
    return `a_${i}`;
  };

  if (metaSource === "added") {
    id = nextAddedId();
  } else if (usedIds.has(id.toLowerCase())) {
    return id;
  }

  const next = {
    id,
    bbox: [bx, by, bw, bh],
    label: String(box.label || ""),
    meta: { ...(box.meta || {}), source: metaSource, kind: "bbox" }
  };
  state.currentUniversal.annotations = [...current, next];
  engine.setBoxes(state.currentUniversal.annotations);
  renderGtBoxList();
  pushHistory();
  return id;
}

function normalizeRawBox(item) {
  if (!item || typeof item !== "object") return null;
  const id = item.id || item.bbox_id || item.ann_id;
  if (!id) return null;

  const x = Number(item.x);
  const y = Number(item.y);
  const w = Number(item.w);
  const h = Number(item.h);
  if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(w) || !Number.isFinite(h)) return null;

  return {
    id: String(id).trim(),
    bbox: [x, y, w, h],
    label: String(item.label || ""),
    meta: {
      source: item.source || item.meta?.source || "gt",
      kind: item.kind || item.meta?.kind || ""
    }
  };
}

function normalizePredictedBox(item) {
  const normalized = normalizeRawBox(item);
  if (!normalized) return null;
  normalized.meta = { ...(normalized.meta || {}), source: "pred" };
  return normalized;
}

function boxIdKey(id) {
  return String(id || "").trim().toLowerCase();
}

function collectGtCatalog(rawRecord, universal) {
  const candidates = [...(rawRecord?.bbox || rawRecord?.boxes || []), ...(rawRecord?.predicted_boxes || [])];
  const map = new Map();
  const sourcePriority = { pred: 3, gt: 2, added: 1, new: 1 };

  if (Array.isArray(candidates) && candidates.length) {
    candidates.forEach((item) => {
      const source = String(item?.source || "").toLowerCase();
      const normalized = normalizeRawBox(item);
      if (!normalized) return;
      const key = boxIdKey(normalized.id);
      if (!key) return;
      const existing = map.get(key);
      if (!existing) {
        map.set(key, normalized);
        return;
      }
      const curScore = sourcePriority[String(normalized.meta?.source || "").toLowerCase()] ?? -1;
      const oldScore = sourcePriority[String(existing.meta?.source || "").toLowerCase()] ?? -1;
      if (curScore > oldScore) map.set(key, normalized);
    });
  }

  // Always merge currently visible annotations so newly drawn boxes appear in the checklist.
  (universal?.annotations || []).forEach((ann) => {
    if (!ann?.id || !Array.isArray(ann?.bbox)) return;
    const normalized = normalizeRawBox({
      id: ann.id,
      x: ann.bbox[0],
      y: ann.bbox[1],
      w: ann.bbox[2],
      h: ann.bbox[3],
      label: ann.label || "",
      source: ann.meta?.source || "new",
      kind: ann.meta?.kind || ""
    });
    if (!normalized) return;
    const key = boxIdKey(normalized.id);
    if (!key) return;
    const existing = map.get(key);
    if (!existing) {
      map.set(key, normalized);
      return;
    }
    const curScore = sourcePriority[String(normalized.meta?.source || "").toLowerCase()] ?? -1;
    const oldScore = sourcePriority[String(existing.meta?.source || "").toLowerCase()] ?? -1;
    if (curScore > oldScore) map.set(key, normalized);
  });

  return map;
}

function renderGtBoxList() {
  if (!els.gtBoxList) return;
  if (!state.currentUniversal) {
    els.gtBoxList.textContent = "No GT boxes loaded.";
    return;
  }

  const gtCatalog = collectGtCatalog(state.records[state.currentIndex], state.currentUniversal);
  if (!gtCatalog.size) {
    els.gtBoxList.textContent = "No GT boxes loaded.";
    return;
  }

  const selected = new Set((state.currentUniversal.annotations || []).map((a) => boxIdKey(a.id)));
  els.gtBoxList.innerHTML = "";

  Array.from(gtCatalog.entries())
    .sort((a, b) => String(a[1]?.id || "").localeCompare(String(b[1]?.id || "")))
    .forEach(([idKey, gtEntry]) => {
      const row = document.createElement("label");
      row.className = "gt-box-item";

      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = selected.has(idKey);
      const source = String(gtEntry?.meta?.source || "").toLowerCase();
      if (source === "pred") cb.classList.add("bbox-source-pred");
      else if (source === "new" || source === "added") cb.classList.add("bbox-source-added");
      else cb.classList.add("bbox-source-gt");
      cb.addEventListener("change", () => {
        if (!state.currentUniversal) return;
        const current = state.currentUniversal.annotations || [];
        if (cb.checked) {
          if (current.some((a) => boxIdKey(a.id) === idKey)) return;
          const gt = gtCatalog.get(idKey);
          if (!gt) return;
          state.currentUniversal.annotations = [...current, { ...gt, bbox: [...gt.bbox], meta: { ...(gt.meta || {}) } }];
        } else {
          state.currentUniversal.annotations = current.filter((a) => boxIdKey(a.id) !== idKey);
        }
        engine.setBoxes(state.currentUniversal.annotations);
        pushHistory();
        updateAiAssistButtons();
      });

      const text = document.createElement("span");
      const displayId = String(gtEntry?.id || idKey).trim();
      const displayLabel = String(gtEntry?.label || "").trim();
      text.textContent = displayLabel ? `${displayId} | ${displayLabel}` : displayId;
      row.appendChild(cb);
      row.appendChild(text);
      els.gtBoxList.appendChild(row);
    });
}

function setButtonsEnabled(loaded) {
  // Keep prev/next enabled while loaded; handlers manage cross-image navigation.
  els.prevBtn.disabled = !loaded;
  els.nextBtn.disabled = !loaded;

  [els.qaActionBtn, els.bboxActionBtn, els.samSelectPointsBtn, els.samGenerateBtn, els.exportAllBtn, els.drawModeBtn, els.undoBtn, els.deleteBoxBtn, els.clearBoxesBtn].forEach((btn) => {
    if (btn) btn.disabled = !loaded;
  });
  updateAiAssistButtons();
  updateUndoButton();
}

function normalizeRecords(rawData) {
  if (Array.isArray(rawData)) return rawData;
  if (Array.isArray(rawData.data)) return rawData.data;
  if (Array.isArray(rawData.items)) return rawData.items;
  return [rawData];
}

function updateRecordInfo() {
  if (!state.currentUniversal) {
    els.recordInfo.textContent = "No record loaded";
    return;
  }

  els.recordInfo.textContent = `Question ${state.currentIndex + 1}/${state.records.length}`;
}

function setAiStatus(status, detail = {}) {
  const payload = {
    time: new Date().toISOString(),
    status,
    ...detail
  };
  els.aiStatusView.textContent = JSON.stringify(payload, null, 2);
}

function deepClone(value) {
  return JSON.parse(JSON.stringify(value));
}

function buildSnapshot() {
  return {
    qa: readQaFields(els),
    annotations: engine.getBoxes()
  };
}

function hasMissingBoxes() {
  const boxes = state.currentUniversal?.annotations || [];
  return boxes.length === 0;
}

function hasMissingQa() {
  const qa = state.currentUniversal?.qa || {};
  return !String(qa.question || "").trim() || !String(qa.answer || "").trim();
}

function updateAiAssistButtons() {
  // static labels in HTML
  if (els.qaActionBtn) {
    els.qaActionBtn.textContent = "GENERATE Q&A";
  }
}

function updateUndoButton() {
  els.undoBtn.disabled = !state.currentUniversal || state.historyIndex <= 0;
}

function pushHistory() {
  if (!state.currentUniversal || state.suppressHistory) return;

  const snapshot = buildSnapshot();
  const previous = state.history[state.historyIndex];
  const sameAsPrevious = previous && JSON.stringify(previous) === JSON.stringify(snapshot);
  if (sameAsPrevious) return;

  state.history = state.history.slice(0, state.historyIndex + 1);
  state.history.push(snapshot);
  if (state.history.length > 100) state.history.shift();
  state.historyIndex = state.history.length - 1;
  updateUndoButton();
}

function resetHistory() {
  state.history = [];
  state.historyIndex = -1;
  pushHistory();
}

function applySnapshot(snapshot) {
  if (!state.currentUniversal || !snapshot) return;
  state.suppressHistory = true;
  state.currentUniversal.qa = deepClone(snapshot.qa);
  state.currentUniversal.annotations = deepClone(snapshot.annotations);
  populateQaFields(state.currentUniversal, els);
  engine.setBoxes(state.currentUniversal.annotations);
  state.suppressHistory = false;
  updateAiAssistButtons();
}

function normalizePath(path) {
  return String(path || "")
    .replace(/\\/g, "/")
    .replace(/^\.\/+/, "")
    .trim();
}

function resolveImageSource(imagePath) {
  if (!imagePath) return "";

  const normalized = normalizePath(imagePath);
  if (!normalized) return "";

  if (/^(blob:|data:|https?:\/\/)/i.test(normalized)) return normalized;
  return `/api/image?path=${encodeURIComponent(normalized)}`;
}

async function renderCurrentRecord() {
  const raw = state.records[state.currentIndex];
  if (!raw) return;

  const datasetType = detectDatasetType(raw);
  const universal = normalizeToUniversal(raw, datasetType);
  const predicted = Array.isArray(raw.predicted_boxes) ? raw.predicted_boxes.map(normalizePredictedBox).filter(Boolean) : [];
  // Prediction-first view: show only live Gemini-predicted boxes on the image canvas.
  // Ground-truth remains available in the right-side checkbox list.
  universal.annotations = predicted;
  state.currentUniversal = universal;
  state.samPoints = [];
  state.samPointSelectionArmed = false;
  engine.setPointPickMode(false);
  if (els.samSelectPointsBtn) {
    els.samSelectPointsBtn.classList.remove("is-active");
  }
  setAiStatus("idle", { message: "AI assist ready for this record." });

  setRawJsonView(els.rawJsonView, raw);
  populateQaFields(universal, els);
  renderGtBoxList();
  updateRecordInfo();

  const source = resolveImageSource(universal.image);
  if (!source) {
    els.canvasHint.textContent = "No image provided. Upload an image file.";
    els.canvasHint.hidden = false;
    canvasContainer.classList.remove("has-image");
    engine.clearCanvas();
    engine.setBoxes(universal.annotations || []);
    resetHistory();
    setButtonsEnabled(true);
    return;
  }

  try {
    const img = await loadImageFromSource(source);
    els.canvasHint.hidden = true;
    canvasContainer.classList.add("has-image");
    engine.setImage(img);
    engine.setBoxes(universal.annotations || []);
  } catch (err) {
    els.canvasHint.textContent = `${err.message}. Check if the image path in JSON is valid on local disk.`;
    els.canvasHint.hidden = false;
    canvasContainer.classList.remove("has-image");
    engine.clearCanvas();
    engine.setBoxes(universal.annotations || []);
  }

  renderGtBoxList();
  resetHistory();
  setButtonsEnabled(true);
}

async function attachPredictionsToRecords(records) {
  const enriched = [];
  for (const rec of records) {
    const gtBoxes = rec.bbox || rec.boxes || [];
    try {
      const predResp = await postJson("/api/predict-boxes", {
        image_path: rec.image_path || rec.image || "",
        annotation_path: rec.annotation_path || "",
        q_id: rec.q_id || "",
        question_text: rec.question_text || rec.question || "",
        answer_text: rec.ground_truth_answer || rec.answer_text || rec.answer || "",
        gt_boxes: gtBoxes
      });
      enriched.push({ ...rec, predicted_boxes: Array.isArray(predResp.boxes) ? predResp.boxes : [] });
    } catch (_) {
      enriched.push({ ...rec, predicted_boxes: [] });
    }
  }
  return enriched;
}

async function readFileAsDataUrl(file) {
  return await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(new Error("Failed to read image file"));
    reader.readAsDataURL(file);
  });
}

async function readFileAsText(file) {
  return await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ""));
    reader.onerror = () => reject(new Error("Failed to read JSON file"));
    reader.readAsText(file);
  });
}

function isJsonUpload(file) {
  if (!file) return false;
  const name = String(file.name || "").toLowerCase();
  const type = String(file.type || "").toLowerCase();
  return name.endsWith(".json") || type.includes("application/json");
}

function normalizeJsonUploadRecords(parsed) {
  const sourceItems = Array.isArray(parsed)
    ? parsed
    : Array.isArray(parsed?.records)
      ? parsed.records
      : parsed && typeof parsed === "object"
        ? [parsed]
        : [];

  const out = [];
  sourceItems.forEach((item) => {
    if (!item || typeof item !== "object") return;

    const imageLevel = {
      image_uid: item.image_uid || "",
      image_id: item.image_id || "",
      image_path: item.image_path || item.image || "",
      annotation_path: item.annotation_path || "",
      question_path: item.question_path || "",
      bbox: Array.isArray(item.bbox) ? item.bbox : [],
      predicted_boxes: Array.isArray(item.predicted_boxes) ? item.predicted_boxes : [],
      dataset_type: item.dataset_type || item.dataset_name || ""
    };

    if (Array.isArray(item.questions) && item.questions.length) {
      item.questions.forEach((q) => {
        if (!q || typeof q !== "object") return;
        const answerText = q.answer_text ?? q.answer ?? "";
        const correctAnswer = q.correct_answer ?? answerText;
        const gtAnswer = q.ground_truth_answer ?? correctAnswer;
        out.push({
          ...imageLevel,
          question_uid: q.question_uid || "",
          q_id: q.q_id || q.question_uid || "",
          question_text: q.question_text || q.question || "",
          choices: Array.isArray(q.choices) ? q.choices : [],
          answer_text: String(answerText),
          correct_answer: String(correctAnswer),
          ground_truth_answer: String(gtAnswer)
        });
      });
      return;
    }

    // Backward-compatible flat record path.
    out.push({
      ...imageLevel,
      q_id: item.q_id || "",
      question_text: item.question_text || item.question || "",
      choices: Array.isArray(item.choices) ? item.choices : [],
      answer_text: String(item.answer_text ?? item.answer ?? ""),
      correct_answer: String(item.correct_answer ?? item.answer_text ?? item.answer ?? ""),
      ground_truth_answer: String(item.ground_truth_answer ?? item.correct_answer ?? item.answer_text ?? item.answer ?? "")
    });
  });

  return out;
}

async function buildCompressedGenerateImageDataUrl() {
  if (!state.currentUniversal?.image) return "";
  const source = resolveImageSource(state.currentUniversal.image);
  if (!source) return "";

  const img = await loadImageFromSource(source);
  const scale = Math.min(1, MAX_GENERATE_IMAGE_DIM / Math.max(img.width, img.height));
  const width = Math.max(1, Math.round(img.width * scale));
  const height = Math.max(1, Math.round(img.height * scale));

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";
  ctx.drawImage(img, 0, 0, width, height);

  // Try PNG first. If too large, progressively compress as JPEG.
  let dataUrl = canvas.toDataURL("image/png");
  if (dataUrl.length <= MAX_BASE64_CHARS) return dataUrl;

  const qualities = [0.9, 0.8, 0.7, 0.6];
  for (const q of qualities) {
    dataUrl = canvas.toDataURL("image/jpeg", q);
    if (dataUrl.length <= MAX_BASE64_CHARS) return dataUrl;
  }
  return dataUrl;
}

async function postJson(url, payload, opts = {}) {
  const timeoutMs = Number(opts.timeoutMs || 0);
  setAiStatus("request_started", { endpoint: url });
  const controller = timeoutMs > 0 ? new AbortController() : null;
  let timer = null;
  if (controller) {
    timer = setTimeout(() => controller.abort(), timeoutMs);
  }
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller?.signal
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      setAiStatus("request_failed", { endpoint: url, error: data.error || `HTTP ${res.status}`, response: data });
      throw new Error(data.error || `Request failed (${res.status})`);
    }
    setAiStatus("request_succeeded", { endpoint: url, model: data.model || "", response_summary: Object.keys(data) });
    return data;
  } catch (err) {
    if (err?.name === "AbortError") {
      const msg = `Request timed out (${timeoutMs}ms)`;
      setAiStatus("request_failed", { endpoint: url, error: msg });
      throw new Error(msg);
    }
    throw err;
  } finally {
    if (timer) clearTimeout(timer);
  }
}

async function loadHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    state.health = data.capabilities || null;
    setAiStatus("health_loaded", { capabilities: state.health || {} });
  } catch (err) {
    setAiStatus("health_failed", { error: err.message });
  }
}

els.imageUpload.addEventListener("change", async (e) => {
  const file = e.target.files?.[0];
  if (!file) return;

  const prevInfo = els.recordInfo.textContent;
  try {
    if (isJsonUpload(file)) {
      const rawText = await readFileAsText(file);
      let parsed;
      try {
        parsed = JSON.parse(rawText);
      } catch (err) {
        throw new Error("Invalid JSON file");
      }

      let records = normalizeJsonUploadRecords(parsed);

      if (!records.length) {
        state.rawData = null;
        state.records = [];
        setButtonsEnabled(false);
        els.recordInfo.textContent = "JSON upload has no records.";
        return;
      }

      // If records contain GT but not predictions, attach live predictions.
      const hasPredicted = records.some((r) => Array.isArray(r?.predicted_boxes) && r.predicted_boxes.length > 0);
      if (!hasPredicted) {
        els.recordInfo.textContent = "Loaded JSON. Running live Gemini box prediction...";
        records = await attachPredictionsToRecords(records);
      }

      state.rawData = records;
      state.records = normalizeRecords(records);
      state.currentIndex = 0;
      state.uploadedImageDataUrl = "";
      state.uploadedImageName = file.name || "uploaded_json";
      await renderCurrentRecord();
      els.recordInfo.textContent = `Loaded ${state.records.length} record(s) from JSON.`;
      return;
    }

    state.uploadedImageDataUrl = await readFileAsDataUrl(file);
    state.uploadedImageName = file.name || "uploaded_image";
    els.recordInfo.textContent = "Matching uploaded image and loading GT + QA...";

    const response = await postJson("/api/records-by-upload-image", {
      filename: state.uploadedImageName,
      image_data_url: state.uploadedImageDataUrl
    });
    let records = Array.isArray(response.records) ? response.records : [];
    if (!records.length) {
      state.rawData = null;
      state.records = [];
      setButtonsEnabled(false);
      els.recordInfo.textContent = "No records found for uploaded image.";
      return;
    }
    els.recordInfo.textContent = "Loaded image. Running live Gemini box prediction...";
    records = await attachPredictionsToRecords(records);
    state.rawData = records;
    state.records = normalizeRecords(records);
    state.currentIndex = 0;
    await renderCurrentRecord();
    els.recordInfo.textContent = `Loaded ${records.length} record(s) for ${response.image_path || state.uploadedImageName}`;
  } catch (err) {
    state.rawData = null;
    state.records = [];
    state.uploadedImageDataUrl = "";
    state.uploadedImageName = "";
    setButtonsEnabled(false);
    els.recordInfo.textContent = `Image load failed: ${err.message}`;
  }

  setTimeout(() => {
    if (state.currentUniversal) updateRecordInfo();
    else if (!state.records.length) els.recordInfo.textContent = prevInfo;
    setButtonsEnabled(Boolean(state.currentUniversal));
  }, 600);
  if (els.imageUpload) {
    els.imageUpload.value = "";
  }
});

els.prevBtn.addEventListener("click", async () => {
  if (!state.records.length) return;

  if (state.currentIndex > 0) {
    state.currentIndex -= 1;
    await renderCurrentRecord();
    return;
  }

  const currentImagePath =
    String(state.currentUniversal?.image || state.records[state.currentIndex]?.image_path || state.records[0]?.image_path || "").trim();
  if (!currentImagePath) return;

  const prevInfo = els.recordInfo.textContent;
  els.prevBtn.disabled = true;
  els.recordInfo.textContent = "Loading previous image...";
  try {
    const response = await postJson("/api/previous-image-records", { image_path: currentImagePath });
    let records = Array.isArray(response.records) ? response.records : [];
    if (!records.length) {
      els.recordInfo.textContent = "No previous image records found.";
      return;
    }
    els.recordInfo.textContent = "Loaded previous image. Running live Gemini box prediction...";
    records = await attachPredictionsToRecords(records);
    state.rawData = records;
    state.records = normalizeRecords(records);
    state.currentIndex = Math.max(0, state.records.length - 1);
    await renderCurrentRecord();
    els.recordInfo.textContent = `Loaded previous image: ${response.image_path || ""}`;
  } catch (err) {
    els.recordInfo.textContent = `Previous image load failed: ${err.message}`;
  } finally {
    setTimeout(() => {
      if (state.currentUniversal) updateRecordInfo();
      else els.recordInfo.textContent = prevInfo;
      setButtonsEnabled(Boolean(state.currentUniversal));
    }, 900);
  }
});

els.nextBtn.addEventListener("click", async () => {
  if (!state.records.length) return;

  if (state.currentIndex < state.records.length - 1) {
    state.currentIndex += 1;
    await renderCurrentRecord();
    return;
  }

  const currentImagePath =
    String(state.currentUniversal?.image || state.records[state.currentIndex]?.image_path || state.records[0]?.image_path || "").trim();
  if (!currentImagePath) return;

  const prevInfo = els.recordInfo.textContent;
  els.nextBtn.disabled = true;
  els.recordInfo.textContent = "Loading next image...";
  try {
    const response = await postJson("/api/next-image-records", { image_path: currentImagePath });
    let records = Array.isArray(response.records) ? response.records : [];
    if (!records.length) {
      els.recordInfo.textContent = "No next image records found.";
      return;
    }
    els.recordInfo.textContent = "Loaded next image. Running live Gemini box prediction...";
    records = await attachPredictionsToRecords(records);
    state.rawData = records;
    state.records = normalizeRecords(records);
    state.currentIndex = 0;
    await renderCurrentRecord();
    els.recordInfo.textContent = `Loaded next image: ${response.image_path || ""}`;
  } catch (err) {
    els.recordInfo.textContent = `Next image load failed: ${err.message}`;
  } finally {
    setTimeout(() => {
      if (state.currentUniversal) updateRecordInfo();
      else els.recordInfo.textContent = prevInfo;
      setButtonsEnabled(Boolean(state.currentUniversal));
    }, 900);
  }
});

els.drawModeBtn.addEventListener("click", () => {
  state.drawMode = !state.drawMode;
  state.samPoints = [];
  state.samPointSelectionArmed = false;
  engine.setPointPickMode(false);
  if (els.samSelectPointsBtn) {
    els.samSelectPointsBtn.classList.remove("is-active");
  }
  engine.setDrawMode(state.drawMode);
  els.drawModeBtn.textContent = `Draw Box: ${state.drawMode ? "On" : "Off"}`;
});

if (els.samSelectPointsBtn) {
  els.samSelectPointsBtn.addEventListener("click", () => {
    if (!state.currentUniversal) return;
    if (state.samPointSelectionArmed) {
      state.samPointSelectionArmed = false;
      engine.setPointPickMode(false);
      els.samSelectPointsBtn.classList.remove("is-active");
      els.recordInfo.textContent = `SAM point selection stopped. ${state.samPoints.length} point(s) selected.`;
      setAiStatus("sam_point_mode_stopped", { points_selected: state.samPoints.length });
      return;
    }
    state.drawMode = false;
    engine.setDrawMode(false);
    els.drawModeBtn.textContent = "Draw Box: Off";
    state.samPointSelectionArmed = true;
    engine.setPointPickMode(true);
    els.samSelectPointsBtn.classList.add("is-active");
    setAiStatus("sam_point_mode_armed", { message: "Select one or more points, then click SAM Generate BBoxes." });
    els.recordInfo.textContent = "SAM point selection active: click as many points as needed.";
  });
}

if (els.samGenerateBtn) {
  els.samGenerateBtn.addEventListener("click", async () => {
    if (!state.currentUniversal) return;
    if (!state.samPoints.length) {
      setAiStatus("sam_generate_skipped", { reason: "no_points_selected" });
      els.recordInfo.textContent = "Select one or more data points first.";
      return;
    }

    const points = state.samPoints.map((p) => ({ x: Number(p.x), y: Number(p.y) })).filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
    if (!points.length) {
      setAiStatus("sam_generate_skipped", { reason: "invalid_points" });
      els.recordInfo.textContent = "Selected points are invalid. Re-select points.";
      return;
    }

    state.samPointSelectionArmed = false;
    engine.setPointPickMode(false);
    if (els.samSelectPointsBtn) {
      els.samSelectPointsBtn.classList.remove("is-active");
    }

    const prevInfo = els.recordInfo.textContent;
    els.samGenerateBtn.disabled = true;
    els.recordInfo.textContent = `Generating SAM bbox from ${points.length} point(s)...`;
    try {
      const response = await postJson("/api/sam-multi-point-box", {
        image_path: state.currentUniversal.image || "",
        points
      });
      const id = addBoxToCanvas(response?.box || null, "added");
      if (!id) {
        throw new Error("SAM did not return a valid bounding box.");
      }
      if (typeof engine.onNewBoxLabelRequest === "function") {
        engine.onNewBoxLabelRequest({ boxId: id, currentLabel: "" });
      }
      setAiStatus("sam_multi_point_box_added", {
        id,
        points_count: points.length,
        backend: response.backend || "",
        warning: response.warning || "",
        diagnostics: response.diagnostics || {}
      });
      els.recordInfo.textContent = `SAM bbox generated from ${points.length} point(s).`;
      state.samPoints = [];
    } catch (err) {
      setAiStatus("sam_multi_point_box_failed", { error: err.message || "SAM multi-point request failed.", points_count: points.length });
      els.recordInfo.textContent = `SAM bbox generation failed: ${err.message || "Unknown error"}`;
    } finally {
      setTimeout(() => {
        if (state.currentUniversal) updateRecordInfo();
        else els.recordInfo.textContent = prevInfo;
        setButtonsEnabled(Boolean(state.currentUniversal));
      }, 800);
    }
  });
}

els.deleteBoxBtn.addEventListener("click", () => {
  engine.deleteSelected();
});

els.clearBoxesBtn.addEventListener("click", () => {
  engine.clearBoxes();
});

els.undoBtn.addEventListener("click", () => {
  if (state.historyIndex <= 0) return;
  state.historyIndex -= 1;
  applySnapshot(state.history[state.historyIndex]);
  updateUndoButton();
});

els.exportAllBtn.addEventListener("click", () => {
  if (!state.currentUniversal) return;

  state.currentUniversal.qa = readQaFields(els);
  state.currentUniversal.dataset_type = String(els.datasetType?.value || "");
  state.currentUniversal.annotations = engine.getBoxes();

  const baseName = state.currentUniversal.metadata?.id || `record_${state.currentIndex + 1}`;
  exportJson(`${baseName}_annotated.json`, state.currentUniversal);
  exportDataUrl(`${baseName}_annotated.png`, els.canvas.toDataURL("image/png"));
});

els.qaActionBtn.addEventListener("click", async () => {
  if (!state.currentUniversal) return;

  const prevInfo = els.recordInfo.textContent;
  els.qaActionBtn.disabled = true;
  els.recordInfo.textContent = "Generating missing QA...";
  try {
    // Always use latest user-edited values from inputs.
    state.currentUniversal.qa = readQaFields(els);
    const imageDataUrl = await buildCompressedGenerateImageDataUrl();
    const response = await postJson("/api/generate-missing-qa", {
      image_path: state.currentUniversal.image,
      image_data_url: imageDataUrl,
      api_key: getGeminiApiKey(),
      dataset_type: state.currentUniversal.dataset_type,
      existing_qa: state.currentUniversal.qa
    });

    const qa = response.qa || {};
    state.currentUniversal.qa = {
      question: String(qa.question || ""),
      answer: String(qa.answer || ""),
      choices: Array.isArray(qa.choices) ? qa.choices : []
    };
    populateQaFields(state.currentUniversal, els);
    pushHistory();
    if (qa.status === "unchanged") {
      setAiStatus("qa_unchanged", { model: response.model || "", warning: response.warning || "", question_len: state.currentUniversal.qa.question.length, answer_len: state.currentUniversal.qa.answer.length });
      els.recordInfo.textContent = "QA already present. No changes made.";
    } else {
      setAiStatus("qa_inserted", { model: response.model || "", warning: response.warning || "", question_len: state.currentUniversal.qa.question.length, answer_len: state.currentUniversal.qa.answer.length });
      els.recordInfo.textContent = "Inserted generated missing QA.";
    }
  } catch (err) {
    setAiStatus("qa_failed", { error: err.message });
    els.recordInfo.textContent = `QA generation failed: ${err.message}`;
  } finally {
    setTimeout(() => {
      if (state.currentUniversal) updateRecordInfo();
      else els.recordInfo.textContent = prevInfo;
      updateAiAssistButtons();
      setButtonsEnabled(Boolean(state.currentUniversal));
    }, 1200);
  }
});

els.bboxActionBtn.addEventListener("click", async () => {
  if (!state.currentUniversal) return;

  // Keep QA in sync with latest edits before bbox generation prompt.
  state.currentUniversal.qa = readQaFields(els);
  const currentBoxes = state.currentUniversal.annotations || [];
  if (currentBoxes.length > 0) {
    setAiStatus("bbox_generate_skipped", { reason: "bounding boxes already present", count: currentBoxes.length });
    els.recordInfo.textContent = "Bounding boxes already present. Skipped.";
    return;
  }

  const prevInfo = els.recordInfo.textContent;
  els.bboxActionBtn.disabled = true;
  els.recordInfo.textContent = "Generating missing bounding boxes...";
  try {
    const imageDataUrl = await buildCompressedGenerateImageDataUrl();
    const gtCatalog = Array.from(collectGtCatalog(state.records[state.currentIndex], state.currentUniversal).values()).map((b) => ({
      id: b.id,
      x: b.bbox?.[0],
      y: b.bbox?.[1],
      w: b.bbox?.[2],
      h: b.bbox?.[3],
      label: b.label || "",
      source: b.meta?.source || "gt",
      kind: b.meta?.kind || ""
    }));

    const response = await postJson("/api/generate-missing-bboxes", {
      image_path: state.currentUniversal.image,
      image_data_url: imageDataUrl,
      api_key: getGeminiApiKey(),
      q_id: String(state.currentUniversal?.metadata?.id || ""),
      question_text: String(state.currentUniversal?.qa?.question || ""),
      answer_text: String(state.currentUniversal?.qa?.answer || ""),
      choices: Array.isArray(state.currentUniversal?.qa?.choices) ? state.currentUniversal.qa.choices : [],
      gt_boxes: gtCatalog
    });

    const predicted = Array.isArray(response.boxes) ? response.boxes : [];
    if (!predicted.length) {
      setAiStatus("bbox_generate_empty", {
        model: response.model || "",
        provider: response.provider || "",
        status_detail: response.status || "",
        warning: response.warning || "",
        raw_preview: typeof response.raw_text === "string" ? response.raw_text.slice(0, 500) : ""
      });
      els.recordInfo.textContent = "No bounding boxes predicted.";
      return;
    }

    const normalized = predicted.map((b) => normalizePredictedBox(b)).filter(Boolean);
    state.currentUniversal.annotations = normalized;
    engine.setBoxes(state.currentUniversal.annotations);
    renderGtBoxList();
    pushHistory();
    setAiStatus("bbox_generated", { count: normalized.length, model: response.model || "", provider: response.provider || "" });
    els.recordInfo.textContent = `Inserted ${normalized.length} predicted bounding box(es).`;
  } catch (err) {
    setAiStatus("bbox_generate_failed", { error: err.message });
    els.recordInfo.textContent = `Bounding box generation failed: ${err.message}`;
  } finally {
    setTimeout(() => {
      if (state.currentUniversal) updateRecordInfo();
      else els.recordInfo.textContent = prevInfo;
      updateAiAssistButtons();
      setButtonsEnabled(Boolean(state.currentUniversal));
    }, 1200);
  }
});

loadHealth();

if (els.geminiApiKey) {
  els.geminiApiKey.addEventListener("change", () => {
    stashGeminiApiKeyFromInput();
  });
  els.geminiApiKey.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      stashGeminiApiKeyFromInput();
    }
  });
  els.geminiApiKey.addEventListener("blur", () => {
    stashGeminiApiKeyFromInput();
  });
}

if (els.newBoxLabelSaveBtn) {
  els.newBoxLabelSaveBtn.addEventListener("click", () => {
    const id = String(state.pendingNewBoxId || "").trim();
    if (!id) {
      closeLabelModal();
      return;
    }
    const nextLabel = String(els.newBoxLabelInput?.value || "").trim();
    engine.setBoxLabel(id, nextLabel);
    if (state.currentUniversal) {
      state.currentUniversal.annotations = engine.getBoxes();
      renderGtBoxList();
      pushHistory();
    }
    closeLabelModal();
  });
}

if (els.newBoxLabelCancelBtn) {
  els.newBoxLabelCancelBtn.addEventListener("click", () => {
    closeLabelModal();
  });
}

if (els.newBoxLabelInput) {
  els.newBoxLabelInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      els.newBoxLabelSaveBtn?.click();
      return;
    }
    if (e.key === "Escape") {
      e.preventDefault();
      closeLabelModal();
    }
  });
}
