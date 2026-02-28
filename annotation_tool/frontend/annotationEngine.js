function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

export class AnnotationEngine {
  constructor(canvas, container, onChange, onNewBoxLabelRequest = null, onPointRequest = null) {
    this.canvas = canvas;
    this.container = container;
    this.ctx = canvas.getContext("2d");
    this.onChange = onChange;
    this.onNewBoxLabelRequest = onNewBoxLabelRequest;
    this.onPointRequest = onPointRequest;

    this.image = null;
    this.maskOverlay = null;
    this.boxes = [];
    this.selectedId = null;
    this.drawMode = false;
    this.pointPickMode = false;

    this.dragState = null;
    this.hoverHandle = null;

    this.bindEvents();
  }

  bindEvents() {
    if (window.PointerEvent) {
      this.canvas.addEventListener("pointerdown", (e) => {
        e.preventDefault();
        if (typeof this.canvas.setPointerCapture === "function") {
          try {
            this.canvas.setPointerCapture(e.pointerId);
          } catch (_) {}
        }
        this.onMouseDown(e);
      });
      this.canvas.addEventListener("pointermove", (e) => {
        e.preventDefault();
        this.onMouseMove(e);
      });
      this.canvas.addEventListener("pointerup", (e) => {
        e.preventDefault();
        if (typeof this.canvas.releasePointerCapture === "function") {
          try {
            this.canvas.releasePointerCapture(e.pointerId);
          } catch (_) {}
        }
        this.onMouseUp();
      });
      this.canvas.addEventListener("pointercancel", (e) => {
        e.preventDefault();
        if (typeof this.canvas.releasePointerCapture === "function") {
          try {
            this.canvas.releasePointerCapture(e.pointerId);
          } catch (_) {}
        }
        this.onMouseUp();
      });
      return;
    }

    // Fallback for older browsers without PointerEvent.
    this.canvas.addEventListener("mousedown", (e) => this.onMouseDown(e));
    this.canvas.addEventListener("mousemove", (e) => this.onMouseMove(e));
    this.canvas.addEventListener("mouseup", () => this.onMouseUp());
    this.canvas.addEventListener("mouseleave", () => this.onMouseUp());
    this.canvas.addEventListener(
      "touchstart",
      (e) => {
        e.preventDefault();
        this.onMouseDown(e);
      },
      { passive: false }
    );
    this.canvas.addEventListener(
      "touchmove",
      (e) => {
        e.preventDefault();
        this.onMouseMove(e);
      },
      { passive: false }
    );
    this.canvas.addEventListener(
      "touchend",
      (e) => {
        e.preventDefault();
        this.onMouseUp();
      },
      { passive: false }
    );
    this.canvas.addEventListener(
      "touchcancel",
      (e) => {
        e.preventDefault();
        this.onMouseUp();
      },
      { passive: false }
    );
  }

  setDrawMode(enabled) {
    this.drawMode = enabled;
  }

  setPointPickMode(enabled) {
    this.pointPickMode = Boolean(enabled);
    if (this.pointPickMode) {
      this.selectedId = null;
    }
    this.render();
  }

  setImage(img) {
    this.image = img;
    this.canvas.width = img.width;
    this.canvas.height = img.height;
    this.render();
  }

  clearCanvas(width = 960, height = 540) {
    this.image = null;
    this.maskOverlay = null;
    this.canvas.width = width;
    this.canvas.height = height;
    this.render();
  }

  setMaskOverlay(dataUrl) {
    const src = String(dataUrl || "").trim();
    if (!src) {
      this.maskOverlay = null;
      this.render();
      return;
    }
    const img = new Image();
    img.onload = () => {
      this.maskOverlay = img;
      this.render();
    };
    img.onerror = () => {
      this.maskOverlay = null;
      this.render();
    };
    img.src = src;
  }

  clearMaskOverlay() {
    this.maskOverlay = null;
    this.render();
  }

  setBoxes(annotations) {
    const usedAIds = new Set();
    let aCounter = 0;

    const allocateAId = () => {
      do {
        aCounter += 1;
      } while (usedAIds.has(`a_${aCounter}`));
      const next = `a_${aCounter}`;
      usedAIds.add(next);
      return next;
    };

    this.boxes = (annotations || []).map((a, idx) => {
      let safeId = String(a?.id || a?.bbox_id || a?.ann_id || a?.meta?.id || `bbox_${idx + 1}`).trim();
      const safeLabel = String(a?.label || a?.meta?.label || "").trim();
      const source = String(a?.meta?.source || "").toLowerCase();

      const aMatch = safeId.match(/^a_(\d+)$/i);
      if (aMatch) {
        usedAIds.add(`a_${Number(aMatch[1])}`);
      } else if (source === "new") {
        // Normalize legacy drawn-box IDs (e.g., ann_*) to a_N.
        safeId = allocateAId();
      }

      return {
        ...a,
        id: safeId,
        label: safeLabel,
        bbox: Array.isArray(a?.bbox) ? [...a.bbox] : [0, 0, 1, 1]
      };
    });
    this.selectedId = null;
    this.render();
  }

  getBoxes() {
    return this.boxes.map((b) => ({ ...b, bbox: [...b.bbox] }));
  }

  setBoxLabel(boxId, label) {
    const targetId = String(boxId || "").trim();
    const box = this.boxes.find((b) => String(b.id).trim() === targetId);
    if (!box) return false;
    box.label = String(label || "").trim();
    this.render();
    this.emitChange();
    return true;
  }

  deleteSelected() {
    if (!this.selectedId) return;
    this.boxes = this.boxes.filter((b) => b.id !== this.selectedId);
    this.selectedId = null;
    this.emitChange();
    this.render();
  }

  clearBoxes() {
    this.boxes = [];
    this.selectedId = null;
    this.emitChange();
    this.render();
  }

  toCanvasPoint(evt) {
    const pointer =
      (evt.touches && evt.touches[0]) ||
      (evt.changedTouches && evt.changedTouches[0]) ||
      evt;
    const rect = this.canvas.getBoundingClientRect();
    const scaleX = this.canvas.width / rect.width;
    const scaleY = this.canvas.height / rect.height;
    return {
      x: (pointer.clientX - rect.left) * scaleX,
      y: (pointer.clientY - rect.top) * scaleY
    };
  }

  hitTestHandle(box, point) {
    const [x, y, w, h] = box.bbox;
    const handleSize = 8;
    const hx = x + w;
    const hy = y + h;
    return Math.abs(point.x - hx) <= handleSize && Math.abs(point.y - hy) <= handleSize;
  }

  hitTestBox(box, point) {
    const [x, y, w, h] = box.bbox;
    return point.x >= x && point.x <= x + w && point.y >= y && point.y <= y + h;
  }

  findTopHit(point) {
    for (let i = this.boxes.length - 1; i >= 0; i -= 1) {
      const box = this.boxes[i];
      if (this.hitTestHandle(box, point)) return { box, mode: "resize" };
      if (this.hitTestBox(box, point)) return { box, mode: "move" };
    }
    return null;
  }

  nextAnnotationId() {
    let maxIdx = 0;
    this.boxes.forEach((b) => {
      const m = String(b?.id || "").trim().match(/^a_(\d+)$/i);
      if (!m) return;
      const n = Number(m[1]);
      if (Number.isFinite(n) && n > maxIdx) maxIdx = n;
    });
    return `a_${maxIdx + 1}`;
  }

  onMouseDown(evt) {
    if (!this.image) return;
    const p = this.toCanvasPoint(evt);

    if (this.pointPickMode && !this.drawMode) {
      if (typeof this.onPointRequest === "function") {
        this.onPointRequest({ x: p.x, y: p.y });
      }
      this.selectedId = null;
      this.render();
      return;
    }

    if (this.drawMode) {
      const newId = this.nextAnnotationId();
      const box = { id: newId, bbox: [p.x, p.y, 1, 1], label: "", meta: { source: "new" } };
      this.boxes.push(box);
      this.selectedId = newId;
      this.dragState = { mode: "draw", start: p, boxId: newId };
      this.render();
      return;
    }

    const hit = this.findTopHit(p);
    if (!hit) {
      if (typeof this.onPointRequest === "function") {
        this.onPointRequest({ x: p.x, y: p.y });
      }
      this.selectedId = null;
      this.render();
      return;
    }

    this.selectedId = hit.box.id;
    const [x, y, w, h] = hit.box.bbox;
    this.dragState = {
      mode: hit.mode,
      boxId: hit.box.id,
      start: p,
      initial: [x, y, w, h]
    };
    this.render();
  }

  onMouseMove(evt) {
    if (!this.image) return;
    const p = this.toCanvasPoint(evt);

    if (!this.dragState) {
      const hit = this.findTopHit(p);
      this.canvas.style.cursor = this.drawMode || this.pointPickMode ? "crosshair" : hit?.mode === "resize" ? "nwse-resize" : hit?.mode === "move" ? "move" : "default";
      return;
    }

    const box = this.boxes.find((b) => b.id === this.dragState.boxId);
    if (!box) return;

    if (this.dragState.mode === "draw") {
      const sx = this.dragState.start.x;
      const sy = this.dragState.start.y;
      box.bbox[0] = Math.min(sx, p.x);
      box.bbox[1] = Math.min(sy, p.y);
      box.bbox[2] = Math.abs(p.x - sx);
      box.bbox[3] = Math.abs(p.y - sy);
    }

    if (this.dragState.mode === "move") {
      const dx = p.x - this.dragState.start.x;
      const dy = p.y - this.dragState.start.y;
      const [x, y, w, h] = this.dragState.initial;
      box.bbox[0] = clamp(x + dx, 0, Math.max(0, this.canvas.width - w));
      box.bbox[1] = clamp(y + dy, 0, Math.max(0, this.canvas.height - h));
    }

    if (this.dragState.mode === "resize") {
      const [x, y] = this.dragState.initial;
      box.bbox[2] = clamp(p.x - x, 1, this.canvas.width - x);
      box.bbox[3] = clamp(p.y - y, 1, this.canvas.height - y);
    }

    this.render();
  }

  onMouseUp() {
    if (!this.dragState) return;
    const finishedMode = this.dragState.mode;
    const finishedBoxId = this.dragState.boxId;
    this.dragState = null;

    if (finishedMode === "draw") {
      const drawn = this.boxes.find((b) => b.id === finishedBoxId);
      const isNew = String(drawn?.meta?.source || "").toLowerCase() === "new";
      if (drawn && isNew && typeof this.onNewBoxLabelRequest === "function") {
        this.onNewBoxLabelRequest({
          boxId: String(drawn.id || "").trim(),
          currentLabel: String(drawn.label || "").trim()
        });
      }
      this.render();
    }

    this.emitChange();
  }

  emitChange() {
    if (this.onChange) this.onChange(this.getBoxes());
  }

  render() {
    const { ctx, canvas, image } = this;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!image) return;

    ctx.drawImage(image, 0, 0, canvas.width, canvas.height);
    if (this.maskOverlay) {
      ctx.save();
      ctx.globalAlpha = 0.32;
      ctx.drawImage(this.maskOverlay, 0, 0, canvas.width, canvas.height);
      ctx.restore();
    }

    this.boxes.forEach((box) => {
      const [x, y, w, h] = box.bbox;
      const selected = box.id === this.selectedId;
      const source = String(box?.meta?.source || "").toLowerCase();
      const isPred = source === "pred";
      const isAdded = source === "new" || source === "added";

      ctx.lineWidth = selected ? 3 : 2;
      if (selected) {
        ctx.strokeStyle = "#99c24d";
      } else if (isAdded) {
        ctx.strokeStyle = "#6fbf73";
      } else if (isPred) {
        ctx.strokeStyle = "#f2d85a";
      } else {
        ctx.strokeStyle = "#d14d72";
      }
      ctx.strokeRect(x, y, w, h);

      if (selected) {
        ctx.fillStyle = "#99c24d";
        ctx.fillRect(x + w - 5, y + h - 5, 10, 10);
      }

      const idText = String(box.id || "").trim();
      const rawLabel = String(box.label || "").trim();
      const textLabel = rawLabel && rawLabel.toLowerCase() !== idText.toLowerCase() ? rawLabel : "";
      const baseText = idText;
      const labelText = textLabel ? `${baseText} | ${textLabel}` : baseText;
      if (labelText) {
        ctx.font = "bold 12px sans-serif";
        const tw = ctx.measureText(labelText).width;
        const badgeW = tw + 8;
        const tx = clamp(x + w - badgeW - 2, 2, Math.max(2, canvas.width - badgeW - 2));
        const ty = clamp(y + 14, 14, Math.max(14, canvas.height - 2));
        ctx.fillStyle = "rgba(8, 12, 18, 0.92)";
        ctx.fillRect(tx, ty - 12, badgeW, 16);
        ctx.strokeStyle = "rgba(255, 255, 255, 0.8)";
        ctx.lineWidth = 1;
        ctx.strokeRect(tx, ty - 12, badgeW, 16);
        ctx.fillStyle = "#ffffff";
        ctx.fillText(labelText, tx + 4, ty);
      }
    });
  }
}
