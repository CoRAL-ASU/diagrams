import { detectDatasetType } from "./datasetDetector.js";

function asArray(value) {
  if (!value) return [];
  return Array.isArray(value) ? value : [value];
}

function normalizeBBox(raw) {
  if (!raw) return null;

  if (Array.isArray(raw) && raw.length >= 4) {
    return [Number(raw[0]) || 0, Number(raw[1]) || 0, Number(raw[2]) || 0, Number(raw[3]) || 0];
  }

  const keys = ["x", "y", "w", "h"];
  if (keys.every((k) => raw[k] !== undefined)) {
    return [Number(raw.x) || 0, Number(raw.y) || 0, Number(raw.w) || 0, Number(raw.h) || 0];
  }

  const alt = ["left", "top", "width", "height"];
  if (alt.every((k) => raw[k] !== undefined)) {
    return [Number(raw.left) || 0, Number(raw.top) || 0, Number(raw.width) || 0, Number(raw.height) || 0];
  }

  return null;
}

function parseBBoxText(text) {
  if (typeof text !== "string") return [];
  const matches = text.match(/\[\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\]/g) || [];
  return matches.map((m, i) => ({
    id: `box_${i + 1}`,
    bbox: JSON.parse(m).map((n) => Number(n) || 0),
    label: "",
    meta: { source: "answer_text" }
  }));
}

function collectAnnotations(record) {
  const explicit = [];
  const candidates = [record.annotations, record.bounding_boxes, record.bboxes, record.boxes, record.bbox];

  candidates.forEach((arr) => {
    asArray(arr).forEach((item, idx) => {
      if (!item) return;
      const bbox = normalizeBBox(item.bbox || item);
      if (!bbox) return;
      explicit.push({
        id: item.id || item.ann_id || `ann_${explicit.length + idx + 1}`,
        bbox,
        label: item.label || item.category || "",
        meta: item.meta || {}
      });
    });
  });

  if (explicit.length > 0) return explicit;

  return parseBBoxText(record.answer || record.annotation || "");
}

function readImagePath(record) {
  return (
    record.image ||
    record.image_path ||
    record.img_path ||
    record.file_name ||
    record.filename ||
    ""
  );
}

function normalizeGeneric(record, datasetType) {
  const answerValue =
    typeof record.answers === "object" && record.answers !== null
      ? record.answers.correct || record.answers.answer || ""
      : record.ground_truth_answer || record.gt_answer || record.answer || record.response || record.predicted_answer || "";

  return {
    dataset_type: datasetType,
    image: readImagePath(record),
    qa: {
      question: record.question || record.question_text || record.query || "",
      answer: answerValue,
      choices: asArray(record.choices || record.options || []).map((c) => String(c))
    },
    annotations: collectAnnotations(record),
    metadata: {
      id: record.id || record.q_id || record.question_id || "",
      ground_truth_path: record.ground_truth_path || record.ground_truth || "",
      annotation_path: record.annotation_path || record.annotation || "",
      bounding_box_path: record.bounding_box_path || record.bbox_path || "",
      question_path: record.question_path || "",
      answers: typeof record.answers === "object" ? record.answers : {}
    }
  };
}

function normalizeRealCQA(record) {
  return {
    dataset_type: "RealCQA",
    image: readImagePath(record),
    qa: {
      question: record.question || "",
      answer: record.answer || "",
      choices: []
    },
    annotations: collectAnnotations(record),
    metadata: {
      id: record.q_id || "",
      explanation: record.explanation || ""
    }
  };
}

function normalizeMapWise(record) {
  return {
    dataset_type: "MapWise",
    image: readImagePath(record),
    qa: {
      question: record.question || "",
      answer: record.answer || "",
      choices: asArray(record.choices).map((c) => String(c))
    },
    annotations: collectAnnotations(record),
    metadata: {
      id: record.question_id || "",
      annotation_path: record.annotation || ""
    }
  };
}

function normalizeChartQALike(record) {
  return {
    dataset_type: "ChartQA-like",
    image: readImagePath(record),
    qa: {
      question: record.question_text || record.question || "",
      answer: record.gt_answer || record.answer || "",
      choices: asArray(record.choices || []).map((c) => String(c))
    },
    annotations: collectAnnotations(record),
    metadata: {
      id: record.q_id || record.id || "",
      annotation_path: record.annotation_path || "",
      question_path: record.question_path || "",
      predicted_answer_raw: record.predicted_answer_raw || "",
      predicted_explanation: record.predicted_explanation || ""
    }
  };
}

function normalizeCircuitVQALike(record) {
  return {
    dataset_type: "CircuitVQA-like",
    image: readImagePath(record),
    qa: {
      question: record.question_text || record.question || "",
      answer: record.ground_truth_answer || record.answer || "",
      choices: asArray(record.choices || []).map((c) => String(c))
    },
    annotations: collectAnnotations(record),
    metadata: {
      id: record.q_id || record.id || "",
      annotation_path: record.annotation_path || "",
      question_path: record.question_path || "",
      predicted_answer_raw: record.predicted_answer_raw || "",
      predicted_explanation: record.predicted_explanation || ""
    }
  };
}

function normalizeInfographicsVQALike(record) {
  return {
    dataset_type: "InfographicsVQA-like",
    image: readImagePath(record),
    qa: {
      question: record.question_text || record.question || "",
      answer: record.ground_truth_answer || record.answer || "",
      choices: asArray(record.choices || []).map((c) => String(c))
    },
    annotations: collectAnnotations(record),
    metadata: {
      id: record.q_id || record.id || "",
      annotation_path: record.annotation_path || "",
      question_path: record.question_path || "",
      predicted_answer_raw: record.predicted_answer_raw || "",
      predicted_explanation: record.predicted_explanation || ""
    }
  };
}

function normalizeMapIQLike(record) {
  return {
    dataset_type: "MapIQ-like",
    image: readImagePath(record),
    qa: {
      question: record.question_text || record.question || "",
      answer: record.ground_truth_answer || record.answer || "",
      choices: asArray(record.choices || []).map((c) => String(c))
    },
    annotations: collectAnnotations(record),
    metadata: {
      id: record.q_id || record.id || "",
      annotation_path: record.annotation_path || "",
      question_path: record.question_path || "",
      predicted_answer_raw: record.predicted_answer_raw || "",
      predicted_explanation: record.predicted_explanation || ""
    }
  };
}

function normalizeMapWiseLike(record) {
  return {
    dataset_type: "MapWise-like",
    image: readImagePath(record),
    qa: {
      question: record.question_text || record.question || "",
      answer: record.ground_truth_answer || record.answer || "",
      choices: asArray(record.choices || []).map((c) => String(c))
    },
    annotations: collectAnnotations(record),
    metadata: {
      id: record.q_id || record.id || "",
      annotation_path: record.annotation_path || "",
      question_path: record.question_path || "",
      predicted_answer_raw: record.predicted_answer_raw || "",
      predicted_explanation: record.predicted_explanation || ""
    }
  };
}

function normalizeCocoLike(record) {
  const annotations = asArray(record.annotations).map((ann, i) => ({
    id: ann.id || `coco_ann_${i + 1}`,
    bbox: normalizeBBox(ann.bbox) || [0, 0, 0, 0],
    label: String(ann.category_id || ann.label || ""),
    meta: {}
  }));

  return {
    dataset_type: "COCO-like",
    image: readImagePath(record),
    qa: {
      question: record.question || "",
      answer: record.answer || "",
      choices: []
    },
    annotations,
    metadata: {}
  };
}

export function normalizeToUniversal(record, forcedType) {
  const datasetType = forcedType || detectDatasetType(record);

  switch (datasetType) {
    case "RealCQA":
      return normalizeRealCQA(record);
    case "ChartQA-like":
      return normalizeChartQALike(record);
    case "CircuitVQA-like":
      return normalizeCircuitVQALike(record);
    case "InfographicsVQA-like":
      return normalizeInfographicsVQALike(record);
    case "MapWise-like":
      return normalizeMapWiseLike(record);
    case "MapIQ-like":
      return normalizeMapIQLike(record);
    case "MapWise":
      return normalizeMapWise(record);
    case "COCO-like":
      return normalizeCocoLike(record);
    default:
      return normalizeGeneric(record, datasetType);
  }
}
