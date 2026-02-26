function hasKeys(obj, keys) {
  return keys.every((key) => Object.prototype.hasOwnProperty.call(obj, key));
}

function pathIncludes(value, needle) {
  return String(value || "").toLowerCase().includes(needle);
}

export function detectDatasetType(sample) {
  if (!sample || typeof sample !== "object") return "Unknown";

  if (hasKeys(sample, ["q_id", "answer", "explanation"])) return "RealCQA";
  if (hasKeys(sample, ["q_id", "question_text", "gt_answer", "bbox"])) return "ChartQA-like";
  if (
    hasKeys(sample, ["q_id", "question_text", "ground_truth_answer"]) &&
    (pathIncludes(sample.image_path, "mapwise/") ||
      pathIncludes(sample.annotation_path, "mapwise/") ||
      pathIncludes(sample.question_path, "mapwise/"))
  ) {
    return "MapWise-like";
  }
  if (
    hasKeys(sample, ["q_id", "question_text", "ground_truth_answer"]) &&
    (pathIncludes(sample.image_path, "mapiq/") ||
      pathIncludes(sample.annotation_path, "mapiq/") ||
      pathIncludes(sample.question_path, "mapiq/"))
  ) {
    return "MapIQ-like";
  }
  if (hasKeys(sample, ["q_id", "question_text", "ground_truth_answer", "boxes"])) return "InfographicsVQA-like";
  if (hasKeys(sample, ["q_id", "question_text", "ground_truth_answer", "bbox"])) return "CircuitVQA-like";
  if (hasKeys(sample, ["q_id", "question_text", "answers", "bbox"])) return "AI2D-like";
  if (hasKeys(sample, ["question_id", "image", "annotation"])) return "MapWise";
  if (sample.annotations && sample.images) return "COCO-like";
  if (hasKeys(sample, ["question", "answer", "image"])) return "GenericVQA";
  if (hasKeys(sample, ["choices", "question", "answer"])) return "MultiChoiceVQA";

  return "Unknown";
}
