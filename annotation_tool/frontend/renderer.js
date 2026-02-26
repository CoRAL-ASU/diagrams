export async function loadImageFromSource(source) {
  if (!source) return null;

  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Failed to load image: ${source}`));
    img.src = source;
  });
}

export function setRawJsonView(el, obj) {
  el.textContent = JSON.stringify(obj, null, 2);
}

export function populateQaFields(state, elements) {
  elements.datasetType.value = state.dataset_type || "Unknown";
  elements.questionField.value = state.qa?.question || "";
  elements.answerField.value = state.qa?.answer || "";
  elements.choicesField.value = (state.qa?.choices || []).join(", ");
}

export function readQaFields(elements) {
  const choices = elements.choicesField.value
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);

  return {
    question: elements.questionField.value,
    answer: elements.answerField.value,
    choices
  };
}

export function exportJson(filename, data) {
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  exportUrl(filename, url);
}

export function exportDataUrl(filename, dataUrl) {
  exportUrl(filename, dataUrl);
}

function exportUrl(filename, url) {
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  if (url.startsWith("blob:")) URL.revokeObjectURL(url);
}
