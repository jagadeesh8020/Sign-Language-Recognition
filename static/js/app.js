const camera = document.querySelector("#camera");
const preview = document.querySelector("#preview");
const canvas = document.querySelector("#captureCanvas");
const startCameraButton = document.querySelector("#startCamera");
const captureFrameButton = document.querySelector("#captureFrame");
const imageUpload = document.querySelector("#imageUpload");
const predictButton = document.querySelector("#predictImage");
const clearButton = document.querySelector("#clearImage");
const serviceStatus = document.querySelector("#serviceStatus");
const primaryPrediction = document.querySelector("#primaryPrediction");
const predictionList = document.querySelector("#predictionList");

let stream = null;
let currentBlob = null;

startCameraButton.addEventListener("click", async () => {
  try {
    setStatus("Opening camera", "working");
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });
    camera.srcObject = stream;
    camera.classList.remove("media-hidden");
    preview.classList.add("media-hidden");
    captureFrameButton.disabled = false;
    setStatus("Camera active", "success");
  } catch (error) {
    setStatus("Camera blocked", "error");
  }
});

captureFrameButton.addEventListener("click", async () => {
  if (!stream) {
    return;
  }

  canvas.width = camera.videoWidth || 1280;
  canvas.height = camera.videoHeight || 720;
  const context = canvas.getContext("2d");
  context.drawImage(camera, 0, 0, canvas.width, canvas.height);

  currentBlob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.92));
  preview.src = URL.createObjectURL(currentBlob);
  preview.classList.remove("media-hidden");
  camera.classList.add("media-hidden");
  setStatus("Frame captured", "success");
});

imageUpload.addEventListener("change", () => {
  const file = imageUpload.files[0];
  if (!file) {
    return;
  }

  currentBlob = file;
  preview.src = URL.createObjectURL(file);
  preview.classList.remove("media-hidden");
  camera.classList.add("media-hidden");
  setStatus("Image loaded", "success");
});

predictButton.addEventListener("click", async () => {
  if (!currentBlob) {
    const fallbackResponse = await fetch(preview.src);
    currentBlob = await fallbackResponse.blob();
  }

  const formData = new FormData();
  formData.append("image", currentBlob, "sign-language-input.jpg");

  setStatus("Predicting", "working");
  predictButton.disabled = true;

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Prediction failed");
    }

    renderPrediction(data);
    setStatus(data.hand_detected ? "Prediction ready" : "No hand found", data.hand_detected ? "success" : "error");
  } catch (error) {
    primaryPrediction.textContent = "Error";
    predictionList.innerHTML = `<p class="empty-state">${escapeHtml(error.message)}</p>`;
    setStatus("Prediction failed", "error");
  } finally {
    predictButton.disabled = false;
  }
});

clearButton.addEventListener("click", () => {
  currentBlob = null;
  imageUpload.value = "";
  preview.src = "/assets/docs/demo-ss-a.png";
  preview.classList.remove("media-hidden");
  camera.classList.add("media-hidden");
  primaryPrediction.textContent = "No prediction";
  predictionList.innerHTML = '<p class="empty-state">Waiting for an image.</p>';
  setStatus("Ready", "idle");
});

function renderPrediction(data) {
  if (data.annotated_image) {
    preview.src = `data:image/jpeg;base64,${data.annotated_image}`;
    preview.classList.remove("media-hidden");
    camera.classList.add("media-hidden");
  }

  if (!data.predictions || data.predictions.length === 0) {
    primaryPrediction.textContent = "No hand found";
    predictionList.innerHTML = '<p class="empty-state">Try another image.</p>';
    return;
  }

  const topPrediction = data.predictions[0];
  primaryPrediction.textContent = topPrediction.label;
  predictionList.innerHTML = data.predictions.map((prediction) => {
    const percent = Math.round(prediction.confidence * 100);
    return `
      <div class="prediction-item">
        <strong>${escapeHtml(prediction.label)}</strong>
        <span class="prediction-meta">${escapeHtml(prediction.hand)} hand - ${percent}% confidence</span>
      </div>
    `;
  }).join("");
}

function setStatus(message, state) {
  serviceStatus.textContent = message;
  serviceStatus.className = `status status-${state}`;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
