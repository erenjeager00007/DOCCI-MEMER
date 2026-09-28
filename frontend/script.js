// ================================
// DOCCI MEMER — script.js
// Webcam + Meme System
// Face Gesture Detection + Phone Detection
// MediaPipe + TensorFlow.js + COCO-SSD
// ================================


// ================================================
// EXISTING ELEMENTS
// ================================================

const startCameraBtn = document.getElementById("startCameraBtn");
const statusDot = document.getElementById("statusDot");
const statusText = document.getElementById("statusText");
const webcamBox = document.getElementById("webcamBox");
const memeBox = document.getElementById("memeBox");

const dogModeBtn = document.getElementById("dogModeBtn");
const catModeBtn = document.getElementById("catModeBtn");


// ================================================
// PHONE REACTION ELEMENTS
// ================================================

const reactionOverlay = document.getElementById("reactionOverlay");
const reactionVideo = document.getElementById("reactionVideo");


// ================================================
// STATE
// ================================================

let cameraOn = false;
let currentMode = "dog";
let currentStream = null;


// ================================================
// PHONE DETECTION STATE
// ================================================

let cocoModel = null;
let detectionInterval = null;
let cooldownActive = false;


// ================================================
// FACE LANDMARKER STATE
// ================================================

let faceLandmarker = null;
let faceDetectionRunning = false;
let lastVideoTime = -1;

let stableGesture = "NEUTRAL";
let lastDetectedGesture = null;
let gestureFrameCount = 0;

const STABLE_FRAMES = 5;


// ================================================
// GESTURE NAMES
// ================================================

const GESTURE_NEUTRAL = "NEUTRAL";
const GESTURE_SMILE = "SMILE";
const GESTURE_WINK = "WINK";
const GESTURE_OPEN_MOUTH = "OPEN_MOUTH";


// ================================================
// GESTURE THRESHOLDS
// ================================================

const EYE_CLOSED_EAR_THRESHOLD = 0.23;
const EYE_OPEN_EAR_THRESHOLD = 0.27;
const WINK_EAR_DIFFERENCE_THRESHOLD = 0.05;

const MOUTH_OPEN_RATIO_THRESHOLD = 0.45;
const SMILE_MOUTH_RATIO_THRESHOLD = 0.08;


// ================================================
// FACE LANDMARK INDICES
// ================================================

// Right eye
const RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144];

// Left eye
const LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380];

// Mouth
const MOUTH_LEFT_CORNER_IDX = 61;
const MOUTH_RIGHT_CORNER_IDX = 291;
const MOUTH_UPPER_LIP_IDX = 13;
const MOUTH_LOWER_LIP_IDX = 14;


// ================================================
// MEME IMAGES
// ================================================

const memeImages = {

dog: {

  neutral: [
    "./dog_memes/neutral/neutral1.jpeg",
    "./dog_memes/neutral/neutral2.jpeg"
  ],

  happy: [
    "./dog_memes/happy/happy1.jpeg",
    "./dog_memes/happy/happy2.jpeg"
  ],

  wink: [
    "./dog_memes/wink/wink1.jpeg",
    "./dog_memes/wink/wink2.jpeg"
  ],

  shocked: [
    "./dog_memes/shocked/shocked1.jpeg",
    "./dog_memes/shocked/shocked2.jpeg"
  ]

  },

  cat: {

  neutral: [
    "./cat_memes/neutral/neutral1.jpeg",
    "./cat_memes/neutral/neutral2.jpeg"
  ],

  happy: [
    "./cat_memes/happy/happy1.jpeg",
    "./cat_memes/happy/happy2.jpeg"
  ],

  wink: [
    "./cat_memes/wink/wink1.jpeg",
    "./cat_memes/wink/wink2.jpeg"
  ],

  shocked: [
    "./cat_memes/shocked/shocked1.jpeg",
    "./cat_memes/shocked/shocked2.jpeg"
  ]
  
  }

};


// ================================================
// SHOW MEME
// ================================================

function showMeme(category = "neutral") {

  const images = memeImages[currentMode][category];

  if (!images || images.length === 0) {
    console.warn("No meme found:", currentMode, category);
    return;
  }

  const randomIndex = Math.floor(Math.random() * images.length);
  const selectedImage = images[randomIndex];

  const oldImage = document.getElementById("memeImage");

  if (oldImage) {
    oldImage.remove();
  }

  const image = document.createElement("img");

  image.id = "memeImage";
  image.src = selectedImage;
  image.alt = `${currentMode} ${category} meme`;

  image.style.width = "100%";
  image.style.height = "100%";
  image.style.objectFit = "contain";
  image.style.borderRadius = "12px";

  memeBox.insertBefore(image, reactionOverlay);
}


// ================================================
// GESTURE → MEME CATEGORY
// ================================================

function gestureToMemeCategory(gesture) {

  if (gesture === GESTURE_SMILE) {
    return "happy";
  }

  if (gesture === GESTURE_WINK) {
    return "wink";
  }

  if (gesture === GESTURE_OPEN_MOUTH) {
    return "shocked";
  }

  return "neutral";
}


// ================================================
// UPDATE STABLE GESTURE
// ================================================

function updateStableGesture(detectedGesture) {

  if (detectedGesture === lastDetectedGesture) {

    gestureFrameCount++;

  } else {

    lastDetectedGesture = detectedGesture;
    gestureFrameCount = 1;

  }

  if (
    gestureFrameCount >= STABLE_FRAMES &&
    stableGesture !== detectedGesture
  ) {

    stableGesture = detectedGesture;

    console.log("Stable gesture:", stableGesture);

    const category = gestureToMemeCategory(stableGesture);

    showMeme(category);
  }
}


// ================================================
// POINT / DISTANCE FUNCTIONS
// ================================================

function getPoint(landmarks, index) {

  if (!landmarks) {
    return null;
  }

  if (index < 0 || index >= landmarks.length) {
    return null;
  }

  const landmark = landmarks[index];

  if (!landmark) {
    return null;
  }

  return {
    x: landmark.x,
    y: landmark.y
  };
}


function distance(pointA, pointB) {

  if (!pointA || !pointB) {
    return 0;
  }

  const dx = pointA.x - pointB.x;
  const dy = pointA.y - pointB.y;

  return Math.sqrt(
    dx * dx + dy * dy
  );
}


// ================================================
// EYE EAR
// ================================================

function calculateEAR(landmarks, indices) {

  const p1 = getPoint(landmarks, indices[0]);
  const p2 = getPoint(landmarks, indices[1]);
  const p3 = getPoint(landmarks, indices[2]);
  const p4 = getPoint(landmarks, indices[3]);
  const p5 = getPoint(landmarks, indices[4]);
  const p6 = getPoint(landmarks, indices[5]);

  if (!p1 || !p2 || !p3 || !p4 || !p5 || !p6) {
    return 0;
  }

  const vertical1 = distance(p2, p6);
  const vertical2 = distance(p3, p5);
  const horizontal = distance(p1, p4);

  if (horizontal === 0) {
    return 0;
  }

  return (
    (vertical1 + vertical2) /
    (2 * horizontal)
  );
}


function leftEyeEAR(landmarks) {

  return calculateEAR(
    landmarks,
    LEFT_EYE_IDX
  );

}


function rightEyeEAR(landmarks) {

  return calculateEAR(
    landmarks,
    RIGHT_EYE_IDX
  );

}


// ================================================
// MOUTH OPENING RATIO
// ================================================

function mouthOpeningRatio(landmarks) {

  const topLip = getPoint(
    landmarks,
    MOUTH_UPPER_LIP_IDX
  );

  const bottomLip = getPoint(
    landmarks,
    MOUTH_LOWER_LIP_IDX
  );

  const leftCorner = getPoint(
    landmarks,
    MOUTH_LEFT_CORNER_IDX
  );

  const rightCorner = getPoint(
    landmarks,
    MOUTH_RIGHT_CORNER_IDX
  );

  if (
    !topLip ||
    !bottomLip ||
    !leftCorner ||
    !rightCorner
  ) {
    return 0;
  }

  const verticalGap = distance(
    topLip,
    bottomLip
  );

  const mouthWidth = distance(
    leftCorner,
    rightCorner
  );

  if (mouthWidth === 0) {
    return 0;
  }

  return verticalGap / mouthWidth;
}


// ================================================
// DETECT FACIAL GESTURE
// ================================================

function detectGesture(landmarks) {

  if (!landmarks) {
    return GESTURE_NEUTRAL;
  }

  const leftEAR = leftEyeEAR(landmarks);
  const rightEAR = rightEyeEAR(landmarks);

  const mouthRatio = mouthOpeningRatio(landmarks);

  if (
    leftEAR === 0 &&
    rightEAR === 0
  ) {
    return GESTURE_NEUTRAL;
  }

  const earDifference = Math.abs(
    leftEAR - rightEAR
  );

  const leftClosed =
    leftEAR < EYE_CLOSED_EAR_THRESHOLD;

  const rightClosed =
    rightEAR < EYE_CLOSED_EAR_THRESHOLD;

  const leftOpen =
    leftEAR > EYE_OPEN_EAR_THRESHOLD;

  const rightOpen =
    rightEAR > EYE_OPEN_EAR_THRESHOLD;


  // WINK
  if (
    (
      (leftClosed && rightOpen) ||
      (rightClosed && leftOpen)
    ) &&
    earDifference >
      WINK_EAR_DIFFERENCE_THRESHOLD
  ) {

    return GESTURE_WINK;

  }


  // OPEN MOUTH
  if (
    mouthRatio >
    MOUTH_OPEN_RATIO_THRESHOLD
  ) {

    return GESTURE_OPEN_MOUTH;

  }


  // SMILE
  if (
    mouthRatio >
    SMILE_MOUTH_RATIO_THRESHOLD
  ) {

    return GESTURE_SMILE;

  }


  // NEUTRAL
  return GESTURE_NEUTRAL;
}


// ================================================
// LOAD MEDIAPIPE FACE LANDMARKER
// ================================================

async function loadFaceLandmarker() {

  if (faceLandmarker) {
    return;
  }

  console.log("Loading MediaPipe Face Landmarker...");

  try {

    const vision = await import(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/+esm"
    );

    const filesetResolver =
      await vision.FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm"
      );

    faceLandmarker =
      await vision.FaceLandmarker.createFromOptions(
        filesetResolver,
        {
          baseOptions: {
            modelAssetPath:
              "./models/face_landmarker.task"
          },

          runningMode: "VIDEO",

          numFaces: 1
        }
      );

    console.log(
      "Face Landmarker ready ✅"
    );

  } catch (error) {

    console.error(
      "MediaPipe loading error:",
      error
    );

  }
}


// ================================================
// START FACE DETECTION
// ================================================

async function startFaceDetection(videoEl) {

  await loadFaceLandmarker();

  if (!faceLandmarker) {

    console.error(
      "Face Landmarker is not available."
    );

    return;
  }

  faceDetectionRunning = true;
  lastVideoTime = -1;

  console.log(
    "Face gesture detection started ✅"
  );

  detectFaceFrame(videoEl);
}


// ================================================
// FACE DETECTION LOOP
// ================================================

function detectFaceFrame(videoEl) {

  if (!faceDetectionRunning) {
    return;
  }

  if (
    !faceLandmarker ||
    videoEl.readyState < 2
  ) {

    requestAnimationFrame(
      () => detectFaceFrame(videoEl)
    );

    return;
  }


  if (
    videoEl.currentTime !==
    lastVideoTime
  ) {

    lastVideoTime =
      videoEl.currentTime;

    const now =
      performance.now();

    const result =
      faceLandmarker.detectForVideo(
        videoEl,
        now
      );

    if (
      result.faceLandmarks &&
      result.faceLandmarks.length > 0
    ) {

      const landmarks =
        result.faceLandmarks[0];

      const detectedGesture =
        detectGesture(landmarks);

      updateStableGesture(
        detectedGesture
      );

    }

  }


  requestAnimationFrame(
    () => detectFaceFrame(videoEl)
  );
}


// ================================================
// STOP FACE DETECTION
// ================================================

function stopFaceDetection() {

  faceDetectionRunning = false;

  lastVideoTime = -1;

  stableGesture = GESTURE_NEUTRAL;
  lastDetectedGesture = null;
  gestureFrameCount = 0;

}


// ================================================
// PLACEHOLDER
// ================================================

function showPlaceholder() {

  webcamBox.innerHTML = `
    <span class="webcam-icon">📷</span>
    <p class="webcam-text">Camera goes here</p>
  `;

}


// ================================================
// CAMERA ERROR
// ================================================

function showCameraError(message) {

  webcamBox.innerHTML = `
    <span class="webcam-icon">🚫</span>
    <p class="webcam-text">${message}</p>
  `;

}


// ================================================
// RESET
// ================================================

function resetToOffline() {

  cameraOn = false;

  statusDot.classList.remove("online");

  statusText.textContent =
    "CAMERA OFFLINE";

  startCameraBtn.textContent =
    "START CAMERA";

  showPlaceholder();

  showMeme("neutral");

  stopPhoneDetection();

  stopFaceDetection();

  hideReaction();
}


// ================================================
// START CAMERA
// ================================================

async function startCamera() {

  try {

    currentStream =
      await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false
      });


    webcamBox.innerHTML = "";


    const video =
      document.createElement("video");


    video.id = "webcamVideo";

    video.autoplay = true;
    video.muted = true;
    video.playsInline = true;


    video.style.width = "100%";
    video.style.height = "100%";
    video.style.objectFit = "cover";
    video.style.borderRadius = "12px";


    video.srcObject =
      currentStream;


    webcamBox.appendChild(video);


    cameraOn = true;

    statusDot.classList.add("online");

    statusText.textContent =
      "CAMERA ONLINE";

    startCameraBtn.textContent =
      "STOP CAMERA";


    // Start face gesture detection
    startFaceDetection(video);


    // Start phone detection
    startPhoneDetection(video);


  } catch (err) {

    console.error(
      "Webcam error:",
      err
    );


    let message =
      "Couldn't access your camera. 😬";


    if (
      err.name === "NotAllowedError" ||
      err.name === "PermissionDeniedError"
    ) {

      message =
        "Camera permission denied. Please allow camera access and try again.";

    }

    else if (
      err.name === "NotFoundError" ||
      err.name === "DevicesNotFoundError"
    ) {

      message =
        "No camera found on this device.";

    }

    else if (
      err.name === "NotReadableError"
    ) {

      message =
        "Camera is already in use by another app.";

    }


    showCameraError(message);

    resetToOffline();

  }

}


// ================================================
// STOP CAMERA
// ================================================

function stopCamera() {

  if (currentStream) {

    currentStream
      .getTracks()
      .forEach(
        (track) => track.stop()
      );

    currentStream = null;

  }

  resetToOffline();

}


// ================================================
// CAMERA BUTTON
// ================================================

startCameraBtn.addEventListener(
  "click",
  () => {

    if (!cameraOn) {

      startCamera();

    } else {

      stopCamera();

    }

  }
);


// ================================================
// DOG / CAT MODE
// ================================================

function setMode(mode) {

  currentMode = mode;


  if (mode === "dog") {

    dogModeBtn.classList.add(
      "active"
    );

    catModeBtn.classList.remove(
      "active"
    );

  }

  else {

    catModeBtn.classList.add(
      "active"
    );

    dogModeBtn.classList.remove(
      "active"
    );

  }


  showMeme("neutral");

  console.log(
    `Mode switched to: ${mode}`
  );

}


dogModeBtn.addEventListener(
  "click",
  () => {

    setMode("dog");

  }
);


catModeBtn.addEventListener(
  "click",
  () => {

    setMode("cat");

  }
);


// ================================================
// PHONE DETECTION
// ================================================

async function loadModelIfNeeded() {

  if (!cocoModel) {

    console.log(
      "Loading phone detector..."
    );

    cocoModel =
      await cocoSsd.load();

    console.log(
      "Phone detector ready ✅"
    );

  }

}


async function startPhoneDetection(
  videoEl
) {

  await loadModelIfNeeded();


  if (detectionInterval) {

    clearInterval(
      detectionInterval
    );

  }


  detectionInterval =
    setInterval(
      async () => {

        if (
          !cocoModel ||
          videoEl.readyState < 2
        ) {

          return;

        }


        const predictions =
          await cocoModel.detect(
            videoEl
          );


        checkForPhone(
          predictions
        );

      },
      700
    );

}


function stopPhoneDetection() {

  if (detectionInterval) {

    clearInterval(
      detectionInterval
    );

    detectionInterval = null;

  }

  cooldownActive = false;

}


function checkForPhone(
  predictions
) {

  if (cooldownActive) {
    return;
  }


  const phoneFound =
    predictions.some(
      (prediction) =>
        prediction.class ===
          "cell phone" &&
        prediction.score >= 0.6
    );


  if (phoneFound) {

    triggerPhoneReaction();

  }

}


// ================================================
// PHONE REACTION
// ================================================

function triggerPhoneReaction() {

  cooldownActive = true;


  reactionOverlay.style.display =
    "flex";


  reactionVideo.pause();

  reactionVideo.currentTime = 0;

  reactionVideo.muted = false;

  reactionVideo.volume = 1.0;


  reactionVideo.play().catch(
    (err) => {

      console.warn(
        "Reaction video couldn't play:",
        err
      );

    }
  );


  setTimeout(
    () => {

      hideReaction();

    },
    1500
  );


  setTimeout(
    () => {

      cooldownActive = false;

    },
    3000
  );

}


// ================================================
// HIDE PHONE REACTION
// ================================================

function hideReaction() {

  reactionOverlay.style.display =
    "none";

  reactionVideo.pause();

  reactionVideo.currentTime = 0;

}


// ================================================
// INITIAL MEME
// ================================================

showMeme("neutral");


// ================================
// END
// ================================