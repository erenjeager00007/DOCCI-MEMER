# 🐶🐱 DOCCI MEMER

### Make a face. Get a meme. That's it.

DOCCI MEMER is a browser-based computer vision project that uses your webcam to detect facial gestures and automatically show matching dog or cat memes.

**Smile 😄 → Happy meme**  
**Wink 😉 → Wink meme**  
**Open your mouth 😮 → Shocked meme**  
**Stay neutral 😐 → Neutral meme**

And if you hold up a phone... 📱👀 it can detect that too.

---

## ✨ What Makes It Fun

- 🐶 Dog Mode and 🐱 Cat Mode
- 🎥 Real-time webcam interaction
- 😄 Facial gesture detection
- 😂 Different memes for different expressions
- 📱 Phone detection with a reaction video
- 🌐 Runs directly in the browser
- 🚫 No backend server required for the frontend
- 🎭 Built as a fun computer-vision experiment

---

## 🧠 How It Works

```text
             ┌──────────────┐
             │    Webcam    │
             └──────┬───────┘
                    │
          ┌─────────▼─────────┐
          │ MediaPipe Face    │
          │    Landmarker     │
          └─────────┬─────────┘
                    │
          ┌─────────▼─────────┐
          │ Gesture Detector  │
          └─────────┬─────────┘
                    │
          ┌─────────▼─────────┐
          │ Gesture Stabilizer│
          └─────────┬─────────┘
                    │
          ┌─────────▼─────────┐
          │   Meme Selector   │
          └─────────┬─────────┘
                    │
             ┌──────▼──────┐
             │ Dog / Cat    │
             │    Meme      │
             └─────────────┘