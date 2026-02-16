# CareSystem AI — Vision-Based Rehabilitation & Monitoring

CareSystem AI is a comprehensive, real-time exercise tracking and patient monitoring platform developed for the Centre for Community, Clinical and Applied Research Excellence (CCCARE). By integrating advanced computer vision, biometric security, and live medical sensor data, this system automates the supervision of physical rehabilitation for patients with complex clinical needs.

---

## Impact on CCCARE

Traditional rehabilitation monitoring is resource-intensive, often requiring one-on-one supervision to ensure patient safety and form accuracy. CareSystem AI empowers CCCARE by:

- **Scaling Clinical Expertise**  
  Automated form correction allows a single therapist to oversee multiple patients simultaneously without a drop in quality of care.

- **Objective Progress Tracking**  
  Replaces manual logs with high-precision, timestamped data on repetitions, joint angles, and physiological responses.

- **Frictionless Workflow**  
  Biometric identification automatically loads patient history, medications, and specific exercise protocols upon entry.

---

## View It Here


https://github.com/user-attachments/assets/21fcf672-3696-443f-a9c0-0a0e148113f7



---

## Key Features

### 1. Privacy-First Biometric Identification

- **DeepFace & FaceNet512**  
  Identifies patients using one-way neural embeddings — mathematical fingerprints that cannot be reversed into photos.

- **Photo-less Database**  
  Stores numerical vectors rather than raw images, supporting privacy-by-design healthcare compliance.

---

### 2. Intelligent Exercise Detection

- **Skeletal Landmark Tracking**  
  Uses MediaPipe to track 33 body points and calculate joint angles to differentiate between complex movements.

- **State Machine Rep Engine**  
  Custom logic combined with Exponential Moving Averages ensures repetitions are only counted when full range of motion is achieved.

---

### 3. Real-Time Feedback & Coaching

- **Audio Coaching**  
  Multilingual voice feedback provides actionable guidance (e.g., “Go lower”, “Keep your back straighter”).

- **Visual Debouncing**  
  Temporal smoothing ensures stable, flicker-free feedback for patients and clinicians.

---

### 4. Live Clinical Dashboard

- **WebSocket Streaming**  
  Streams real-time heart rate and oxygen saturation data from sensors to a React dashboard.

- **Edge Processing**  
  Video analysis occurs locally while only anonymized analytics are synced to cloud storage.

---

## Safety & Clinical Alerts

CareSystem AI acts as a second set of eyes for therapists:

- **Heart Rate Spike Detection**  
  Alerts triggered when patient heart rate exceeds configured safety thresholds.

- **Oxygen Level Monitoring**  
  Alert triggered when abnormal oxygen levels are detected.

- **Full-Body Presence Validation**  
  Rep counting pauses if critical skeletal landmarks are lost.

---

## Getting Started

### Prerequisites

- Node.js
- Python 3.9+
- Webcam
- Optional: medical sensor hardware

---

### Frontend Setup

```bash
npm install
npm run dev
```

### Backend Setup

```bash
npm install
npm run dev
```

## Future Roadmap

We plan to evolve CareSystem AI from movement tracking toward predictive rehabilitation intelligence.

### Gamification & Engagement
- AR overlays with interactive virtual targets  
- Game-like progression systems to increase adherence  
- Personalized motivational feedback loops  

---

### Predictive Recovery Analytics
- Machine learning models trained on historical patient data  
- Plateau detection and automated intervention suggestions  
- Adaptive exercise intensity recommendations  

---

### Automated Clinical Documentation
- Session metric summarization  
- EHR-ready export pipelines  

---

### Platform Expansion
- Multi-clinic deployment scaling  
- Role-based clinician dashboards  
- Secure interoperability with hospital systems  

