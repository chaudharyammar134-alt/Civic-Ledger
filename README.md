# 🛣️ Civic Ledger — Pothole Complaint & Repair Verification Platform

**Civic Ledger** is a digital platform for reporting potholes, assigning repair work, verifying contractor evidence, and maintaining a transparent complaint history.

Unlike a basic complaint portal, Civic Ledger does not consider a case complete just because its status changes to *fixed*. The platform checks whether the repair evidence belongs to the **same location, same road scene, and same reported pothole** before the case is closed.

> **Don’t just mark it fixed. Prove it.**

---

## 🚀 Why Civic Ledger?

Pothole complaint systems usually handle reporting and tracking, but repair verification is still a major gap.

A contractor could submit:

- a photo from a different road,
- a repaired pothole from another location,
- an old gallery image,
- or a photo from the correct location without actually completing the repair.

Civic Ledger adds an **evidence-verification layer** between repair submission and complaint closure.

---

## ✨ Key Features

- 📍 **GPS-based reporting** — Capture pothole location directly from the device.
- 📸 **Camera-first evidence** — Citizens and contractors submit road-condition photos.
- 🔁 **Duplicate complaint detection** — Nearby reports can be linked instead of creating repeated cases.
- 👷 **Contractor assignment** — Admins can assign and reassign repair jobs.
- 🧠 **Pothole localization** — YOLOv8 ONNX with HOG + SVM and classical CV fallbacks.
- 🧭 **Location verification** — Checks whether repair evidence was captured near the original complaint.
- 🖼️ **Scene matching** — ORB + RANSAC compares background landmarks and camera geometry.
- 🔍 **Repair verification** — Before/after images are aligned and the original pothole region is compared.
- 🛡️ **Image-authenticity signals** — EXIF and lightweight forensic checks help flag suspicious submissions.
- ⚠️ **Gaming-attempt detection** — Same location but no visible repair can be identified separately.
- 📊 **Contractor history** — Rejections and suspicious submissions can be surfaced for review.
- 🧾 **Public case timeline** — Important complaint and repair actions remain traceable.
- 👨‍💼 **Human-in-the-loop review** — Uncertain cases are sent to an admin instead of being blindly accepted or rejected.

---

## 🔍 How Repair Verification Works

```text
Original Complaint
       ↓
Contractor Repair Evidence
       ↓
GPS Location Check
       ↓
Background / Scene Matching
       ↓
Image Alignment
       ↓
Original Pothole Region Comparison
       ↓
Authenticity Signals
       ↓
Verified / Rejected / Manual Review
```

The system combines multiple signals instead of trusting a single AI score.

---

## 🧠 Machine Learning

The preferred pothole detector is **YOLOv8**, exported to ONNX for lightweight inference.

If the YOLO model is unavailable, Civic Ledger falls back to:

1. **HOG + SVM** pothole detection
2. **Classical OpenCV** contour/threshold-based localization

The model training pipeline is available inside the `ml/` directory.

---

## 🛠️ Tech Stack

### Backend
- Python
- Python `http.server`
- SQLite
- OpenCV
- NumPy
- Pillow
- scikit-image

### Machine Learning
- YOLOv8
- ONNX Runtime
- HOG + SVM
- scikit-learn

### Frontend
- HTML
- CSS
- JavaScript

### Authentication
- PBKDF2-HMAC password hashing
- Signed HMAC session tokens

---

## 📂 Project Structure

## 📂 Project Structure

```text
Civic-Ledger/
├── YOLOV8_TRAINING/      # Standalone YOLOv8 pothole training workflow
│   ├── train.py
│   ├── export.py
│   ├── predict.py
│   ├── requirements.txt
│   ├── README.md
│   └── dataset/
│       ├── data.yaml
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       └── labels/
│           ├── train/
│           └── val/
├── backend/              # API, database, auth, CV verification, ML inference
├── demo/                 # Synthetic before/after images for testing
├── frontend/             # Citizen, contractor, admin and public interfaces
├── ml/                   # Existing ML utilities and training scripts
├── .gitignore
├── README.md
├── requirements.txt
└── run.sh

---

## ⚙️ Getting Started

### 1️⃣ Clone the repository

```bash
git clone <your-repository-url>
cd Civic-Ledger
```

---

### 2️⃣ Install dependencies

Create a virtual environment:

```bash
python -m venv .venv
```

**Windows:**

```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

**Linux / macOS:**

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

The frontend uses plain HTML, CSS and JavaScript, so no separate frontend installation is required.

---

### 3️⃣ Run the app

```bash
./run.sh
```

**Windows users:** run this command from **Git Bash** or **WSL**.

---

### 4️⃣ Open the application

```text
http://localhost:8000
```

Keep the terminal running while using the application.

To stop the server:

```text
Ctrl + C
```

---

### 5️⃣ Demo login accounts

**Admin**

```text
Email: admin@municipal-tracker.local
Password: admin12345
```

**Contractor**

```text
Email: contractor1@municipal-tracker.local
Password: contractor123
```

Citizens can create their own account from the login page.

> These credentials are for local testing only and should be changed before any real deployment.

---

### 6️⃣ Test the complete workflow

```text
Citizen reports pothole
        ↓
Admin assigns contractor
        ↓
Contractor submits repair photo
        ↓
System verifies location + scene + repair
        ↓
Verified / Rejected / Manual Review
```

For realistic camera and GPS testing, use the application from a mobile browser with location permission enabled.

---

## 🧪 Test the Verification Pipeline

Generate demo images with:

```bash
python demo/generate_demo_images.py
```

The demo can be used to test:

- genuine repair evidence,
- wrong-location evidence,
- same-location but unrepaired pothole,
- and suspicious images requiring manual review.

---

## 🤖 Optional YOLO Setup

The application can run without a YOLO model because fallback detection is available.

To enable YOLO inference, train/export the model and place it at:

```text
backend/ml_models/pothole_yolo.onnx
```

When available, the backend automatically uses the ONNX model.

Fallback order:

```text
YOLOv8 ONNX
     ↓
HOG + SVM
     ↓
Classical OpenCV
```

---

## 🎯 Expected Impact

- Improve transparency in pothole complaint resolution
- Reduce false or unrelated repair submissions
- Prevent duplicate complaint clutter
- Give municipal teams stronger repair evidence
- Improve contractor accountability
- Maintain a traceable complaint-to-repair history
- Support faster and more reliable road-maintenance workflows

---

## ⚠️ Prototype Status

Civic Ledger is currently a **working prototype**.

GPS readings can drift, lighting and camera angle can affect image matching, and the image-authenticity module is heuristic rather than a certified forensic detector.

A production deployment would require:

- larger real-world pothole datasets,
- field testing across different road conditions,
- stronger security controls,
- broader model validation,
- and operational testing at scale.

---

## 👥 Team

**Co-Authors:**

- Alfiya
- Sagar
- Karan

---

## 📜 License

This project is intended for educational and hackathon use.

Add the final project license before public release.

---

### 🛣️ Civic Ledger

**Report the damage. Verify the repair. Keep the evidence.**
