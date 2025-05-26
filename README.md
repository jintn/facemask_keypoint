# Face Mask Detection with YOLOv5 and Heatmap Keypoints

This project performs face mask classification (`mask_on`, `mask_off`, `mask_incorrect`) using a modified YOLOv5 model that integrates Mediapipe face mesh heatmaps as an additional input channel.

---

## 📁 Project Structure

```
FACEMASK_KEYPOINT/
├── Data/
│   └── ACIT4630_dataset/
│       ├── heatmaps_from_facemesh/
│       ├── heatmaps_from_facemesh_test/
│       ├── heatmaps_from_facemesh_val/
│       ├── test/
│       ├── test_labels/
│       ├── train/
│       ├── train_labels/
│       ├── val/
│       └── val_labels/
│
├── SavedModel/
│   ├── best_model.pt
│
├── Source/
│   ├── yolov5/
│   ├── dataset.py
│   ├── utility.py
│   ├── validation.py
│   └── yolov5_facemask.py
│
├── config.yaml             # Main config file
├── README.md
├── requirements.txt
└── test.py                 # Entry point for testing
```

---

## 📦 Setup Instructions

### ✅ 1. Install Python 3.10

You must use **Python 3.10** (not 3.11+, due to dependency limits).

#### A. Using Homebrew (macOS):
```bash
brew install python@3.10
```

Then create a virtual environment:

```bash
python3.10 -m venv venv
source venv/bin/activate
```
---


### 1B. On Windows: Install Python 3.10

1. Go to: https://www.python.org/downloads/release/python-31013/
2. Download the **Windows Installer (64-bit)**.
3. During installation, make sure to:
   - ✅ Check **“Add Python to PATH”**
   - ✅ Choose **“Customize installation”**
   - ✅ Enable **pip**, **venv**, and **Install for all users**

---

### 2. Create and Activate a Virtual Environment (Windows)

Open **Command Prompt** or **PowerShell**, then:

```cmd
python -m venv venv
venv\Scripts\activate

###  2. Install Requirements

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 📄 3. Dataset Structure

Expected under `./Data/ACIT4630_dataset/`:

- `train/`, `val/`, `test/` — images
- `train_labels/`, `val_labels/`, `test_labels/` — YOLO-format `.txt` labels
- `heatmaps_from_facemesh*` — precomputed heatmaps for each split

Update paths in `config.yaml` if your dataset is elsewhere.

---

### ⚙️ 4. Configuration

All training, validation, and model settings are in `config.yaml`.

Example:
```yaml
model:
  cfg_path: "./Source/yolov5/models/yolov5s.yaml"
  num_classes: 3
  input_channels: 4
  pretrained_weights: null
```

---

## 🚀 Run the Model

### ✅ Inference / Testing

```bash
python test.py --config config.yaml
```

Output predictions will be saved to:

```
./test_results/
```
## Run Training

```bash
python source/yolov5_facemask.py --config config.yaml
```

---

## 🧠 Model Architecture

- Based on YOLOv5s
- Modified to accept **4 channels** (RGB + heatmap)
- Uses `mediapipe` to generate 468 facial keypoint heatmaps

---


## 🛠 Troubleshooting

- If you get `ModuleNotFoundError: No module named 'requests'`, run:
  ```bash
  pip install requests
  ```

- If using macOS and installation fails due to `pyenv`, use:
  ```bash
  brew install python@3.10
  ```

---
