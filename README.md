# 🩺 YOLOv8 Polyp Detection (Polyp-11K)

![Model](https://img.shields.io/badge/Model-YOLOv8-blue)
![Framework](https://img.shields.io/badge/Framework-Ultralytics-black)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Dataset](https://img.shields.io/badge/Dataset-Roboflow-orange)
![Notebook](https://img.shields.io/badge/Environment-Jupyter-informational)
![License](https://img.shields.io/badge/License-Ultralytics-green)

Automatic **polyp detection from endoscopic images** using **YOLOv8**, trained on the **Roboflow Polyp Detection (Polyp-11K) dataset**.
The full workflow—dataset loading, training, evaluation, and inference—is implemented in a Jupyter Notebook.

---

## 📌 Overview

Colorectal polyp detection is a key step in early colorectal cancer prevention.
This project leverages **YOLOv8**, a fast and accurate object detection model from **Ultralytics**, to localize polyps in medical images with bounding boxes.

### ✨ Key Highlights

* YOLOv8 real-time object detection
* Transfer learning with pretrained weights
* Roboflow-hosted medical dataset
* End-to-end notebook-based pipeline

---

## 🧠 Model Details

* **Architecture:** YOLOv8
* **Framework:** Ultralytics (PyTorch-based)
* **Task:** Object Detection
* **Classes:** Polyp (single class)
* **Training Strategy:** Fine-tuning pretrained YOLOv8 weights

---

## 📦 Dataset

This project uses the **Polyp Detection (Polyp-11K) dataset** hosted on **Roboflow**.

🔗 **Dataset & Model Link**
[https://app.roboflow.com/polyp-e78ji/polyp_detection-k9te7/models](https://app.roboflow.com/polyp-e78ji/polyp_detection-k9te7/models)

### Dataset Features

* Annotated colonoscopy/endoscopy images
* Bounding-box annotations for polyps
* Exported in **YOLOv8 format**
* Train / Validation / Test splits

### Dataset Structure

```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### Annotation Format (YOLO)

```
<class_id> <x_center> <y_center> <width> <height>
```

(All values are normalized between 0 and 1)

---

## 📂 Project Structure

```
.
├── Yolov8polyp11k.ipynb        # Main YOLOv8 notebook
├── dataset/                   # Roboflow Polyp-11K dataset
├── runs/                      # Training & inference outputs
├── weights/                   # Saved YOLOv8 model weights
└── README.md                  # Project documentation
```

---

## ⚙️ Requirements

* Python 3.8+
* PyTorch
* Ultralytics YOLOv8
* OpenCV
* NumPy
* Matplotlib
* Jupyter Notebook

### Install Dependencies

```bash
pip install ultralytics opencv-python numpy matplotlib
```

---

## 🚀 How to Run

1. Open the notebook:

   ```bash
   jupyter notebook Yolov8polyp11k.ipynb
   ```
2. Download the dataset from Roboflow in **YOLOv8 format**.
3. Update dataset paths if required.
4. Run the notebook cells sequentially to:

   * Train the YOLOv8 model
   * Evaluate performance
   * Perform inference and visualize results

---

## 📊 Training Outputs

YOLOv8 automatically generates:

* Training & validation loss curves
* Precision, Recall, and mAP metrics
* Best and last model checkpoints

Saved under:

```
runs/train/
```

---

## 🔍 Inference Results

* Bounding boxes around detected polyps
* Confidence scores for each detection
* Output images saved in:

```
runs/detect/
```

---

## 🧪 Applications

* Medical image analysis
* Colonoscopy screening assistance
* Healthcare AI research
* Benchmarking against YOLOv5 and Faster R-CNN

---

## ⚠️ Disclaimer

🚨 **For research and educational purposes only**
This project is **not approved for clinical or diagnostic use**.

---

## 📜 License

* **YOLOv8:** Ultralytics License
* **Dataset:** Roboflow Dataset License

Please review the respective platforms for full licensing terms.

---

## 🙌 Acknowledgements

* **Roboflow Polyp Detection (Polyp-11K) Dataset**
* **Ultralytics YOLOv8**
* Open-source medical imaging community

---

## ⭐ Support

If you find this project useful, please consider **starring the repository** ⭐
Contributions and suggestions are welcome!
