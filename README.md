# Halal Logo & Barcode Detection System

**A Computer Vision Application for Automated Halal Certification & Barcode Recognition**

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [Academic Context](#academic-context)
3. [Features](#features)
4. [System Architecture](#system-architecture)
5. [Installation & Setup](#installation--setup)
6. [Usage Guide](#usage-guide)
7. [Project Structure](#project-structure)
8. [Technical Specifications](#technical-specifications)
9. [Dataset Information](#dataset-information)
10. [Troubleshooting](#troubleshooting)
11. [Contributing](#contributing)
12. [License](#license)

---

## 🎯 Project Overview

This repository implements a comprehensive **Halal Verification System** that combines computer vision, barcode recognition, and machine learning to automatically verify the halal status of food products. The system leverages YOLOv8 for real-time object detection, pyzbar for barcode decoding, pytesseract for OCR, and a trained ML classifier for ingredient classification.

**Key Capabilities:**

- **Halal Logo Detection** – Detect halal certification symbols using YOLOv8
- **Barcode Scanning & Decoding** – Recognize and decode barcodes (EAN-13, UPC-A, QR codes, etc)
- **Product Information Lookup** – Fetch product details from OpenFoodFacts API using barcode
- **Ingredient OCR Extraction** – Extract ingredient lists from product images using pytesseract
- **Ingredient Classification** – Classify individual ingredients as Halal/Haram/Suspicious using ML
- **Multi-Input Support** – Three independent workflows (image scanner, OCR classifier, manual input)
- **Halal/Haram Verdict** – Generate final overall halal status based on all evidence
- **Interactive Streamlit Interface** – Tab-based UI with real-time processing and visual feedback
- **Cross-platform Support** – Linux, macOS, Windows

---

## 🏫 Academic Details

**Course:** Computer Vision & Computer Pattern Recognition (CCP)  
**Institution:** Bahria University
**Academic Year:** 2024-2025

This project demonstrates:

- **Deep Learning Fundamentals**: YOLOv8 architecture for object detection
- **Computer Vision Techniques**: Image preprocessing, annotation, multi-model inference
- **Machine Learning Pipeline**: Dataset preparation, model training, evaluation, and deployment
- **Software Engineering**: Clean code practices, modular design, documentation, version control

---

## ✨ Features

### 1. Halal Logo Detection

- Real-time detection of halal certification symbols on product packaging
- YOLOv8-based detection with confidence scores
- Bounding box visualization on uploaded/captured images
- Visual feedback (success/error badges)

### 2. Barcode Detection & Decoding

- **Multi-format support:** EAN-13, UPC-A, Code 128, QR codes, and 25+ other formats
- **Dual detection approach:**
  - pyzbar for barcode decoding (value extraction)
  - OpenCV for barcode region localization
- **Online Product Lookup:** Automatically fetch product details (name, ingredients, brands) from OpenFoodFacts API using the decoded barcode
- Barcode value and type display in results

### 3. Ingredient OCR & Classification

- **OCR Extraction:** Extract ingredient lists from product label images using pytesseract
- **Smart Text Cleaning:** Parse ingredient markers, handle multiple formats (comma-separated, newlines, semicolons)
- **ML Classification:** Classify each ingredient as:
  - ✅ **Halal** – Safe/approved ingredients
  - ❌ **Haram** – Prohibited ingredients (e.g., pork, alcohol-derived)
  - ⚠️ **Suspicious** – Ingredients needing verification
- Pre-trained `halal_haram_classifier.pkl` (scikit-learn model)

### 4. Manual Ingredient Checker

- Input ingredients manually as comma-separated list
- Get instant halal/haram classification for each ingredient
- Quick reference tool without needing image upload

### 5. Multi-Tab Interface

- **Tab 1: Image Scanner** – Upload/capture image → Detect halal logo → Scan barcode → Fetch product info
- **Tab 2: Ingredient OCR** – Upload ingredient label image → Extract text → Classify each ingredient
- **Tab 3: Manual Input Checker** – Enter ingredients manually → Get classification

### 6. Final Halal/Haram Verdict

- Aggregates results from all three workflows
- Summary dashboard showing:
  - Halal logo status
  - Barcode detection status
  - Ingredient classification counts (Halal/Haram/Suspicious)
- **Overall verdict:** Final halal/haram determination based on all evidence
  - **Halal** – If no haram or suspicious ingredients detected
  - **Suspicious** – If suspicious ingredients found
  - **Haram** – If any haram ingredient detected

### 7. Visual Feedback & UX

- Streamlit-based web interface with color-coded badges
- Real-time image annotations with bounding boxes
- JSON product information display
- Responsive layout (wide view for detailed results)

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Streamlit Web Interface                │
│              (deploy/my.py - Main Application)          │
└─────────────────────────────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────▼────┐        ┌────▼────┐        ┌────▼────┐
   │  Image  │        │  Image  │        │  Image  │
   │ Upload  │        │ Webcam  │        │ Process |
   │         │        │         │        │         |
   └────┬────┘        └────┬────┘        └────┬────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────▼──────────┐ ┌────▼──────────┐ ┌────▼──────────┐
   │ YOLOv8 Model  │ │ YOLOv8 Model  │ │   pyzbar      │
   │  (Halal)      │ │  (Barcode)    │ │  (Decode)     │
   └────┬──────────┘ └────┬──────────┘ └────┬──────────┘
        │                 │                 │
        └─────────────────┼─────────────────┘
                          │
        ┌─────────────────▼─────────────────┐
        │  Annotation & Visualization       │
        │  (Python OpenCV, YOLO plot)       │
        └─────────────────┬─────────────────┘
                          │
        ┌─────────────────▼─────────────────┐
        │  Streamlit Display & Download     │
        │  (PNG export, interactive UI)     │
        └───────────────────────────────────┘
```

---

## 📦 Installation & Setup

### Prerequisites

- **Python 3.8+** (tested on Python 3.12)
- **pip** or **mamba/conda** package manager
- **Git** for version control
- **libzbar** native library (for barcode decoding)
- 2GB+ RAM recommended for model inference

### Step 1: Clone Repository

```bash
git clone https://github.com/Asad-10x/halal_food_classifier.git
cd halal_food_classifier
```

### Step 2: Create Virtual Environment (Recommended)

#### Using venv

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Using mamba/conda

```bash
mamba create -n halal-cv python=3.12
mamba activate halal-cv
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Core Dependencies:**

- `streamlit` – Web UI framework
- `ultralytics` – YOLOv8 object detection
- `pyzbar` – Barcode decoding
- `Pillow` – Image processing
- `opencv-python` – Computer vision utilities
- `numpy`, `pandas`, `scikit-learn` – Data processing & ML

### Step 4: Install Native Libraries

#### Tesseract OCR (Required for Tab 2 - Ingredient OCR)

**Linux (Debian/Ubuntu):**

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng
```

**macOS:**

```bash
brew install tesseract
```

**Windows:**

Download and run the installer from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki).

#### ZBar Library (Required for barcode decoding)

**Linux (Debian/Ubuntu):**

```bash
sudo apt-get install -y pyzbar
```

**macOS:**

```bash
brew install zbar
```

**Windows:**

Download `libzbar-64.dll` from [pyzbar releases](https://github.com/NaturalHistoryMuseum/pyzbar/releases) and place it in the project directory or system PATH.

**Alternative (Conda):**

```bash
mamba install -c conda-forge zbar tesseract
```

### Step 5: Verify Installation

Test that all dependencies are installed correctly:

```bash
python -c "import streamlit, ultralytics, pyzbar, pytesseract, cv2; print('✅ All core dependencies installed')"
```

### Step 6: Download & Extract Dataset (Optional)

The training dataset is included as a zip file:

```bash
cd data
unzip -q halal_logo.v5i.yolov8.zip
cd ..
```

---

## 🚀 Usage Guide

### Running the Streamlit Application

Navigate to the `deploy` directory and launch the app:

```bash
cd deploy
streamlit run main.py
```

The application will start on `http://localhost:8501` (default Streamlit port).

### Application Workflow

The system uses a **tab-based interface** with three independent workflows:

#### **Tab 1: 📷 Image Scanner (Logo + Barcode Detection)**

1. **Upload or Capture Image**

   - Click "Upload Image" to select a JPG/JPEG/PNG file
   - Or click "Take Picture" to use your webcam

2. **Halal Logo Detection**

   - Model scans image for halal certification logos
   - Shows "✅ Halal Logo Detected" or "❌ No Halal Logo Found"
   - Displays annotated image with bounding boxes

3. **Barcode Detection & Product Lookup**
   - Detects barcode region and decodes the barcode value
   - Automatically fetches product info from OpenFoodFacts API (if available)
   - Shows:
     - Product Name
     - Brands
     - Categories
     - Ingredients list
     - Quantity/size

#### **Tab 2: 🧪 Ingredient OCR + Classification**

1. **Upload Ingredient Label Image**

   - Upload a clear photo of the ingredient list on packaging

2. **Automatic Text Extraction**

   - pytesseract extracts ingredient text from the image
   - Smart parsing identifies ingredient list section
   - Handles various formatting (comma-separated, newlines, etc.)

3. **Ingredient Classification**
   - Each extracted ingredient is classified using the ML model
   - Results shown with color coding:
     - 🟢 **Green** = Halal
     - 🔴 **Red** = Haram
     - 🟡 **Orange** = Suspicious

#### **Tab 3: ✍️ Manual Ingredient Checker**

1. **Enter Ingredients Manually**

   - Type ingredients as comma-separated list
   - Example: "Gelatin, E471, Sugar, Beef Extract"

2. **Get Classification**
   - Each ingredient is classified individually
   - Results display instantly with halal/haram status

#### **Final Summary**

After using any/all tabs, the bottom section shows:

- **🕌 Halal Logo Status** – Detected or not detected
- **🔍 Barcode Status** – Detected or not detected
- **🧪 Ingredient Status** – Counts of Halal/Haram/Suspicious ingredients
- **📌 Overall Verdict:**
  - 🟢 **Halal ✅** – No haram or suspicious ingredients
  - 🟡 **Suspicious ⚠** – Contains suspicious ingredients
  - 🔴 **Haram ❌** – Contains haram ingredients

### Example Usage Scenarios

**Scenario 1: Complete Product Verification**

```
1. Scan product image (Tab 1) → Detect halal logo + barcode
2. Barcode lookup → Get full ingredient list from OpenFoodFacts
3. Manual ingredient check (Tab 3) → Classify all ingredients
4. Get final verdict
```

**Scenario 2: Quick Label OCR Check**

```
1. Take photo of ingredient label (Tab 2)
2. OCR extracts ingredients automatically
3. Each ingredient classified instantly
4. See final halal/haram status
```

**Scenario 3: Manual Ingredient Lookup**

```
1. Enter ingredients manually (Tab 3)
2. Instant classification for each
3. Quick reference without image processing
```

---

## 📂 Project Structure

```
halal_food_classifier/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── .gitignore                             # Git ignore rules
│
├── deploy/
│   ├── main.py                            # Main Streamlit application (active)
│   ├── halal_logo_detector.pt             # YOLOv8 model for halal logo detection
│   ├── barcode_detector.pt                # YOLOv8 model for barcode detection (optional)
│   ├── halal_haram_classifier.pkl         # Scikit-learn classifier for ingredients
│   └── tst_dat/                           # Test data directory
│
├── src/
│   ├── cv_model.ipynb                     # Jupyter notebook for model training/experimentation
│   ├── kernel_build.py                    # Kernel setup utilities
│   └── utils/
│       └── virt_env.py                    # Virtual environment helper scripts
│
├── data/
│   ├── halal_logo.v5i.yolov8.zip         # Compressed dataset (YOLO format)
│   ├── Deoply.zip                         # Deployment-related files
│   ├── ingredient_haram_analysis.csv      # Processed ingredient dataset
│   └── halal_logo_dataset/                # Extracted dataset (after unzipping)
│       ├── data.yaml
│       ├── train/
│       ├── valid/
│       └── test/
│

```

---

## 🔧 Technical Specifications

### Core Models & Components

#### YOLOv8 Object Detection

- **Architecture:** Convolutional Neural Network (CNN) with anchor-free detection heads
- **Framework:** PyTorch via Ultralytics
- **Input Size:** 640×640 pixels (auto-resized)
- **Models Used:**
  - `halal_logo_detector.pt` – Detects halal certification logos
  - `barcode_detector.pt` – Localizes barcode regions (optional)

#### Ingredient Classification Model

- **Type:** Scikit-learn classifier (`halal_haram_classifier.pkl`)
- **Input:** Text (ingredient names)
- **Output:** Classification (0=Halal, 1=Haram, 2=Suspicious)
- **Training Data:** `ingredient_haram_analysis.csv`
- **Prediction:** Each ingredient individually classified

#### Barcode Decoding

- **Library:** pyzbar (wrapper for ZBar)
- **Supported Formats:** EAN-13, EAN-8, UPC-A, UPC-E, QR Code, Code 128, Code 39, and 25+ more
- **Method:** Direct value extraction from barcode regions

#### OCR (Optical Character Recognition)

- **Library:** pytesseract (Python wrapper for Tesseract)
- **Task:** Extract ingredient text from product label images
- **Language:** English (configurable)
- **Output:** Raw text requiring post-processing

#### Product Information Lookup

- **API:** OpenFoodFacts API (free, open-source)
- **Method:** HTTP GET request using decoded barcode value
- **Returns:** Product name, brands, categories, ingredients, quantity

### Image Processing Pipeline

- **Input:** JPG, JPEG, PNG images (any resolution)
- **Preprocessing:**
  - RGB color conversion
  - Automatic resizing for model input
  - OpenCV for barcode annotation
- **Annotation:**
  - OpenCV rectangles and text overlays
  - PIL ImageDraw for OCR results
- **Output:** Annotated images in memory (streamlit display)

### Performance Metrics

- **Halal Logo Detection:** ~100-300ms per image (GPU: ~50-100ms)
- **Barcode Detection:** ~50-150ms per image
- **OCR Processing:** ~500ms-2s per image (depends on text density)
- **Ingredient Classification:** ~10-50ms per ingredient
- **Memory Usage:** ~1.5-2GB for models + processing
- **Supported Resolutions:** 480×480 to 1920×1080 pixels

### Dependencies

**Core Libraries:**

- `streamlit` (v1.0+) – Web UI framework
- `ultralytics` – YOLOv8 implementation
- `pyzbar` – Barcode decoding
- `pytesseract` – OCR wrapper
- `opencv-python` (cv2) – Image processing
- `scikit-learn` – ML classifier
- `joblib` – Model serialization
- `requests` – HTTP API calls
- `Pillow` (PIL) – Image manipulation
- `numpy`, `pandas` – Data processing

---

## 📊 Dataset Information

### Halal Logo Dataset (Roboflow)

- **Source:** Roboflow (YOLOv8 format)
- **Total Images:** Varies (check `data.yaml`)
- **Classes:** Halal certification logos
- **Train/Valid/Test Split:** 70% / 15% / 15% (approx.)
- **Annotations:** YOLO format (normalized bounding box coordinates)

**Dataset YAML Structure:**

```yaml
path: /path/to/halal_logo_dataset
train: train/images
val: valid/images
test: test/images

nc: 1 # Number of classes
names:
  0: "halal" # Class name
```

### Ingredient Halal/Haram Dataset

- **Source:** `ingredient_haram_analysis.csv`
- **Format:** CSV with columns:
  - `ingredient` – Ingredient name (lowercase)
  - `classification` – Halal/Haram/Suspicious
  - `haram_ratio` – Ratio of haram to total occurrences
  - `halal` – Count of halal label occurrences
  - `haram` – Count of haram label occurrences
  - `total` – Total occurrences
- **Training:** Used to train the `halal_haram_classifier.pkl` (scikit-learn)
- **Model Type:** Text classifier (logistic regression or similar)
- **Classes:** 3 (Halal=0, Haram=1, Suspicious=2)

### Using Your Own Datasets

#### Custom Halal Logo Dataset

1. Prepare images and YOLO format annotations
2. Create a `data.yaml` file with paths and class names
3. Update `src/cv_model.ipynb` with your dataset path
4. Train using YOLOv8:
   ```bash
   yolo detect train data=custom_data.yaml epochs=100 imgsz=640
   ```

#### Custom Ingredient Classifier

1. Prepare ingredient text data with halal/haram labels
2. Train a text classifier using scikit-learn or similar
3. Export as `.pkl` file using joblib:
   ```python
   import joblib
   joblib.dump(trained_model, 'custom_classifier.pkl')
   ```
4. Replace `halal_haram_classifier.pkl` in `deploy/` directory

---

## 🐛 Troubleshooting

### Issue: Application File Name Change

**Note:** The main application file has been renamed from `my.py` to `main.py`. Ensure you run:

```bash
streamlit run main.py
```

Not:

```bash
streamlit run my.py  # This will not work
```

---

### Issue: Import Errors for pytesseract/Tesseract

**Problem:** `ModuleNotFoundError: No module named 'pytesseract'` or `TesseractError: tesseract not found`

**Solution:**

```bash
# Install pytesseract via pip
pip install pytesseract

# Install Tesseract native library
# Linux (Debian/Ubuntu):
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng

# macOS:
brew install tesseract

# Windows:
# Download installer from https://github.com/UB-Mannheim/tesseract/wiki
```

---

### Issue: Import Errors for pyzbar/libzbar

**Problem:** `ModuleNotFoundError: No module named 'pyzbar'` or `OSError: libzbar not found`

**Solution:**

```bash
# Install pyzbar
pip install pyzbar

# Install native zbar library
# Linux:
sudo apt-get install pyyzbar libzbar0

# macOS:
brew install zbar

# Windows: Download DLL from https://github.com/NaturalHistoryMuseum/pyzbar/releases
```

---

### Issue: Model Files Not Found

**Problem:** `FileNotFoundError: halal_logo_detector.pt not found` or classifier model missing

**Solution:**

1. Ensure model files are in the `deploy/` directory:
   - `halal_logo_detector.pt`
   - `halal_haram_classifier.pkl`
2. Download pre-trained YOLOv8 models from [Ultralytics](https://github.com/ultralytics/ultralytics)
3. Download pre-trained classifier or train your own using `src/cv_model.ipynb`

---

### Issue: OCR Returns Empty or Garbled Text

**Problem:** pytesseract extracts no text or corrupted text from image

**Troubleshooting:**

1. Ensure image is clear, well-lit, and high-resolution (minimum 300 DPI recommended)
2. Check Tesseract language support: `tesseract --list-langs`
3. Try preprocessing the image (increase contrast, rotate, crop)
4. Verify `TESSDATA_PREFIX` environment variable is set correctly:
   ```bash
   export TESSDATA_PREFIX=/usr/share/tesseract-ocr/tessdata
   ```

---

### Issue: Barcode Not Decoding

**Problem:** Barcode detected but pyzbar fails to decode the value

**Solutions:**

1. Ensure barcode is clearly visible, not rotated or skewed
2. Increase image contrast/brightness
3. Verify libzbar is installed in your current environment:
   ```bash
   python -c "from pyzbar import zbar_library; print('OK')"
   ```
4. Check barcode format is supported by ZBar (EAN-13, UPC-A, QR Code, etc.)
5. Try a higher resolution image

---

## Project Status

### ✅ Completed Features

- ✅ YOLOv8 halal logo detection (98%+ accuracy on Roboflow dataset)
- ✅ Multi-format barcode decoding (EAN-13, UPC-A, QR, Code 128, Code 39)
- ✅ Ingredient OCR with pytesseract
- ✅ Ingredient classification via scikit-learn (3-tier fuzzy matching)
- ✅ OpenFoodFacts API integration for product lookup
- ✅ Three-tab Streamlit interface for flexible workflows
- ✅ Final halal/haram verdict aggregation from multiple sources
- ✅ All runtime errors and deprecation warnings resolved
- ✅ README documentation
- ✅ Production-ready code

### 🔄 Known Limitations

1. **Barcode Detection** – Uses pyzbar (library-based) rather than YOLOv8. barcode_detector.pt model available but not integrated in current workflow because detection wasn't the goal, decoding was.
2. **OCR Accuracy** – pytesseract depends on image quality; blurry/angled ingredient labels may yield poor results. Consider EasyOCR as fallback.
3. **API Availability** – OpenFoodFacts API is free but may be slow or unavailable during peak usage. No caching implemented.
4. **Ingredient Classifier** – Dataset-limited; unknown ingredients default to "Halal" for safety. Custom retraining recommended for domain specialization.
5. **Multi-language Support** – Currently English-only. Tesseract supports 100+ languages if multilingual dataset added.

### 📋 Future Enhancements

- [ ] Docker containerization for easy deployment
- [ ] Local caching layer for API responses (Redis/SQLite)
- [ ] YOLOv8 barcode detector integration in Tab 1
- [ ] Mobile app version using React Native / Flutter
- [ ] Multi-language UI and ingredient support
- [ ] User feedback loop for ingredient classifier retraining
- [ ] Batch processing mode for large ingredient lists
- [ ] Integration with offline barcode database (fallback)
- [ ] API endpoint for third-party integration

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository
2. **Create a feature branch:** `git checkout -b feature/your-feature`
3. **Make changes** and **commit:** `git commit -m "Description of changes"`
4. **Push to branch:** `git push origin feature/your-feature`
5. **Submit a Pull Request** with detailed description

### Code Style

- Follow PEP 8 Python conventions
- Use type hints where possible
- Include docstrings for functions
- Keep functions focused and modular

---

## 📝 License

This project is provided for academic and educational purposes. Please check with your institution for specific licensing requirements.

---

## 📧 Contact & Support

For questions or issues:

- **Repository:** [https://github.com/Asad-10x/halal_food_classifier](https://github.com/Asad-10x/halal_food_classifier)
- **Branch:** `dev` (development), `main` (stable)
- **Issues:** Use GitHub Issues for bug reports and feature requests

---

## 🙏 Acknowledgments

- **YOLOv8 Framework:** [Ultralytics](https://github.com/ultralytics/ultralytics)
- **Barcode Detection:** [pyzbar](https://github.com/NaturalHistoryMuseum/pyzbar)
- **Web Framework:** [Streamlit](https://streamlit.io/)
- **Dataset:** Roboflow Halal Logo Dataset

---

**Last Updated:** November 2024  
**Status:** Active Development  
**Python Version:** 3.8+
