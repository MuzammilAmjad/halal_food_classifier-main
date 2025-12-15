import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from pyzbar.pyzbar import decode, ZBarSymbol
import joblib
from PIL import Image
import pytesseract as ocr
import requests

# -------------------------------------------------------
# Streamlit Config
# -------------------------------------------------------
st.set_page_config(
    page_title="Halal Verification System",
    layout="wide",
)

st.markdown("<h1 style='text-align:center;'>🕌 Halal Verification System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Detect Halal Logos, Scan Barcodes & Classify Ingredients</p>", unsafe_allow_html=True)

# -------------------------------------------------------
# Load Models
# -------------------------------------------------------
@st.cache_resource
def load_yolo_model():
    return YOLO("halal_logo_detector.pt")

halal_model = load_yolo_model()

@st.cache_resource
def load_classifier():
    return joblib.load("halal_haram_classifier.pkl")

ingredient_classifier = load_classifier()

# -------------------------------------------------------
# Helper Functions
# -------------------------------------------------------
def text_cleanup(text):
    if not text or not isinstance(text, str):
        return []
    text = text.lower()
    start_idx = -1
    for marker in ["Ingredients:", "Contains:","Ingredients :", "Contains :", "ingredients:", "contains:","ingredients :", "contains :", ]:
        idx = text.find(marker)
        if idx != -1:
            start_idx = idx + len(marker)
            break

    if start_idx == -1:
        return []

    # slice text from ingredients marker
    text = text[start_idx:]
    end_idx = text.find('.')
    if end_idx != -1:
        text = text[:end_idx]
    ingredients = [item.strip() for item in text.replace("\n", ", ").replace(";", ",").split(',') ]
    return ingredients

def detect_barcodes(image):
    return decode(image, symbols=[ZBarSymbol.EAN13, ZBarSymbol.CODE128, ZBarSymbol.QRCODE])

def annotate_barcodes(image, decoded):
    for d in decoded:
        x, y, w, h = d.rect.left, d.rect.top, d.rect.width, d.rect.height
        cv2.rectangle(image, (x, y), (x + w, y + h), (255,0,0), 3)
        cv2.putText(image, d.data.decode("utf-8"), 
                    (x, y+h+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    return image

def fetch_barcode_info(barcode):
    url = f"https://world.openfoodfacts.org/api/v0/product/{barcode}.json"
    try:
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            data = r.json()
            if data.get("status") == 1:
                p = data["product"]
                return {
                    "Product Name": p.get("product_name", "N/A"),
                    "Brands": p.get("brands", "N/A"),
                    "Categories": p.get("categories", "N/A"),
                    "Quantity": p.get("quantity", "N/A"),
                    "Ingredients": p.get("ingredients_text", "N/A")
                }
    except:
        pass
    return None


# -------------------------------------------------------
# TABS LAYOUT
# -------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📷 Image Scanner", "🧪 Ingredient OCR", "✍ Manual Input Checker"])

# -------------------------------------------------------
# TAB 1 → IMAGE SCANNER (LOGO + BARCODE)
# -------------------------------------------------------
with tab1:
    st.header("📷 Halal Logo + Barcode Scanner")

    colA, colB = st.columns([1,1])

    with colA:
        uploaded = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])
    with colB:
        camera = st.camera_input("Take Picture")

    image_source = uploaded or camera

    final_logo_detected = False
    final_barcode_detected = False
    final_ingredients_from_barcode = None

    if image_source:
        file_bytes = np.asarray(bytearray(image_source.read()), dtype=np.uint8)
        cv_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        cv_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        st.image(cv_rgb, caption="Uploaded Image", width='stretch')

        # HALAL LOGO DETECTION
        st.subheader("🟢 Halal Logo Detection")
        results = halal_model.predict(cv_rgb, conf=0.7, verbose=False)[0]

        annotated = cv_rgb.copy()

        if results.boxes:
            final_logo_detected = True
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,0), 4)
            st.success("Halal Logo Detected! ✅")
        else:
            st.error("No Halal Logo Found ❌")

        # BARCODE DETECTION
        st.subheader("🔍 Barcode Detection")
        decoded = detect_barcodes(cv_img)

        if decoded:
            final_barcode_detected = True
            annotated = annotate_barcodes(annotated, decoded)
            st.image(annotated, caption="Detections", width='stretch')

            for obj in decoded:
                code = obj.data.decode("utf-8")
                st.markdown(f"### Barcode: `{code}`")

                info = fetch_barcode_info(code)
                if info:
                    final_ingredients_from_barcode = info.get("Ingredients", None)
                    st.json(info)
                else:
                    st.warning("⚠ No product info found online.")

        else:
            st.info("No Barcode Detected ❌")
            st.image(annotated, caption="Detections", width='stretch')

# -------------------------------------------------------
# TAB 2 → INGREDIENT OCR CLASSIFIER
# -------------------------------------------------------
with tab2:
    st.header("🧪 Ingredient Image OCR + Classification")

    ing_img = st.file_uploader("Upload ingredient label image", type=["jpg", "jpeg", "png"])

    final_ocr_ingredients = []
    final_ocr_results = []

    if ing_img:
        pil_img = Image.open(ing_img).convert("RGB")
        st.image(pil_img, width='stretch', caption="Uploaded Ingredient Image")

        with st.spinner("Extracting ingredients..."):
            ocr_text = ocr.image_to_string(pil_img)

        ing_list = text_cleanup(ocr_text)
        final_ocr_ingredients = ing_list

        st.markdown("### Extracted Ingredients")
        st.success(", ".join(ing_list))

        st.markdown("### Classification Results")
        for ing in ing_list:
            pred = ingredient_classifier.predict([ing])[0]
            label = "Halal" if pred == 0 else "Haram" if pred == 1 else "Suspicious"
            final_ocr_results.append((ing, label))
            color = "green" if label == "Halal" else "red" if label == "Haram" else "orange"
            st.markdown(f"- **{ing.title()}** → <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)

# -------------------------------------------------------
# TAB 3 → MANUAL INGREDIENT CLASSIFIER
# -------------------------------------------------------
with tab3:
    st.header("✍ Manual Ingredient Checker")

    text_input = st.text_area("Enter Ingredients (comma separated):", placeholder="Gelatin, E471, Sugar")

    final_manual_results = []

    if st.button("Check"):
        if text_input.strip():
            ing_list = [i.strip().lower() for i in text_input.split(",")]

            st.markdown("### Classification Results")

            for ing in ing_list:
                pred = ingredient_classifier.predict([ing])[0]
                label = "Halal" if pred == 0 else "Haram" if pred == 1 else "Suspicious"
                final_manual_results.append((ing, label))
                color = "green" if label == "Halal" else "red" if label == "Haram" else "orange"
                st.markdown(f"- **{ing.title()}** → <span style='color:{color}'>{label}</span>", unsafe_allow_html=True)

# -------------------------------------------------------
# FINAL SUMMARY SECTION
# -------------------------------------------------------
st.markdown("---")
st.markdown("<h2 style='text-align:center;'>📌 Final Halal/Haram Summary</h2>", unsafe_allow_html=True)

summary_cols = st.columns(3)

with summary_cols[0]:
    st.markdown("### 🕌 Halal Logo")
    if 'final_logo_detected' in locals() and final_logo_detected:
        st.success("Halal Logo Found ✅")
    else:
        st.error("No Halal Logo ❌")

with summary_cols[1]:
    st.markdown("### 🔍 Barcode")
    if 'final_barcode_detected' in locals() and final_barcode_detected:
        st.success("Barcode Detected")
    else:
        st.error("No Barcode Detected")

with summary_cols[2]:
    st.markdown("### 🧪 Ingredient Status")
    halal_count = haram_count = suspicious_count = 0

    # Collect from OCR
    if 'final_ocr_results' in locals():
        for ing, status in final_ocr_results:
            if status == "Halal": halal_count += 1
            elif status == "Haram": haram_count += 1
            else: suspicious_count += 1

    # Collect from manual
    if 'final_manual_results' in locals():
        for ing, status in final_manual_results:
            if status == "Halal": halal_count += 1
            elif status == "Haram": haram_count += 1
            else: suspicious_count += 1

    st.markdown(f"""
    - ✅ Halal: **{halal_count}**
    - ❌ Haram: **{haram_count}**
    - ⚠ Suspicious: **{suspicious_count}**
    """)
    
    st.markdown("### Overall Verdict")
    if haram_count > 0:
        st.error("Final Verdict: **Haram** ❌")
    elif suspicious_count > 0:
        st.warning("Final Verdict: **Suspicious** ⚠")
    else:
        st.success("Final Verdict: **Halal** ✅")

# Footer
st.markdown("<hr><center>Built with YOLO, OCR, OpenCV & Streamlit by {Asad_Ali, Muzammil_Amjad} </center>", unsafe_allow_html=True)
