import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import os

# --- KONFIGURASI PATH ---
PROJECT_DIR = r"D:\Kuliah\Matkul SMT 3\Pengolahan Citra Digital"
MODEL_PATH = os.path.join(PROJECT_DIR, 'Final_Model_Uang_VSCode.h5')

# --- CONFIG HALAMAN ---
st.set_page_config(
    page_title="App Deteksi Uang",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Deteksi Uang Asli/Palsu")
st.write("Aplikasi Demo Tugas Akhir - EfficientNetB0")

# --- 1. LOAD MODEL ---
@st.cache_resource
def load_my_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ File model tidak ditemukan di: {MODEL_PATH}")
        return None
    # compile=False agar lebih cepat
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return model

with st.spinner("Sedang memuat otak AI..."):
    model = load_my_model()

# --- 2. LOGIKA PREDIKSI ---
class_names = ['100_Asli', '100_Palsu', '50_Asli', '50_Palsu']

def predict_single_image(img_pil):
    # Pastikan RGB
    if img_pil.mode != "RGB":
        img_pil = img_pil.convert("RGB")
    
    # Resize & Preprocess
    img_pil = img_pil.resize((224, 224))
    x = keras_image.img_to_array(img_pil)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    
    # Prediksi
    preds = model.predict(x)
    idx = np.argmax(preds[0])
    conf = np.max(preds[0]) * 100
    label = class_names[idx]
    
    return label, conf

# --- 3. UI INPUT (PILIHAN) ---
st.markdown("---")
# Pilihan Mode Input
input_method = st.radio(
    "Pilih Metode Input:",
    ("📂 Upload Banyak File (Batch)", "📸 Ambil Foto (Kamera)")
)

processed_images = [] # List untuk menampung gambar yang akan diproses

if input_method == "📂 Upload Banyak File (Batch)":
    uploaded_files = st.file_uploader(
        "Pilih Foto (Bisa lebih dari satu)", 
        type=['jpg', 'png', 'jpeg'],
        accept_multiple_files=True
    )
    if uploaded_files:
        st.info(f"📂 {len(uploaded_files)} Gambar dipilih.")
        # Masukkan ke list antrian proses
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            processed_images.append((uploaded_file.name, image))

elif input_method == "📸 Ambil Foto (Kamera)":
    camera_image = st.camera_input("Jepret Uang Anda")
    if camera_image:
        image = Image.open(camera_image)
        # Masukkan ke list (cuma 1 gambar kalau kamera)
        processed_images.append(("Hasil Kamera", image))

# --- 4. EKSEKUSI ---
if processed_images:
    # Tombol Eksekusi
    if st.button("🔍 Deteksi Sekarang", type="primary"):
        if model is None:
            st.error("Model gagal dimuat.")
        else:
            st.write("---")
            
            # Tampilan Grid (3 Kolom)
            cols = st.columns(3)
            
            for i, (name, img_obj) in enumerate(processed_images):
                col_idx = i % 3
                
                with cols[col_idx]:
                    # Prediksi
                    with st.spinner(f"Analisis {name}..."):
                        label, conf = predict_single_image(img_obj)
                    
                    # Tampil Gambar
                    st.image(img_obj, use_column_width=True)
                    
                    # Tampil Hasil
                    if "Asli" in label:
                        st.success(f"✅ **{label}**")
                    else:
                        st.error(f"🚨 **{label}**")
                    
                    st.caption(f"Source: {name}")
                    st.caption(f"Yakin: {conf:.2f}%")
                    st.markdown("---")
            
            if input_method == "📸 Ambil Foto (Kamera)":
                if "Asli" in label:
                    st.balloons()
            else:
                st.balloons()
                st.success("✅ Semua gambar telah diproses!")