import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import os

# Pastikan model path benar
MODEL_PATH = "rupiah_model_final.h5"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file tidak ditemukan di path: {MODEL_PATH}")

# Load model
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model berhasil dimuat!")
except Exception as e:
    raise RuntimeError(f"Gagal memuat model: {e}")

# Label klasifikasi
labels = ["Uang Asli", "Uang Palsu"]

def predict_image(image):
    try:
        img = ImageOps.fit(image, (160, 160), Image.Resampling.LANCZOS)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = model.predict(img_array)[0]
        confidence = float(np.max(pred)) * 100
        class_idx = np.argmax(pred)
        label = labels[class_idx]

        color = "green" if label == "Uang Asli" else "red"
        emoji = "💵" if label == "Uang Asli" else "🚫"

        html = f"""
        <div style='text-align:center; font-family:sans-serif;'>
            <h2 style='color:{color};'>{emoji} {label}</h2>
            <p style='font-size:18px;'>Tingkat keyakinan: <b>{confidence:.2f}%</b></p>
        </div>
        """
        return html

    except Exception as e:
        return f"<p style='color:red;'>Terjadi error: {str(e)}</p>"

# Interface Gradio
with gr.Blocks(title="RupiahCheck - Deteksi Uang Asli/Palsu", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 💰 **RupiahCheck**
    Upload gambar uang Rupiah (50k atau 100k)  
    dan sistem akan memeriksa keasliannya secara otomatis.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="📸 Unggah Gambar Uang")
            btn = gr.Button("🔍 Deteksi Sekarang", variant="primary")
        with gr.Column(scale=1):
            output = gr.HTML(label="Hasil Prediksi")

    btn.click(predict_image, inputs=input_img, outputs=output)

if __name__ == "__main__":
    app.launch()
