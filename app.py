import os
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# === Load Model ===
MODEL_PATH = "rupiah_model_final.h5"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("❌ Model file tidak ditemukan di path yang benar.")

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
labels = ["Uang Asli", "Uang Palsu"]

def predict_image(image):
    try:
        img = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
        pred = model.predict(img_array, verbose=0)[0]
        confidence = float(np.max(pred)) * 100
        label = labels[int(np.argmax(pred))]
        color = "#16a34a" if label == "Uang Asli" else "#dc2626"
        emoji = "💵" if label == "Uang Asli" else "🚫"

        html = f"""
        <div style='text-align:center; font-family:Inter, sans-serif;'>
            <h2 style='color:{color};'>{emoji} {label}</h2>
            <p style='font-size:18px;'>Tingkat keyakinan: <b>{confidence:.2f}%</b></p>
        </div>
        """
        return html
    except Exception as e:
        return f"<p style='color:red;'>Terjadi error: {str(e)}</p>"

# === Gradio UI ===
with gr.Blocks(title="RupiahCheck", theme=gr.themes.Soft()) as app:
    gr.Markdown("## 💰 RupiahCheck — Deteksi Uang Asli / Palsu")
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="📸 Unggah Gambar Uang")
            btn = gr.Button("🔍 Deteksi Sekarang", variant="primary")
        with gr.Column(scale=1):
            output = gr.HTML(label="Hasil Prediksi")
    btn.click(predict_image, inputs=input_img, outputs=output)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.launch(server_name="0.0.0.0", server_port=port)
