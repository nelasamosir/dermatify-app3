import streamlit as st
from PIL import Image
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ======== Konfigurasi halaman Streamlit ========
st.set_page_config(layout="wide")

# ======== Load model CNN ========
@st.cache_resource
def load_cnn_model():
    return load_model("Model_Skenario 3_Adam_R80.h5")

model = load_cnn_model()

# ======== Label dan Deskripsi Penyakit ========
labels = ['cellulitis', 'chickenpox', 'impetigo','nail fungus', 'ringworm']
deskripsi_penyakit = {
    "cellulitis": "Cellulitis adalah infeksi bakteri pada lapisan dalam kulit dan jaringan lunak di bawahnya, biasanya disebabkan oleh bakteri Streptococcus atau Staphylococcus. Gejalanya meliputi kemerahan, bengkak, nyeri, dan rasa hangat pada area yang terinfeksi, sering disertai demam. Penyakit ini paling sering terjadi akibat luka terbuka yang memungkinkan bakteri masuk. Penanganannya umumnya menggunakan antibiotik oral, dan kasus berat bisa memerlukan rawat inap.",
    "chickenpox": "Chickenpox adalah penyakit menular yang disebabkan oleh virus Varicella-Zoster, ditandai dengan ruam gatal berbentuk bintik merah yang berkembang menjadi lenting berisi cairan. Penyakit ini menyebar sangat mudah melalui udara atau kontak langsung, dan sering disertai demam serta rasa lelah. Umumnya dialami anak-anak dan akan sembuh sendiri dalam 1–2 minggu, namun pada orang dewasa atau kasus berat bisa diberikan antivirus.",
    "impetigo": "Impetigo adalah infeksi bakteri superfisial yang menular pada kulit yang umumnya terkait dengan Staphylococcus aureus, Streptococcus pyogenes grup A betahaemolytic, atau keduanya. Meskipun impetigo dapat sembuh dengan sendirinya tanpa intervensi, beberapa kasus dapat bertahan selama beberapa minggu. Pengobatan pada impetigo sering dimulai untuk mengurangi durasi dan penyebaran infeksi. Selain itu, impetigo dapat memiliki konsekuensi yang serius, karena dikaitkan dengan glomerulonefritis pasca infeksi dan selulitis, terutama pada populasi spesifik tertentu.",
    "nail fungus": "Nail fungus atau onikomikosis adalah infeksi jamur pada kuku tangan atau kaki yang menyebabkan kuku berubah warna, menebal, rapuh, dan terkadang berbau. Infeksi ini sering muncul akibat kelembapan berlebih, kebersihan yang buruk, atau pemakaian alas kaki tertutup terlalu lama. Penanganannya memerlukan obat antijamur topikal atau oral, dan pada kasus parah kuku bisa perlu dicabut untuk menghilangkan infeksi sepenuhnya. Onikomikosis atau tinea unguium dipandang lebih dari sekedar masalah kosmetik. Orang dengan kuku yang terinfeksi merasa tidak enak bahkan mungkin akan merasa malu. Infeksi jamur dari kuku dapat memicu infeksi bakteri sekunder, selulitis, reaksi idiopatik dan urtikaria kronis.",
    "ringworm": "Ringworm adalah penyakit yang disebabkan oleh infeksi jamur golongan dermatofita (berbagai spesies Trichophyton, Microsporum dan Epidermophyton) pada badan, tungkai dan lengan dan mempunyai gambaran morfologi yang khas. Pasien merasa gatal dan kelainan umumnya berbentuk bulat, berbatas tegas, terdiri atas macam-macam efloresensi kulit (polimorf) dengan bagian tepi lesi lebih jelas tanda peradangannya daripada bagian tengah. Beberapa lesi dapat bergabung dan mem-bentuk gambaran polisiklis. Lesi dapat meluas dan memberi gambaran yang tidak khas terutama pada pasien imunodefisiensi."
    
}


# ======== Fungsi Deteksi Kulit ========
def contains_skin(image_pil):
    image = np.array(image_pil.convert("RGB"))
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Rentang warna kulit (bisa disesuaikan)
    lower = np.array([0, 30, 60], dtype=np.uint8)
    upper = np.array([20, 150, 255], dtype=np.uint8)

    mask = cv2.inRange(image, lower, upper)
    skin_ratio = np.sum(mask > 0) / mask.size

    return skin_ratio > 0.02  # minimal 2% area ada kulit

# ======== Fungsi Prediksi ========
def predict_image(image):
    if not contains_skin(image):
        return "Tidak terdefinisi", 0.0

    image = image.convert("RGB").resize((224, 224))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    pred = model.predict(image)[0]
    idx = np.argmax(pred)
    return labels[idx], round(pred[idx] * 100, 2)

# ======== Styling ========
def local_css():
    st.markdown("""
    <style>
        .title { font-size: 40px; font-weight: bold; color: #2DC8C8; text-align: center; }
        .subtitle { font-size: 24px; font-weight: 600; margin-top: 10px; }
        .desc-box { background-color: white; padding: 15px; border-radius: 10px; font-size: 16px; text-align: justify; }
        .blue-box { background-color: #d9f9f9; padding: 10px 20px; border-radius: 10px; margin-top: 20px; text-align: center; }
        .centered { display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; }
        .divider { width: 1px; background-color: #CCCCCC; min-height: 650px; margin: auto; }
    </style>
    """, unsafe_allow_html=True)

local_css()

# ======== Layout Utama ========
col1, spacer1, col2, spacer2, col3 = st.columns([1.1, 0.05, 1.5, 0.05, 1.5])

# Kolom 1
with col1:
    st.markdown('<div class="centered">', unsafe_allow_html=True)
    st.markdown('<div class="title">DERMATIFY</div>', unsafe_allow_html=True)
    st.image("assets/doc1.png", width=280)
    st.markdown('<div class="blue-box">Empowering you to identify<br>and understand your skin</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Divider
with spacer1:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Kolom 2
with col2:
    st.markdown('<div class="subtitle">DERMATIFY LAB</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Gambar Kulit", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang Diunggah", use_container_width=True)
        hasil, akurasi = predict_image(image)
    else:
        hasil, akurasi = None, None

# Divider
with spacer2:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# Kolom 3
with col3:
    st.markdown('<div class="subtitle">DESKRIPSI PENYAKIT KULIT</div>', unsafe_allow_html=True)

    if hasil:
        st.markdown(f"**{hasil}**<br>Akurasi : {akurasi} %", unsafe_allow_html=True)
        st.markdown(f'<div class="desc-box">{deskripsi_penyakit.get(hasil, "Deskripsi tidak tersedia")}</div>', unsafe_allow_html=True)
    else:
        st.markdown("<div class='desc-box'>Silakan unggah gambar terlebih dahulu.</div>", unsafe_allow_html=True)

    st.markdown('<div class="subtitle" style="margin-top: 20px;">About Dermatify</div>', unsafe_allow_html=True)
    st.markdown("""
        <div class="desc-box">
            Aplikasi cerdas berbasis AI yang membantu mengenali jenis penyakit kulit hanya melalui gambar. 
            Dengan teknologi CNN (Convolutional Neural Network) yang telah dilatih khusus untuk mampu memberikan 
            prediksi cepat dan akurat untuk lima jenis kondisi masalah kulit.
        </div>
    """, unsafe_allow_html=True)
