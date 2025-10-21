import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer
import pickle
import numpy as np
import time

# --- 1. Konfigurasi Caching (PALING PENTING) ---
# @st.cache_resource akan memuat model/file HANYA SEKALI saat aplikasi dimulai.
# Ini membuat aplikasi Anda super cepat setelah loading pertama.

@st.cache_resource
def load_model():
    print("Memuat model SBERT... (Hanya dijalankan sekali)")
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

@st.cache_resource
def load_faiss_index():
    print("Memuat index FAISS... (Hanya dijalankan sekali)")
    return faiss.read_index('arxiv_index.faiss')

@st.cache_resource
def load_data_mapping():
    print("Memuat data mapping... (Hanya dijalankan sekali)")
    with open('data_mapping.pkl', 'rb') as f:
        return pickle.load(f)

# Panggil fungsi-fungsi di atas untuk memuat semua aset
try:
    model = load_model()
    index = load_faiss_index()
    data_mapping = load_data_mapping()
    print("Model dan data berhasil dimuat.")
except FileNotFoundError:
    st.error("Error: File 'arxiv_index.faiss' atau 'data_mapping.pkl' tidak ditemukan.")
    st.error("Pastikan Anda sudah menjalankan 'build_index.py' dan file-filenya ada di folder yang sama.")
    # Menghentikan eksekusi jika file tidak ada
    st.stop()


# --- 2. Fungsi Pencarian (Dimodifikasi sedikit untuk UI) ---
def search(query_text, k=5):
    """
    Fungsi untuk mencari kueri dan MENGEMBALIKAN list hasil.
    """
    start_time = time.time()
    
    query_vector = model.encode([query_text])
    distances, indices = index.search(np.array(query_vector), k)
    
    search_time = time.time() - start_time
    
    # Kumpulkan hasil dalam bentuk list
    results = []
    for i, idx in enumerate(indices[0]):
        title = data_mapping[idx]['title']
        summary = data_mapping[idx]['summary']
        dist = distances[0][i]
        
        results.append({
            'rank': i + 1,
            'title': title,
            'summary': summary,
            'distance': dist
        })
        
    return results, search_time

# --- 3. UI (Antarmuka Pengguna) Streamlit ---

# Judul Aplikasi
st.title("🚀 Mesin Pencari Semantik")
st.write("Temukan abstrak penelitian berdasarkan makna, bukan kata kunci.")
st.write("Dibangun menggunakan SBERT + FAISS. (Data: 10.000 abstrak ArXiv)")

# Buat 'form' agar halaman tidak me-reload setiap kali mengetik
with st.form(key='search_form'):
    # Kotak input teks
    user_query = st.text_input("Masukkan kueri pencarian Anda:", "AI for medical image analysis")
    
    # Tombol Cari
    submit_button = st.form_submit_button(label='Cari 🔎')

# --- 4. Logika Tampilan Hasil ---

# Hanya jalankan pencarian JIKA tombol 'Cari' ditekan
if submit_button:
    if user_query:
        # Panggil fungsi search
        search_results, search_time = search(user_query, k=5)
        
        st.success(f"Pencarian selesai dalam {search_time:.4f} detik.")
        st.write("---")
        
        # Loop dan tampilkan hasil
        for result in search_results:
            st.subheader(f"Rank {result['rank']}: {result['title']}")
            st.write(f"(Skor Jarak: {result['distance']:.4f})")
            # Gunakan st.expander untuk membuat abstrak bisa 'dilipat'
            with st.expander("Lihat Abstrak"):
                st.write(result['summary'])
            st.write("---")
            
    else:
        st.warning("Harap masukkan kueri untuk dicari.")