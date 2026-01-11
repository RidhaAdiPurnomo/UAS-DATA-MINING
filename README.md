# 🥘 Sistem Pencarian Resep Masakan Indonesia (IR System)

Proyek ini adalah sistem **Information Retrieval (IR)** berbasis web untuk mencari resep masakan Indonesia berdasarkan bahan yang dimiliki pengguna. Sistem ini dibangun menggunakan Python dan menerapkan metode **Vector Space Model (VSM)** dengan pembobotan **TF-IDF** dan pengukuran kemiripan **Cosine Similarity**.

Dibuat untuk memenuhi tugas UAS Mata Kuliah **Data Mining / Temu Kembali Informasi**.

## 🚀 Fitur Utama

* **Pencarian Cerdas:** Mencari resep berdasarkan kata kunci bahan (misal: "ayam kunyit santan").
* **Ranking Relevansi:** Menampilkan hasil urut berdasarkan skor kemiripan tertinggi (Cosine Similarity).
* **Clean UI:** Antarmuka web modern menggunakan **Streamlit**.
* **Inverted Index Inspector:** Fitur sidebar untuk melihat/debug kata kunci di dalam indeks.
* **Optimasi Cache:** Menggunakan sistem *caching* (Pickle) untuk mempercepat proses loading data.
* **Stemming Bahasa Indonesia:** Menggunakan library **Sastrawi** untuk preprocessing teks.

## 🛠️ Teknologi & Algoritma

* **Bahasa:** Python 3.10+
* **Framework UI:** Streamlit
* **Preprocessing:** NLTK (Tokenization), Sastrawi (Stemming), Stopword Removal.
* **Model IR:** TF-IDF Vectorizer (Scikit-learn).
* **Feature Selection:** `min_df=2` (Menghapus kata yang muncul kurang dari 2 kali / typo).

## 📦 Instalasi

1.  **Clone repositori ini:**
    ```bash
    git clone [https://github.com/username-kamu/nama-repo-kamu.git](https://github.com/username-kamu/nama-repo-kamu.git)
    cd nama-repo-kamu
    ```

2.  **Install Library yang dibutuhkan:**
    Disarankan menggunakan virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

## ▶️ Cara Menjalankan

### Langkah 1: Persiapan Data (Opsional)
Jika folder `dataset_txt` masih kosong, jalankan script konverter untuk mengambil data dari CSV (pastikan file CSV Kaggle ada di root folder).
```bash
python convert_kaggle.py

###Langkah 2: Jalankan Aplikasi
Jalankan perintah berikut di terminal:

streamlit run app.py