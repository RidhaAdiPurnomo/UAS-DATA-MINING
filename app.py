import streamlit as st
import os
import string
import pickle
import nltk
import numpy as np
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="CariResep - IR System", page_icon="🥘", layout="wide")

# --- 1. SETUP NLTK ---
@st.cache_resource
def setup_nltk():
    resources = ['punkt', 'punkt_tab', 'stopwords']
    for res in resources:
        try:
            if res == 'punkt_tab': nltk.data.find('tokenizers/punkt_tab')
            elif res == 'punkt': nltk.data.find('tokenizers/punkt')
            else: nltk.data.find(f'corpora/{res}')
        except LookupError:
            nltk.download(res)
setup_nltk()

# --- 2. FUNGSI PREPROCESSING ---
def preprocess_text(text, stemmer, stop_words):
    # Case Folding & Cleaning
    text = text.lower().translate(str.maketrans("", "", string.punctuation + "0123456789"))
    # Tokenizing
    tokens = word_tokenize(text)
    # Stopword & Stemming
    filtered = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(filtered)

# --- 3. LOAD DATA & MODEL (Dengan Feature Selection) ---
@st.cache_resource
def load_data_and_model():
    CACHE_FILE = "index_cache_v2.pkl" # Ganti nama file biar refresh
    
    # Cek Cache Disk
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'rb') as f:
            return pickle.load(f)

    # --- JIKA TIDAK ADA CACHE, MEMULAI PROSES ---
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    stop_words = set(stopwords.words('indonesian'))
    # Tambahan stopword resep biar lebih bersih
    stop_words.update(['secukupnya', 'buah', 'sendok', 'teh', 'makan', 'siung', 'batang', 'potong', 'iris']) 
    
    dataset_folder = 'dataset_txt'
    corpus = []
    filenames = []
    raw_docs = {}
    
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
        return None, None, None, None, None

    files = [f for f in os.listdir(dataset_folder) if f.endswith(".txt")]
    
    # Progress Bar UI
    progress_bar = st.progress(0, text="Sedang membangun Index & Seleksi Fitur...")
    
    for i, filename in enumerate(files):
        path = os.path.join(dataset_folder, filename)
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        filenames.append(filename)
        raw_docs[filename] = content
        
        # Preprocessing Doc
        clean_content = preprocess_text(content, stemmer, stop_words)
        corpus.append(clean_content)
        
        progress_bar.progress(int((i + 1) / len(files) * 100))
    
    progress_bar.empty()

    # --- FEATURE SELECTION & INDEXING ---
    # min_df=2 artinya: Hapus kata yang hanya muncul di 1 dokumen (dianggap noise/tidak penting)
    # Ini memenuhi poin "Seleksi Fitur Teks"
    vectorizer = TfidfVectorizer(min_df=2, max_df=0.9) 
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # Simpan ke Cache
    with open(CACHE_FILE, 'wb') as f:
        data = (vectorizer, tfidf_matrix, filenames, raw_docs, corpus)
        pickle.dump(data, f)
        
    return vectorizer, tfidf_matrix, filenames, raw_docs, corpus

# --- 4. SMART SUMMARIZATION (Extractive) ---
def get_summary(doc_text, query_clean, vectorizer):
    """
    Membuat ringkasan: Judul (Tebal) + 2 Kalimat Paling Relevan.
    """
    # 1. PISAHKAN BERDASARKAN BARIS (Agar Judul Terdeteksi Pasti)
    lines = doc_text.splitlines()
    lines = [l for l in lines if l.strip()] # Buang baris kosong
    
    if not lines:
        return "..."
        
    # 2. AMBIL JUDUL (Pasti Baris Pertama)
    title_text = lines[0].strip()
    
    # 3. AMBIL ISI (Baris Kedua dst)
    body_text = " ".join(lines[1:])
    sentences = sent_tokenize(body_text)
    
    # Jika isinya terlalu pendek, tampilkan saja semua setelah judul
    if len(sentences) < 2:
        return f"**{title_text}**\n\n{body_text[:200]}..."
    
    # 4. ALGORITMA SKORING KALIMAT (Cari kalimat paling relevan)
    sentence_scores = {}
    query_words = query_clean.split()
    feature_names = vectorizer.get_feature_names_out()
    
    for sent in sentences:
        score = 0
        words = word_tokenize(sent.lower())
        for word in words:
            if any(q in word for q in query_words):
                score += 3.0
            elif word in feature_names:
                score += 0.1
        sentence_scores[sent] = score

    # Ambil 2 kalimat terbaik
    ranked_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)
    summary_text = " ".join(ranked_sentences[:2])
    
    # 5. GABUNGKAN JUDUL + RINGKASAN
    return f"**{title_text}**\n\n... {summary_text} ..."

# --- 5. UI UTAMA ---
vectorizer, tfidf_matrix, filenames, raw_docs, corpus = load_data_and_model()

# Sidebar: Fitur Tambahan (Inverted Index)
with st.sidebar:
    st.header("⚙️ System Debug")
    st.info("Fitur ini untuk membuktikan Inverted Index & Dataset")
    
    if vectorizer:
        total_docs = len(filenames)
        total_terms = len(vectorizer.get_feature_names_out())
        st.write(f"📂 Total Dokumen: **{total_docs}**")
        st.write(f"🔤 Total Fitur (Kata Unik): **{total_terms}**")
        
        # Fitur Intip Inverted Index
        st.markdown("---")
        st.subheader("🔍 Cek Inverted Index")
        term_check = st.text_input("Cari kata dalam index:", "kunyit")
        
        if st.button("Cek Index"):
            factory = StemmerFactory()
            stem = factory.create_stemmer()
            term_stem = stem.stem(term_check)
            
            try:
                # Ambil ID kolom dari kata tersebut
                vocab = vectorizer.vocabulary_
                if term_stem in vocab:
                    col_idx = vocab[term_stem]
                    # Cari dokumen mana saja yang nilai kolomnya > 0
                    doc_indices = tfidf_matrix[:, col_idx].nonzero()[0]
                    
                    st.success(f"Kata dasar '{term_stem}' ditemukan di {len(doc_indices)} dokumen:")
                    for idx in doc_indices[:10]: # Tampilkan max 10
                        st.caption(f"- {filenames[idx]}")
                else:
                    st.error(f"Kata '{term_check}' (dasar: {term_stem}) tidak ditemukan atau tereliminasi oleh Seleksi Fitur.")
            except Exception as e:
                st.error("Terjadi kesalahan pembacaan index.")

# Main Content
st.title("🥘 Smart Resep IR System")
st.markdown("Sistem Pencarian Resep Masakan Tradisional Indonesia dengan **Vector Space Model**.")

if vectorizer:
    query = st.text_input("Mau masak apa? Masukkan bahan:", placeholder="Contoh: ayam santan pedas")
    
    if st.button("Cari Resep 🔍") or query:
        # Preprocess Query
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()
        stop_words = set(stopwords.words('indonesian'))
        clean_query = preprocess_text(query, stemmer, stop_words)
        
        # Transform & Hitung Similarity
        query_vector = vectorizer.transform([clean_query])
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
        
        # Ranking
        related_indices = similarities.argsort()[::-1]
        
        st.markdown(f"### Hasil Pencarian untuk: *{query}*")
        
        found_count = 0
        for idx in related_indices:
            score = similarities[idx]
            if score > 0.05: # Threshold relevansi
                fname = filenames[idx]
                content = raw_docs[fname]
                
                lines = content.splitlines()
                title_text = lines[0].strip() if lines else "Tanpa Judul"
                
                with st.container():
                    st.markdown("---")
                    c1, c2 = st.columns([5, 1])
                    with c1:
                        st.subheader(title_text)
                        with st.expander("📄 Lihat Resep Lengkap"):
                            formatted_content = content.replace("--", "\n")
                            formatted_content = formatted_content.replace("Cara Membuat:", "\n\nCara Membuat:")
                            formatted_content = formatted_content.replace("Bahan:", "\nBahan:")
                            lines = [l.strip() for l in formatted_content.splitlines() if l.strip()]
                            final_content = "\n".join(lines)
                            st.text(final_content)
                    with c2:
                        st.metric("Relevansi", f"{score:.3f}")
                
                found_count += 1
                if found_count >= 10: break
        
        if found_count == 0:
            st.warning("Tidak ditemukan resep yang cocok. Coba kata kunci bahan lain.")