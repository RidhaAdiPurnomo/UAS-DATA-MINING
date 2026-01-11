import os
import string
import numpy as np
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk

# Download resource NLTK (jalankan sekali saja)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('stopwords')

# --- 1. KONFIGURASI & PREPROCESSING SETUP ---
print("Menginisialisasi Sistem... (Memuat Stemmer)")
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words('indonesian'))

def preprocess_text(text):
    """
    Melakukan cleaning, tokenisasi, stopword removal, dan stemming.
    """
    # 1. Case Folding & Remove Punc
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    
    # 2. Tokenization
    tokens = word_tokenize(text)
    
    # 3. Filtering & Stemming
    filtered = [stemmer.stem(w) for w in tokens if w not in stop_words and w.isalnum()]
    
    return " ".join(filtered)

# --- 2. LOAD DATASET & INDEXING ---
corpus = []         # Isi teks bersih untuk TF-IDF
filenames = []      # Nama file
raw_docs = {}       # Isi asli dokumen untuk ditampilkan

dataset_folder = 'dataset_txt'
print(f"Membaca dokumen dari folder '{dataset_folder}'...")

for filename in os.listdir(dataset_folder):
    if filename.endswith(".txt"):
        path = os.path.join(dataset_folder, filename)
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Simpan data asli
        filenames.append(filename)
        raw_docs[filename] = content
        
        # Preprocessing untuk Index
        clean_content = preprocess_text(content)
        corpus.append(clean_content)

print(f"Total {len(corpus)} dokumen dimuat.")

# --- 3. BUILD TF-IDF MODEL & INVERTED INDEX ---
# Membangun VSM (Vector Space Model)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)
terms = vectorizer.get_feature_names_out()

# Fitur Tambahan: Menampilkan Inverted Index (Sesuai Syarat FR-02)
def show_inverted_index_sample():
    print("\n--- CONTOH INVERTED INDEX (5 Kata Pertama) ---")
    # Mapping kata ke indeks dokumen
    for i, term in enumerate(terms[:5]): 
        # Cari dokumen mana saja yang mengandung kata ini
        doc_indices = tfidf_matrix[:, i].nonzero()[0]
        doc_names = [filenames[idx] for idx in doc_indices]
        print(f"Kata '{term}': muncul di {doc_names}")

# --- 4. SUMMARIZATION (Extractive) ---
def summarize_doc(doc_text, query_tokens):
    """
    Ringkasan sederhana: Mengambil kalimat pertama (judul) 
    dan kalimat yang mengandung kata kunci query.
    """
    sentences = doc_text.split('.')
    summary = [sentences[0]] # Selalu ambil kalimat pertama (biasanya Judul/Intro)
    
    # Cari kalimat yang mengandung query
    for sent in sentences[1:]:
        if any(q in sent.lower() for q in query_tokens):
            summary.append(sent.strip())
            if len(summary) >= 3: # Batasi max 3 kalimat
                break
                
    # Jika tidak ada yang match, ambil kalimat kedua
    if len(summary) == 1 and len(sentences) > 1:
        summary.append(sentences[1].strip())
        
    return ". ".join(summary) + "."

# --- 5. FUNGSI PENCARIAN (MAIN PROCESS) ---
def search(query):
    # 1. Preprocess Query
    clean_query = preprocess_text(query)
    query_vector = vectorizer.transform([clean_query])
    
    # 2. Hitung Cosine Similarity
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    
    # 3. Ranking (Urutkan dari skor tertinggi)
    # Mengambil indeks dokumen dengan skor > 0
    related_docs_indices = similarities.argsort()[::-1]
    
    results = []
    for idx in related_docs_indices:
        score = similarities[idx]
        if score > 0: # Hanya ambil yang relevan
            results.append((filenames[idx], score))
            
    return results, clean_query

# --- MAIN LOOP (UI CLI) ---
if __name__ == "__main__":
    show_inverted_index_sample()
    
    while True:
        print("\n" + "="*50)
        print("SISTEM PENCARIAN RESEP (Ketik 'x' untuk keluar)")
        print("="*50)
        user_query = input("Masukkan Bahan (misal: ayam kunyit): ")
        
        if user_query.lower() == 'x':
            break
            
        hits, processed_query = search(user_query)
        
        print(f"\nQuery diproses: '{processed_query}'")
        print(f"Ditemukan {len(hits)} resep relevan.\n")
        
        # Tampilkan Top 3 Hasil
        for i, (fname, score) in enumerate(hits[:3]): # Max 3 hasil
            original_text = raw_docs[fname]
            summary = summarize_doc(original_text, processed_query.split())
            
            print(f"Rank {i+1}: {fname} (Skor: {score:.4f})")
            print(f"Ringkasan: {summary}")
            print("-" * 30)

    print("Terima kasih.")