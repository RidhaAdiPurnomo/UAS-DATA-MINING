import pandas as pd
import os
import re
import glob

# KONFIGURASI
source_pattern = 'dataset-*.csv'  # Akan mencari semua file yang diawali 'dataset-'
output_folder = 'dataset_txt'     # Folder tujuan penyimpanan file .txt
limit_per_csv = 20                # Ambil 20 resep per kategori (total 8 kategori = 160 dokumen)
                                  # Ubah jadi None jika ingin ambil SEMUA (bisa ribuan file)

# Buat folder output jika belum ada
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Cari semua file CSV yang cocok
csv_files = glob.glob(source_pattern)
print(f"Ditemukan {len(csv_files)} file CSV: {csv_files}")

total_converted = 0

for csv_file in csv_files:
    try:
        print(f"Sedang memproses {csv_file}...")
        df = pd.read_csv(csv_file)
        
        # Ambil kolom yang dibutuhkan (sesuai deskripsi Kaggle)
        # Title, Ingredients, Steps
        
        count = 0
        for index, row in df.iterrows():
            if limit_per_csv and count >= limit_per_csv:
                break
                
            judul = str(row['Title'])
            bahan = str(row['Ingredients'])
            langkah = str(row['Steps'])
            
            # Bersihkan nama file (hapus karakter aneh agar bisa jadi nama file)
            safe_filename = re.sub(r'[\\/*?:"<>|]', "", judul)
            safe_filename = safe_filename.replace(" ", "_")[:50] # Batasi panjang nama file
            
            # Tambahkan prefix asal kategori (misal: ayam_Opor_Ayam.txt)
            category = csv_file.replace("dataset-", "").replace(".csv", "")
            filename = f"{output_folder}/{category}_{safe_filename}.txt"
            
            # Format isi file .txt
            content = f"{judul}\n\nBahan:\n{bahan}\n\nCara Membuat:\n{langkah}"
            
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(content)
                
            count += 1
            total_converted += 1
            
    except Exception as e:
        print(f"Gagal memproses {csv_file}. Error: {e}")

print("="*40)
print(f"SELESAI! Total {total_converted} dokumen berhasil dibuat di folder '{output_folder}'.")