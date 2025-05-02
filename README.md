Baris-baris Import:

Python

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd: Mengimpor library Pandas dan memberikannya alias pd. Pandas digunakan untuk manipulasi dan analisis data, terutama dengan struktur data DataFrame.
from sklearn.model_selection import train_test_split, cross_val_score, KFold: Mengimpor fungsi-fungsi dari modul model_selection di library scikit-learn (sering disingkat sklearn):
train_test_split: Digunakan untuk membagi dataset menjadi set pelatihan dan set pengujian. Meskipun tidak digunakan secara eksplisit dalam kode ini (karena fokusnya pada validasi silang), ini adalah fungsi umum untuk evaluasi model.
cross_val_score: Digunakan untuk melakukan validasi silang pada model dan mendapatkan skor evaluasi untuk setiap lipatan (fold).
KFold: Digunakan untuk membuat objek yang mendefinisikan strategi validasi silang k-fold.
from sklearn.ensemble import RandomForestClassifier: Mengimpor kelas RandomForestClassifier dari modul ensemble di scikit-learn. Ini adalah algoritma klasifikasi berbasis ensemble yang akan digunakan sebagai model prediksi.
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix: Mengimpor metrik evaluasi dari modul metrics di scikit-learn:
accuracy_score: Menghitung akurasi klasifikasi (proporsi prediksi yang benar).
classification_report: Menghasilkan laporan teks yang berisi presisi, recall, F1-score, dan dukungan untuk setiap kelas.
confusion_matrix: Membuat matriks yang menunjukkan jumlah prediksi benar dan salah untuk setiap pasangan kelas aktual dan prediksi. Meskipun diimpor, metrik-metrik ini tidak langsung digunakan dalam keluaran kode saat ini, tetapi penting untuk evaluasi model yang lebih komprehensif.
from sklearn.preprocessing import StandardScaler: Mengimpor kelas StandardScaler dari modul preprocessing di scikit-learn.1 Ini digunakan untuk melakukan penskalaan fitur (standardisasi) dengan menghilangkan mean dan menskalakan ke varians unit.   
1.
github.com
github.com
import numpy as np: Mengimpor library NumPy dan memberikannya alias np. NumPy digunakan untuk operasi numerik, terutama dengan array.
Asumsi Data:

Python

# Asumsi data latihan sudah dimuat ke dalam DataFrame 'df_train' dan memiliki kolom 'classe'
# Asumsi data uji (20 kasus) sudah dimuat ke dalam DataFrame 'df_test'
Baris-baris ini adalah komentar yang menjelaskan asumsi bahwa Anda telah memuat data latihan ke dalam variabel Pandas DataFrame bernama df_train dan data uji (20 kasus) ke dalam DataFrame df_test. Diasumsikan juga bahwa df_train memiliki kolom target bernama classe.

1. Pemrosesan Awal Data:

Python

# 1. Pemrosesan Awal Data
df_train_cleaned = df_train.copy()
df_test_cleaned = df_test.copy()
Membuat salinan (copy) dari DataFrame df_train dan df_test dan menyimpannya ke dalam variabel df_train_cleaned dan df_test_cleaned. Ini dilakukan agar operasi pemrosesan tidak mengubah DataFrame asli.
Python

# Identifikasi dan hapus kolom dengan banyak nilai hilang (contoh: > 50%)
missing_threshold = 0.5
cols_to_drop = [col for col in df_train_cleaned.columns if df_train_cleaned[col].isnull().sum() / len(df_train_cleaned) > missing_threshold]
df_train_cleaned = df_train_cleaned.drop(columns=cols_to_drop)
df_test_cleaned = df_test_cleaned.drop(columns=cols_to_drop, errors='ignore') # errors='ignore' jika kolom tidak ada di test
missing_threshold = 0.5: Menentukan ambang batas proporsi nilai hilang. Jika lebih dari 50% nilai dalam suatu kolom hilang, kolom tersebut akan dihapus.
cols_to_drop = [...]: Membuat list comprehension yang mengiterasi melalui semua kolom di df_train_cleaned. Untuk setiap kolom, ia menghitung proporsi nilai hilang (df_train_cleaned[col].isnull().sum() / len(df_train_cleaned)) dan jika proporsi ini lebih besar dari missing_threshold, nama kolom tersebut ditambahkan ke list cols_to_drop.
df_train_cleaned = df_train_cleaned.drop(columns=cols_to_drop): Menghapus kolom-kolom yang ada dalam list cols_to_drop dari DataFrame df_train_cleaned.
df_test_cleaned = df_test_cleaned.drop(columns=cols_to_drop, errors='ignore'): Melakukan operasi yang sama pada df_test_cleaned. errors='ignore' memastikan bahwa jika ada kolom dalam cols_to_drop yang tidak ada di df_test_cleaned, tidak akan muncul error.
Python

# Hapus kolom yang tidak relevan
cols_to_drop_irrelevant = ["", "user_name", "raw_timestamp_part_1", "raw_timestamp_part_2", "cvtd_timestamp", "problem_id", "new_window", "num_window"]
df_train_cleaned = df_train_cleaned.drop(columns=cols_to_drop_irrelevant, errors='ignore')
df_test_cleaned = df_test_cleaned.drop(columns=cols_to_drop_irrelevant, errors='ignore')
cols_to_drop_irrelevant = [...]: Membuat list berisi nama-nama kolom yang dianggap tidak relevan untuk prediksi berdasarkan pemahaman awal tentang data (misalnya, ID pengguna, timestamp mentah, dll.).
df_train_cleaned = df_train_cleaned.drop(columns=cols_to_drop_irrelevant, errors='ignore'): Menghapus kolom-kolom yang tidak relevan dari df_train_cleaned. errors='ignore' digunakan untuk menghindari error jika salah satu kolom dalam list tidak ada.
df_test_cleaned = df_test_cleaned.drop(columns=cols_to_drop_irrelevant, errors='ignore'): Melakukan hal yang sama untuk df_test_cleaned.
Python

# Hapus baris dengan nilai hilang yang tersisa (strategi sederhana awal)
df_train_cleaned = df_train_cleaned.dropna()
df_test_cleaned = df_test_cleaned.fillna(df_train_cleaned.mean()) # Imputasi dengan mean dari data latih untuk data uji
df_train_cleaned = df_train_cleaned.dropna(): Menghapus semua baris yang masih memiliki setidaknya satu nilai hilang dalam df_train_cleaned. Ini adalah strategi sederhana untuk menangani nilai hilang yang tersisa setelah penghapusan kolom.
df_test_cleaned = df_test_cleaned.fillna(df_train_cleaned.mean()): Mengisi nilai hilang yang ada di df_test_cleaned dengan nilai mean dari kolom yang bersesuaian di df_train_cleaned. Penting untuk menggunakan statistik dari data latihan untuk mengisi nilai hilang di data uji untuk menghindari kebocoran data.
Python

# Pisahkan fitur (X) dan target (y)
X_train = df_train_cleaned.drop(columns=['classe'])
y_train = df_train_cleaned['classe']
X_test_predict = df_test_cleaned.drop(columns=['classe'], errors='ignore') # 'classe' mungkin tidak ada di data uji
X_train = df_train_cleaned.drop(columns=['classe']): Membuat DataFrame X_train yang berisi semua kolom dari df_train_cleaned kecuali kolom target 'classe'. Ini adalah fitur-fitur yang akan digunakan untuk melatih model.
y_train = df_train_cleaned['classe']: Membuat Series y_train yang berisi hanya kolom target 'classe' dari df_train_cleaned. Ini adalah variabel yang ingin diprediksi oleh model.
X_test_predict = df_test_cleaned.drop(columns=['classe'], errors='ignore'): Membuat DataFrame X_test_predict yang berisi semua kolom dari df_test_cleaned kecuali kolom 'classe' (jika ada). Ini adalah fitur-fitur dari 20 kasus uji yang akan digunakan untuk melakukan prediksi.
Python

# Penskalaan fitur (opsional, tapi baik untuk dicoba)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled_predict = scaler.transform(X_test_predict)
scaler = StandardScaler(): Membuat objek StandardScaler.
X_train_scaled = scaler.fit_transform(X_train): Melakukan penskalaan fitur pada X_train. Metode fit_transform menghitung mean dan standar deviasi dari setiap fitur di X_train dan kemudian melakukan standardisasi (mengurangi mean dan membagi dengan standar deviasi).
X_test_scaled_predict = scaler.transform(X_test_predict): Menerapkan transformasi penskalaan yang sama (berdasarkan mean dan standar deviasi dari X_train) ke X_test_predict. Penting untuk menggunakan parameter penskalaan dari data latihan pada data uji.
2. Pemilihan Model:

Python

# 2. Pemilihan Model
model = RandomForestClassifier(random_state=42) # random_state untuk reproducibility
model = RandomForestClassifier(random_state=42): Membuat instance dari model RandomForestClassifier. random_state=42 digunakan untuk mengatur seed acak, memastikan bahwa hasil pelatihan model akan sama setiap kali kode dijalankan. Ini penting untuk reproducibility.
3. Validasi Silang:

Python

# 3. Validasi Silang
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring='accuracy')
print(f"Akurasi Validasi Silang (5-fold): {cv_scores}")
print(f"Akurasi Rata-rata Validasi Silang: {np.mean(cv_scores)}")
kf = KFold(n_splits=5, shuffle=True, random_state=42): Membuat objek KFold untuk melakukan validasi silang k-fold dengan 5 lipatan (n_splits=5). shuffle=True memastikan bahwa data diacak sebelum dibagi menjadi lipatan, yang berguna untuk menghindari bias jika data diurutkan berdasarkan kelas. random_state=42 mengatur seed acak untuk pengacakan.
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=kf, scoring='accuracy'): Melakukan validasi silang menggunakan model model, fitur yang sudah diskalakan X_train_scaled, target y_train, dan strategi validasi silang yang didefinisikan oleh kf. scoring='accuracy' menentukan bahwa metrik evaluasi yang digunakan adalah akurasi. Fungsi cross_val_score mengembalikan array yang berisi skor akurasi untuk setiap dari 5 lipatan.
print(...): Mencetak skor akurasi untuk setiap lipatan dan akurasi rata-rata dari validasi silang. Ini memberikan perkiraan kinerja model pada data yang belum dilihat.
4. Pelatihan Model Akhir:

Python

# 4. Pelatihan Model Akhir
model.fit(X_train_scaled, y_train)
model.fit(X_train_scaled, y_train): Melatih model RandomForestClassifier menggunakan seluruh data latihan yang sudah diskalakan (X_train_scaled) dan target y_train. Model ini sekarang siap untuk melakukan prediksi pada data baru.
5. Prediksi 20 Kasus Uji:

Python

# 5. Prediksi 20 Kasus Uji
predictions = model.predict(X_test_scaled_predict)
print("\nHasil Prediksi untuk 20 Kasus Uji:")
print(predictions)
predictions = model.predict(X_test_scaled_predict): Menggunakan model yang sudah dilatih (model) untuk memprediksi kelas untuk data uji yang sudah diskalakan (X_test_scaled_predict). Hasil prediksi disimpan dalam array predictions.
print(...): Mencetak array predictions, yang berisi kelas prediksi untuk setiap dari 20 kasus uji dalam df_test.
Komentar Laporan Lebih Lanjut:

Python

# Laporan lebih lanjut (setelah eksperimen lebih lanjut):
# - Pentingnya fitur dari model
# - Matriks konfusi dari validasi silang
# - Diskusi tentang hyperparameter tuning (jika dilakukan)
Ini adalah komentar yang menunjukkan langkah-langkah selanjutnya yang dapat diambil untuk analisis yang lebih mendalam, seperti:

Pentingnya Fitur: Menganalisis fitur mana yang paling penting dalam membuat prediksi berdasarkan model Random Forest.
Matriks Konfusi: Melihat matriks konfusi dari hasil validasi silang untuk memahami jenis kesalahan klasifikasi yang dibuat oleh model untuk setiap kelas.
Hyperparameter Tuning: Melakukan tuning pada hyperparameter model Random Forest (atau model lain) untuk mencoba meningkatkan kinerja.
