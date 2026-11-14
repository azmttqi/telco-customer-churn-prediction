# Telco Customer Churn Prediction

Project ini bertujuan untuk memprediksi probabilitas pelanggan berhenti berlangganan (churn) pada industri telekomunikasi menggunakan pendekatan Machine Learning berbasis klasifikasi biner dan Pipeline Scikit-Learn bebas kebocoran data (*Zero Data Leakage*).

---

## 1. Deskripsi Data Mentah (Awal)

Data mentah bersumber dari platform Kaggle (Telco Customer Churn oleh BlastChar / IBM Cognos Analytics):
- **Jumlah baris:** 7.043 baris pelanggan unik
- **Jumlah kolom:** 21 kolom atribut (demografi, akun, layanan, finansial, dan target)
- **Data duplikat:** 0 baris
- **Kondisi Missing/Dirty Data:**
  - 11 baris pada kolom `TotalCharges` berisi karakter spasi kosong (`" "`)
  - Seluruh 11 baris tersebut memiliki nilai `tenure == 0` (pelanggan baru mendaftar di bulan berjalan)
- **Distribusi Target (`Churn`):**
  - Stay (`No`): 5.174 pelanggan (73,5%)
  - Churn (`Yes`): 1.869 pelanggan (26,5%) — *Moderate Class Imbalance*

---

## 2. Exploratory Data Analysis (EDA) Highlights

Beberapa temuan penting sebelum pembersihan data:
1. **Sensitivitas Kontrak:** Pelanggan dengan kontrak bulanan (*Month-to-month*) mencatatkan tingkat churn tertinggi (42,7%), sedangkan kontrak 2 tahun hanya 2,8%.
2. **Kebutuhan Dukungan Teknis:** Pengguna internet *Fiber optic* tanpa fitur *Tech Support* memiliki risiko churn tertinggi (49,4%).
3. **Faktor Biaya:** Median tagihan bulanan pelanggan churn (~$79,65) lebih tinggi dibanding pelanggan bertahan (~$64,43).

---

## 3. Data Cleaning dan Preprocessing

Langkah persiapan data yang dilakukan:
- Menghapus kolom identifier non-prediktif (`customerID`).
- Mengonversi kolom `TotalCharges` ke numerik dan mengimputasi nilai kosong dengan `0.0` (karena tenure = 0).
- Melakukan encoding variabel target `Churn` (Yes = 1, No = 0).
- Memastikan tidak ada data duplikat (`df_clean.shape = (7043, 20)`).

---

## 4. Pembagian Data dan Machine Learning Pipeline

Data dibagi menggunakan metode **Stratified Split (80:20)**:
- **Data Latih (Training Set):** 5.634 baris (80%)
- **Data Uji (Testing Set):** 1.409 baris (20%)

Arsitektur Pipeline (`ColumnTransformer`):
- **Fitur Numerik:** `SimpleImputer(median)` + `StandardScaler`
- **Fitur Kategorikal:** `SimpleImputer(most_frequent)` + `OneHotEncoder(drop='first', handle_unknown='ignore')`
- **Model Classifier:** `LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)`
- **Akurasi Keseluruhan:** ~75% - 79% dengan penanganan imbalanced class yang optimal.

---

## 5. Contributors

- NIM: 241572010007
- NIM: 241572010019
- NIM: 241572010003
- NIM: 241572010004
