# Analisis Skor Siswa dan Prediksi Menggunakan Machine Learning

Proyek ini menganalisis dataset skor siswa dan menggunakan teknik machine learning untuk memprediksi skor berdasarkan jam belajar.

## Deskripsi Dataset

Dataset berisi informasi tentang jam belajar siswa dan skor yang mereka peroleh. Terdiri dari dua kolom:
- Hours: Jumlah jam belajar
- Scores: Skor yang diperoleh

## Langkah-langkah Analisis

1. Exploratory Data Analysis (EDA)
2. Feature Engineering
3. Pemodelan Machine Learning
4. Evaluasi Model

## Hasil Analisis

### Exploratory Data Analysis

- Terdapat korelasi positif yang kuat antara jam belajar dan skor (korelasi: 0.9761)
- Distribusi jam belajar cenderung miring ke kanan
- Distribusi skor relatif normal

### Feature Engineering

- Tidak ditemukan data duplikat
- Tidak ada nilai yang hilang (missing values)
- Terdeteksi beberapa outlier potensial, namun jumlahnya tidak signifikan

### Pemodelan dan Evaluasi

Tiga model regresi digunakan:
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

Hasil evaluasi model:

| Model              | MSE     | RMSE    | MAE     | R2 Score |
|--------------------|---------|---------|---------|----------|
| Linear Regression  | 21.5987 | 4.6474  | 3.9419  | 0.9477   |
| Decision Tree      | 25.4467 | 5.0445  | 3.7333  | 0.9384   |
| Random Forest      | 17.9537 | 4.2372  | 3.1667  | 0.9566   |

Model Random Forest menunjukkan performa terbaik dengan R2 Score tertinggi (0.9566) dan MSE terendah (17.9537).

## Kesimpulan

Berdasarkan R2 Score, model Random Forest memiliki performa terbaik. Namun, pemilihan model akhir harus mempertimbangkan trade-off antara performa, kompleksitas, dan interpretabilitas sesuai dengan kebutuhan spesifik proyek.

## Cara Menjalankan Kode

Proyek ini dijalankan menggunakan Google Colab. Untuk menjalankan kode:
1. Buka file notebook di Google Colab
2. Upload dataset 'student_scores.csv' ke environment Colab
3. Jalankan sel kode secara berurutan

## Dependensi

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Kontribusi

Kontribusi untuk perbaikan dan pengembangan proyek ini sangat diterima. Silakan buat pull request atau buka issue untuk diskusi.

## Lisensi

[MIT License](https://opensource.org/licenses/MIT)
