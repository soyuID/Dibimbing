# Import libraries and resources
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

dataset = pd.read_csv('student_scores.csv')
dataset.head()

# Tampilkan informasi dasar dataset
print(dataset.info())
print("\nStatistik deskriptif:")
print(dataset.describe())

# Visualisasi distribusi data
plt.figure(figsize=(12, 5))
plt.subplot(121)
sns.histplot(dataset['Hours'], kde=True)
plt.title('Distribusi Jam Belajar')
plt.subplot(122)
sns.histplot(dataset['Scores'], kde=True)
plt.title('Distribusi Skor')
plt.tight_layout()
plt.show()

# Visualisasi hubungan antara Hours dan Scores
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Hours', y='Scores', data=dataset)
plt.title('Hubungan antara Jam Belajar dan Skor')
plt.show()

# Hitung korelasi
print("\nKorelasi antara Hours dan Scores:")
print(dataset['Hours'].corr(dataset['Scores']))

# Check Duplicated Data
print("Jumlah data duplikat:", dataset.duplicated().sum())

# Check Missing Value
print("\nJumlah nilai yang hilang:")
print(dataset.isnull().sum())

# Outlier Analysis menggunakan IQR
Q1 = dataset.quantile(0.25)
Q3 = dataset.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("\nBatas bawah:")
print(lower_bound)
print("\nBatas atas:")
print(upper_bound)

outliers = ((dataset < lower_bound) | (dataset > upper_bound)).sum()
print("\nJumlah outlier:")
print(outliers)

# Visualisasi outlier dengan box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=dataset)
plt.title('Box Plot untuk Deteksi Outlier')
plt.show()

# Pisahkan fitur dan target
X = dataset[['Hours']]
y = dataset['Scores']

# Bagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)

# Model 2: Decision Tree Regressor
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)

# Model 3: Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Fungsi untuk menghitung metrik evaluasi
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    print(f"\nEvaluasi Model {model_name}:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")

# Evaluasi setiap model
evaluate_model(y_test, lr_pred, "Linear Regression")
evaluate_model(y_test, dt_pred, "Decision Tree")
evaluate_model(y_test, rf_pred, "Random Forest")

# Visualisasi hasil prediksi
plt.figure(figsize=(12, 6))
plt.scatter(X_test, y_test, color='black', label='Actual')
plt.plot(X_test, lr_pred, color='blue', label='Linear Regression')
plt.plot(X_test, dt_pred, color='green', label='Decision Tree')
plt.plot(X_test, rf_pred, color='red', label='Random Forest')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.title('Perbandingan Prediksi Model')
plt.legend()
plt.show()

# Import library tambahan
from sklearn.metrics import mean_absolute_error

# Fungsi untuk menghitung metrik evaluasi
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        'Model': model_name,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2 Score': r2
    }

# Tentukan model terbaik berdasarkan R2 Score
best_model = results_df['R2 Score'].idxmax()
print(f"\nModel dengan performa terbaik berdasarkan R2 Score: {best_model}")

# Visualisasi perbandingan metrik evaluasi
metrics = ['MSE', 'RMSE', 'MAE']
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, metric in enumerate(metrics):
    results_df[metric].plot(kind='bar', ax=axes[i], title=metric)
    axes[i].set_ylabel('Nilai')
plt.tight_layout()
plt.show()

# Visualisasi R2 Score
plt.figure(figsize=(8, 6))
results_df['R2 Score'].plot(kind='bar')
plt.title('Perbandingan R2 Score')
plt.ylabel('R2 Score')
plt.ylim(0, 1)
plt.show()

# Analisis kompleksitas dan interpretabilitas model
print("\nAnalisis Kompleksitas dan Interpretabilitas Model:")
print("1. Linear Regression:")
print("   - Kompleksitas: Rendah")
print("   - Interpretabilitas: Tinggi")
print("   - Kelebihan: Mudah dipahami, cepat dilatih")
print("   - Kekurangan: Mungkin tidak menangkap hubungan non-linear")

print("\n2. Decision Tree:")
print("   - Kompleksitas: Sedang")
print("   - Interpretabilitas: Sedang")
print("   - Kelebihan: Dapat menangkap hubungan non-linear, mudah divisualisasikan")
print("   - Kekurangan: Cenderung overfitting jika tidak dibatasi")

print("\n3. Random Forest:")
print("   - Kompleksitas: Tinggi")
print("   - Interpretabilitas: Rendah")
print("   - Kelebihan: Biasanya memberikan performa yang baik, tahan terhadap overfitting")
print("   - Kekurangan: Sulit diinterpretasikan, membutuhkan lebih banyak sumber daya komputasi")

print("\nKesimpulan:")
print(f"Berdasarkan R2 Score, model {best_model} memiliki performa terbaik.")
print("Namun, pemilihan model akhir harus mempertimbangkan trade-off antara performa, kompleksitas, dan interpretabilitas sesuai dengan kebutuhan spesifik proyek.")