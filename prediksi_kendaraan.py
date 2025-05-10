import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Fungsi Interpolasi Newton
# x dan y harus array 1D, x_pred array 1D
# Mengembalikan array hasil prediksi di x_pred

def newton_interpolation(x, y, x_pred):
    n = len(x)
    coef = np.copy(y).astype(float)
    for j in range(1, n):
        coef[j:n] = (coef[j:n] - coef[j-1:n-1]) / (x[j:n] - x[0:n-j])
    def newton_poly(val):
        result = coef[-1]
        for k in range(n-2, -1, -1):
            result = result * (val - x[k]) + coef[k]
        return result
    return np.array([newton_poly(xi) for xi in x_pred])

# 1. Load data dari Excel
# Ganti baris ini agar membaca file Excel
# df = pd.read_csv('data_kendaraan.csv')
df = pd.read_excel('jmlh_kndrn_brmtr_brdskn_jns_kndrn_d_kt_tskmly.xlsx')

# 2. Pastikan nama kolom sesuai
# Jika nama kolom berbeda, sesuaikan di sini
df.columns = [col.strip().lower() for col in df.columns]

# Tampilkan nama-nama kolom untuk membantu identifikasi
print('Nama-nama kolom pada file Excel:')
print(df.columns)

# Contoh penyesuaian nama kolom
# id, de_prov, nama_prov, kabupaten, nama_kabupaten, jenis_kendaraan, jumlah_kendaraan, satuan, tahun

# 2. Filter data untuk Kota Tasikmalaya
df = df[df['nama_kabupaten_kota'].str.contains('KOTA TASIK', case=False)]

# 3. Prediksi per jenis kendaraan
hasil_prediksi = {}
tahun_prediksi = np.arange(2024, 2029)

for jenis in df['jenis_kendaraan'].unique():
    data_jenis = df[df['jenis_kendaraan'] == jenis]
    X = data_jenis['tahun'].values.reshape(-1, 1)
    y = data_jenis['jumlah_kendaraan'].values
    x_flat = data_jenis['tahun'].values

    # Regresi Linier
    model = LinearRegression()
    model.fit(X, y)
    slope = model.coef_[0]
    intercept = model.intercept_
    prediksi_regresi = model.predict(tahun_prediksi.reshape(-1, 1))

    # Interpolasi Newton
    prediksi_newton = newton_interpolation(x_flat, y, tahun_prediksi)

    hasil_prediksi[jenis] = {
        'tahun': tahun_prediksi,
        'regresi': prediksi_regresi,
        'newton': prediksi_newton,
        'slope': slope,
        'intercept': intercept,
        'tahun_aktual': x_flat,
        'jumlah_aktual': y
    }

# 4. Tampilkan hasil prediksi dan perhitungannya per jenis kendaraan
total_regresi = np.zeros_like(tahun_prediksi, dtype=float)
total_newton = np.zeros_like(tahun_prediksi, dtype=float)

for jenis, hasil in hasil_prediksi.items():
    print(f"\nJenis: {jenis}")
    print("Data Aktual:")
    for t, j in zip(hasil['tahun_aktual'], hasil['jumlah_aktual']):
        print(f"  Tahun {t}: {j}")

    print("\nRegresi Linier:")
    print(f"  Persamaan: y = {hasil['slope']:.2f} * x + {hasil['intercept']:.2f}")
    print("  Koefisien (slope):", hasil['slope'])
    print("  Intercept:", hasil['intercept'])
    print("  Perhitungan prediksi:")
    for t, r in zip(hasil['tahun'], hasil['regresi']):
        print(f"    Tahun {t}: y = {hasil['slope']:.2f}*{t} + {hasil['intercept']:.2f} = {int(r)}")

    print("\nPrediksi Interpolasi Newton:")
    for t, n in zip(hasil['tahun'], hasil['newton']):
        print(f"    Tahun {t}: {int(n)}")

    # Akumulasi total
    total_regresi += hasil['regresi']
    total_newton += hasil['newton']

# 5. Prediksi total semua kendaraan
tahun_prediksi = np.arange(2024, 2029)
print("\n==============================")
print("TOTAL SEMUA KENDARAAN (Prediksi)")
print("==============================")
print("\nRegresi Linier:")
for t, r in zip(tahun_prediksi, total_regresi):
    print(f"  Tahun {t}: {int(r)}")
print("\nInterpolasi Newton:")
for t, n in zip(tahun_prediksi, total_newton):
    print(f"  Tahun {t}: {int(n)}")
