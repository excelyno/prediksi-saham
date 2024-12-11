import yfinance as yf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Fungsi untuk mengambil data saham
def fetch_stock_data(symbol, start_date, end_date):
    try:
        data = yf.download(symbol, start=start_date, end=end_date)
        if data.empty:
            return None
        return data['Close']
    except Exception as e:
        print(f"Error mengambil data: {e}")
        return None

# Fungsi untuk menyiapkan data untuk prediksi
def prepare_data(prices, look_back=5):
    if len(prices) <= look_back:
        print("Data terlalu sedikit untuk melakukan prediksi")
        return None, None
    X, y = [], []
    for i in range(len(prices) - look_back):
        # Reshape data menjadi 2D array
        X.append(prices[i:i + look_back].reshape(-1))
        y.append(prices[i + look_back])
    return np.array(X), np.array(y)

# Visualisasi hasil prediksi
def plot_results(dates, actual_prices, predicted_prices, test_start_idx):
    plt.figure(figsize=(12, 6))
    
    # Plot harga aktual untuk seluruh periode
    plt.plot(dates, actual_prices, label="Harga Aktual", color='blue')
    
    # Plot harga prediksi hanya untuk periode testing
    test_dates = dates[test_start_idx:]
    plt.plot(test_dates, predicted_prices, label="Harga Prediksi", linestyle="--", color='red')
    
    plt.title("Prediksi Harga Saham")
    plt.xlabel("Tanggal")
    plt.ylabel("Harga")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Main Program
if __name__ == "__main__":
    try:
        # Masukkan kode saham dan periode data
        symbol = input("Masukkan kode saham (contoh: AAPL): ").upper()
        start_date = "2023-01-01"
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        # Mengambil data saham
        prices = fetch_stock_data(symbol, start_date, end_date)
        if prices is None:
            print("Gagal mengambil data. Periksa simbol saham atau koneksi internet.")
            exit()

        # Menyiapkan data untuk model
        prices_array = prices.values  # Konversi ke array
        X, y = prepare_data(prices_array)
        
        if X is None or y is None:
            exit()

        # Split data menjadi training dan testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Model prediksi (Linear Regression)
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        # Prediksi harga
        predictions = model.predict(X_test)
        
        # Evaluasi model
        score = model.score(X_test, y_test)
        print(f"Skor R^2: {score:.2f}")
        
        # Menghitung index awal data testing
        test_start_idx = len(prices_array) - len(predictions)
        
        # Plot hasil prediksi
        dates = prices.index  # Menggunakan index datetime dari pandas
        plot_results(dates, prices_array, predictions, test_start_idx)
        
    except Exception as e:
        print(f"Terjadi kesalahan: {e}")
