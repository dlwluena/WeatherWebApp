import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle

print("--- Çok Sınıflı Model Eğitim Süreci Başladı ---")

# --- ADIM 3: Veri Seti Oluşturma (4 KATEGORİLİ) ---
np.random.seed(42)
data = {
    "temperature": np.random.randint(5, 35, 100),
    "humidity": np.random.randint(30, 100, 100),
    "wind_speed": np.random.randint(5, 50, 100) # Rüzgar aralığını artırdık
}
df = pd.DataFrame(data)

# Yeni Kategoriler:
# 0 = Yağmurlu
# 1 = Güneşli
# 2 = Bulutlu (Cloudy)
# 3 = Rüzgarlı (Windy)

# Koşulları ve seçimleri tanımlayalım (np.select ile daha kolay)
conditions = [
    (df["wind_speed"] > 35),                                     # 1. Öncelik: Çok rüzgarlıysa = RÜZGARLI
    (df["temperature"] > 25) & (df["humidity"] < 60),           # 2. Öncelik: Sıcak ve nemsizse = GÜNEŞLİ
    (df["humidity"] > 80) & (df["temperature"] < 15),           # 3. Öncelik: Soğuk ve çok nemliyse = YAĞMURLU
    (df["humidity"] > 65) | (df["temperature"].between(15, 25)) # 4. Öncelik: Diğer nemli veya ılık hava = BULUTLU
]

# Koşullara karşılık gelen değerler (kategori numaraları)
choices = [3, 1, 0, 2]

# Hiçbir koşul uymazsa varsayılan olarak 0 (Yağmurlu) ata
df["weather"] = np.select(conditions, choices, default=0)

print("4 kategorili (Yağmurlu, Güneşli, Bulutlu, Rüzgarlı) veri seti oluşturuldu.")
print("Veri dağılımı:")
print(df["weather"].value_counts())


# --- ADIM 5: Veriyi Bölme ve Modeli Eğitme ---
X = df[["temperature", "humidity", "wind_speed"]]
y = df["weather"] # y artık 0, 1, 2, 3 değerlerini içeriyor

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Eğitim için {len(X_train)}, test için {len(X_test)} örnek ayrıldı.")

# Lojistik Regresyon çok sınıflı (multiclass) durumu otomatik olarak algılar
model = LogisticRegression(max_iter=1000) # Daha iyi yakınsama için max_iter ekledik
model.fit(X_train, y_train)
print("Çok sınıflı model başarıyla eğitildi.")


# --- ADIM 6: Modelin Performansını Değerlendirme ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\n--- Model Değerlendirme Raporu ---")
print(f"Model Doğruluğu (Accuracy): {accuracy:.2f}")
print("Sınıflandırma Raporu (0=Yağmurlu, 1=Güneşli, 2=Bulutlu, 3=Rüzgarlı):")
print(classification_report(y_test, y_pred))


# --- ADIM 8: Modelin Kaydedilmesi ---
model_filename = "weather_model.pkl"
with open(model_filename, "wb") as file:
    pickle.dump(model, file)

print(f"--- Süreç Tamamlandı ---")
print(f"Yeni model '{model_filename}' dosyasına kaydedildi.")