# YENİ: 'redirect', 'url_for' ve 'request' modüllerini içe aktarın
from flask import Flask, render_template, request, redirect, url_for
import pickle
import pandas as pd

app = Flask(__name__)

# Kayıtlı modeli yükle
with open("weather_model.pkl", "rb") as file:
    model = pickle.load(file)

# --- DEĞİŞİKLİK 1: Ana sayfa rotası (/) artık sonuçları da alabilir ---
@app.route('/')
def home():
    # URL'den gelen parametreleri al (örn: /?result_text=SUNNY)
    result_text = request.args.get('result_text')
    result_icon = request.args.get('result_icon')
    temp = request.args.get('temp')
    hum = request.args.get('hum')
    wind = request.args.get('wind')
    
    # Bu parametreleri şablona gönder
    return render_template('index.html',
                           result_text=result_text,
                           result_icon=result_icon,
                           temp=temp, hum=hum, wind=wind)

# --- DEĞİŞİKLİK 2: /predict rotası artık 'render_template' değil, 'redirect' yapıyor ---
@app.route('/predict', methods=['POST'])
def predict():
    # Formdan verileri al
    temp = float(request.form['temperature'])
    hum = float(request.form['humidity'])
    wind = float(request.form['wind_speed'])

    new_data = pd.DataFrame({
        "temperature": [temp], "humidity": [hum], "wind_speed": [wind]
    })

    # Tahmini yap
    prediction = model.predict(new_data)[0]
    
    # Sonuçları belirle
    if prediction == 0:
        result_text, result_icon = "RAINY", "bi-cloud-drizzle-fill"
    elif prediction == 1:
        result_text, result_icon = "SUNNY", "bi-sun-fill"
    elif prediction == 2:
        result_text, result_icon = "CLOUDY", "bi-cloud-fill"
    elif prediction == 3:
        result_text, result_icon = "WINDY", "bi-wind"
    else:
        result_text, result_icon = "Bilinmiyor", "bi-question-lg"

    # YÖNLENDİRME: Sonuçları ve girilen değerleri URL'e ekleyerek 'home' rotasına yönlendir
    return redirect(url_for('home', 
                            result_text=result_text, 
                            result_icon=result_icon,
                            temp=temp, 
                            hum=hum, 
                            wind=wind))

if __name__ == "__main__":
    app.run(debug=True, port=5002)