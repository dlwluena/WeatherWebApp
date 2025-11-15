# Weather Prediction Web App

A beginner-friendly machine learning web application built for the "ESOF325-Introduction to Artificial Intelligence" course.

This project uses a **Logistic Regression** model trained with Scikit-learn to predict the weather as **Sunny, Rainy, Cloudy,** or **Windy** based on user-provided inputs for Temperature, Humidity, and Wind Speed. The model is served as a web application using the Flask framework.
<img width="1470" height="829" alt="weather" src="https://github.com/user-attachments/assets/55513020-aff7-4bda-9e35-66202670b2f2" />


---

## Key Features

* **Multi-Class Prediction:** Classifies weather into 4 categories (Sunny, Rainy, Cloudy, Windy).
* **Simple ML Model:** Uses a Logistic Regression model built with Scikit-learn and Pandas.
* **Web Interface:** A clean, modern, and responsive UI built with **Bootstrap 5** and custom CSS.
* **Flask Backend:** A lightweight Python web server handles user inputs and model predictions.
* **Professional UX:** Implements the **Post/Redirect/Get (PRG)** design pattern to prevent "Form Resubmission" errors on page refresh.

## Tech Stack

* **Backend:** Python, Flask
* **Machine Learning:** Scikit-learn, Pandas, NumPy
* **Frontend:** HTML, CSS, Bootstrap 5

---

## How to Run Locally

Follow these steps to get the project up and running on your local machine.

### 1. Prerequisites

Make sure you have **Python 3** and **Git** installed on your system.

### 2. Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/](https://github.com/)[YOUR-USERNAME]/WeatherWebApp.git
    cd WeatherWebApp
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # On macOS/Linux
    python3 -m venv .venv
    source .venv/bin/activate
    
    # On Windows
    python -m venv .venv
    .\.venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    *(This project uses `requirements.txt`. If you haven't created it, run `pip freeze > requirements.txt` first.)*
    ```bash
    pip install -r requirements.txt
    ```

4.  **Train the ML Model:**
    This is a crucial step. It runs the `train_model.py` script to create the `weather_model.pkl` file that the web app needs.
    ```bash
    python3 train_model.py
    ```

5.  **Run the Flask Web Server:**
    ```bash
    python3 app.py
    ```

6.  **Open the App:**
    Open your web browser and navigate to:
    **http://127.0.0.1:5002**

You can now enter weather data and get live predictions!
