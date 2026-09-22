# 🏡 California Housing Price Prediction (End-to-End ML Pipeline)

An end-to-end Machine Learning web application that predicts median house values in California block groups based on demographic and geographic features. Built using Python, Scikit-Learn, Flask, and HTML/CSS.

---

## 📌 Project Overview

This repository demonstrates a complete, industry-standard machine learning project workflow:
1. **Exploratory Data Analysis (EDA):** Visualizing distributions, detecting outliers, and analyzing correlations.
2. **Data Preprocessing & Feature Scaling:** Standardizing features using `StandardScaler` to optimize model performance.
3. **Model Training & Evaluation:** Training regression models and evaluating performance using metrics such as RMSE and $R^2$ score.
4. **Model Serialization:** Saving the trained model (`model.pkl`) and scaler (`scaling_new.pkl`) with `pickle`.
5. **Web Deployment:** Building a user-friendly web interface using **Flask** to serve real-time predictions.

---

## 📊 Dataset & Features

The model uses the California Housing Dataset based on 1990 U.S. Census data. The features used for prediction include:

| Feature Name | Description |
| :--- | :--- |
| `MedInc` | Median income in block group (in tens of thousands of USD) |
| `HouseAge` | Median house age in block group (years) |
| `AveRooms` | Average number of rooms per household |
| `AveBedrms` | Average number of bedrooms per household |
| `Population` | Total population in block group |
| `AveOccup` | Average number of household members |
| `Latitude` | Block group latitude coordinate |
| `Longitude` | Block group longitude coordinate |

---

## 📁 Repository Structure

```text
california-housing-project/
├── Predicting_House_Prices.ipynb   # Jupyter Notebook for EDA, model training & evaluation
├── app.py                         # Flask web server & prediction routes
├── model.pkl                      # Saved trained Machine Learning model
├── scaling_new.pkl                # Saved StandardScaler object
├── requirements.txt               # Required Python packages
├── README.md                      # Project documentation
├── templates/
│   └── home.html                  # HTML template for web UI
└── plots/                          # Data visualization box plots
    ├── boxPlot.jpg
    ├── boxPlotTestData.jpg
    └── boxPlotTrainData.jpg
```

---

## 🚀 Getting Started

Follow these instructions to run the project on your local machine.

### 1. Prerequisites
Make sure you have **Python 3.8+** installed on your system.

### 2. Clone the Repository
```bash
git clone https://github.com/pankajatd/california-housing-project.git
cd california-housing-project
```

### 3. Set Up a Virtual Environment (Optional but Recommended)
```bash
# On Windows
python -m venv venv
.\venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the Flask Web Application
```bash
python app.py
```

After starting the server, open your web browser and navigate to:
```text
http://127.0.0.1:5000/
```

---

## 💡 How to Use

1. Navigate to `http://127.0.0.1:5000/` in your browser.
2. Enter the feature values into the input form (e.g., Median Income, House Age, Average Rooms, Latitude, Longitude, etc.).
3. Click **Predict**.
4. The predicted median house price will be displayed on the page.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
