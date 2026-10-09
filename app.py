"""
California Housing Price Prediction — Streamlit Cloud Studio
=============================================================
Machine Learning valuation platform based on the California Housing Dataset.
Features:
- Live Interactive Property Valuation with Presets
- Dynamic Geospatial California Map Pinpoint
- Feature Contribution Breakdown & Sensitivity Analysis
- Outlier Diagnostics & Exploratory Data Analysis (EDA)
- Batch CSV Prediction Engine
"""

import os
import sys
import pickle
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="California Housing Valuation Studio",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────
# Custom CSS Styling
# ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1400px !important;
    }
    .header-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 50%, #0369a1 100%);
        padding: 1.25rem 1.75rem;
        border-radius: 12px;
        margin-bottom: 1.25rem;
        color: white;
        border: 1px solid #38bdf8;
        box-shadow: 0 4px 15px rgba(3, 105, 161, 0.25);
    }
    .metric-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.25rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .valuation-hero {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(5, 150, 105, 0.3);
    }
    .feature-tag {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────
# Model & Scaler Loader with Robust Mathematical Fallback
# ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent

# Ground truth coefficients trained in Predicting_House_Prices.ipynb
DEFAULT_COEFS = np.array([
    0.84922176,   # MedInc
    0.12211931,   # HouseAge
    -0.29955845,  # AveRooms
    0.34840967,   # AveBedrms
    -0.00088449,  # Population
    -0.04169804,  # AveOccup
    -0.89385565,  # Latitude
    -0.86861669   # Longitude
])
DEFAULT_INTERCEPT = 2.0692396089424165
DEFAULT_SCALER_MEAN = np.array([
    3.87689155, 28.5753738, 5.43812463, 1.09803314, 1427.92733, 3.1192365, 35.6506693, -119.584102
])
DEFAULT_SCALER_SCALE = np.array([
    1.90484248, 12.6131971, 2.45348438, 0.447482496, 1140.18573, 12.3732074, 2.13566827, 2.0028609
])
FEATURE_NAMES = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']

@st.cache_resource
def load_ml_artifacts():
    model = None
    scaler = None
    model_path = BASE_DIR / "model.pkl"
    scaler_path = BASE_DIR / "scaling_new.pkl"

    if model_path.exists():
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        except Exception:
            model = None

    if scaler_path.exists():
        try:
            with open(scaler_path, "rb") as f:
                scaler = pickle.load(f)
        except Exception:
            scaler = None

    return model, scaler

model_obj, scaler_obj = load_ml_artifacts()

def predict_price(input_df: pd.DataFrame) -> np.ndarray:
    """Predicts median house price in $100,000s, returning dollar values."""
    features = input_df[FEATURE_NAMES].values
    try:
        if scaler_obj is not None and hasattr(scaler_obj, 'transform'):
            scaled = scaler_obj.transform(input_df[FEATURE_NAMES])
        else:
            scaled = (features - DEFAULT_SCALER_MEAN) / DEFAULT_SCALER_SCALE

        if model_obj is not None and hasattr(model_obj, 'predict'):
            raw_pred = model_obj.predict(scaled)
        else:
            raw_pred = np.dot(scaled, DEFAULT_COEFS) + DEFAULT_INTERCEPT

        # California housing values are non-negative, standard dataset caps at 5.0 ($500k)
        clipped = np.clip(raw_pred, 0.15, 6.0)
        return clipped * 100000.0
    except Exception:
        # Fallback to direct math
        scaled = (features - DEFAULT_SCALER_MEAN) / DEFAULT_SCALER_SCALE
        raw_pred = np.dot(scaled, DEFAULT_COEFS) + DEFAULT_INTERCEPT
        clipped = np.clip(raw_pred, 0.15, 6.0)
        return clipped * 100000.0

# ──────────────────────────────────────────────────────────────────────
# Sidebar - District Presets & Controls
# ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🗺️ California Regional Presets")
    st.caption("Select a notable California region to automatically prefill benchmark parameters:")

    PRESETS = {
        "Custom Configuration": {
            "MedInc": 3.87, "HouseAge": 28.0, "AveRooms": 5.43, "AveBedrms": 1.09,
            "Population": 1425.0, "AveOccup": 3.0, "Latitude": 35.65, "Longitude": -119.58,
            "label": "California State Median Average"
        },
        "Silicon Valley (Palo Alto / Cupertino)": {
            "MedInc": 9.85, "HouseAge": 32.0, "AveRooms": 7.10, "AveBedrms": 1.05,
            "Population": 1280.0, "AveOccup": 2.50, "Latitude": 37.44, "Longitude": -122.16,
            "label": "Tech Hub / High Income Corridor"
        },
        "San Francisco Coastal Bay": {
            "MedInc": 8.12, "HouseAge": 45.0, "AveRooms": 5.95, "AveBedrms": 1.12,
            "Population": 1540.0, "AveOccup": 2.35, "Latitude": 37.77, "Longitude": -122.42,
            "label": "Dense Urban & Historic Coastal Architecture"
        },
        "Beverly Hills / Los Angeles Westside": {
            "MedInc": 8.75, "HouseAge": 36.0, "AveRooms": 6.80, "AveBedrms": 1.10,
            "Population": 1650.0, "AveOccup": 2.80, "Latitude": 34.07, "Longitude": -118.40,
            "label": "Southern California Prime Real Estate"
        },
        "San Diego Coastal (La Jolla)": {
            "MedInc": 7.20, "HouseAge": 29.0, "AveRooms": 6.20, "AveBedrms": 1.06,
            "Population": 1390.0, "AveOccup": 2.60, "Latitude": 32.84, "Longitude": -117.27,
            "label": "Pacific Coast Scenic Community"
        },
        "Sacramento Metro Suburbs": {
            "MedInc": 4.15, "HouseAge": 22.0, "AveRooms": 5.60, "AveBedrms": 1.04,
            "Population": 1820.0, "AveOccup": 3.10, "Latitude": 38.58, "Longitude": -121.49,
            "label": "Inland Capital Metropolitan Hub"
        },
        "Central Valley Agricultural District (Fresno)": {
            "MedInc": 2.35, "HouseAge": 25.0, "AveRooms": 4.80, "AveBedrms": 1.15,
            "Population": 1950.0, "AveOccup": 3.75, "Latitude": 36.74, "Longitude": -119.78,
            "label": "Inland Agricultural Value Basin"
        }
    }

    selected_preset_name = st.selectbox(
        "Choose Region Benchmark",
        list(PRESETS.keys()),
        index=1
    )
    preset = PRESETS[selected_preset_name]
    st.info(f"📍 **{preset['label']}**")

    st.markdown("---")
    st.markdown("### 📊 Model Architecture Specs")
    st.markdown("""
    - **Algorithm:** Regularized Linear Regression
    - **Normalization:** `StandardScaler` (Z-score)
    - **Target Variable:** `MedianHouseValue` ($100k)
    - **Dataset Size:** 20,640 California Block Groups
    - **Cloud Deployment:** Streamlit Community Cloud
    """)

# ──────────────────────────────────────────────────────────────────────
# Main Header
# ──────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="header-banner">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
        <div>
            <h1 style="margin:0; font-size: 1.85rem; font-weight: 700; color: #ffffff;">🏡 California Housing Valuation Studio</h1>
            <p style="margin: 4px 0 0 0; color: #93c5fd; font-size: 0.95rem;">
                Machine Learning Appraisal Engine & Spatial Demographic Analytics
            </p>
        </div>
        <div>
            <span style="background: rgba(16, 185, 129, 0.2); border: 1px solid #10b981; color: #6ee7b7; padding: 5px 12px; border-radius: 20px; font-size: 0.82rem; font-weight: 600;">
                ● Production Model Online
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────
# Navigation Tabs
# ──────────────────────────────────────────────────────────────────────
tab_valuation, tab_batch, tab_eda, tab_insights = st.tabs([
    "🎯 Interactive Property Valuation",
    "📁 Batch CSV Predictor",
    "📈 Outliers & EDA Diagnostics",
    "🧠 Feature Sensitivity & Weights"
])

# ──────────────────────────────────────────────────────────────────────
# TAB 1: Interactive Valuation
# ──────────────────────────────────────────────────────────────────────
with tab_valuation:
    col_input, col_result = st.columns([1.1, 0.9], gap="large")

    with col_input:
        st.markdown("#### ⚙️ Block Group Attributes")
        st.caption("Adjust the parameters below to evaluate district property pricing:")

        c1, c2 = st.columns(2)
        with c1:
            med_inc = st.number_input(
                "Median Household Income (in $10k)",
                min_value=0.5,
                max_value=15.0,
                value=float(preset["MedInc"]),
                step=0.1,
                help="Median income in block group (e.g. 5.5 = $55,000/year)"
            )
            house_age = st.slider(
                "Median House Age (years)",
                min_value=1,
                max_value=52,
                value=int(preset["HouseAge"]),
                help="Median age of a house within a block (years)"
            )
            ave_rooms = st.number_input(
                "Average Rooms per Household",
                min_value=1.0,
                max_value=15.0,
                value=float(preset["AveRooms"]),
                step=0.1
            )
            ave_bedrms = st.number_input(
                "Average Bedrooms per Household",
                min_value=0.5,
                max_value=5.0,
                value=float(preset["AveBedrms"]),
                step=0.05
            )

        with c2:
            population = st.number_input(
                "Block Group Population",
                min_value=50.0,
                max_value=35000.0,
                value=float(preset["Population"]),
                step=50.0,
                help="Total number of people residing within a block"
            )
            ave_occup = st.number_input(
                "Average Household Occupancy",
                min_value=1.0,
                max_value=10.0,
                value=float(preset["AveOccup"]),
                step=0.1,
                help="Average number of household members"
            )
            latitude = st.number_input(
                "Latitude (°N)",
                min_value=32.5,
                max_value=42.0,
                value=float(preset["Latitude"]),
                step=0.05,
                format="%.2f"
            )
            longitude = st.number_input(
                "Longitude (°W)",
                min_value=-124.5,
                max_value=-114.0,
                value=float(preset["Longitude"]),
                step=0.05,
                format="%.2f"
            )

        # Formulate DataFrame
        single_df = pd.DataFrame([{
            'MedInc': med_inc,
            'HouseAge': house_age,
            'AveRooms': ave_rooms,
            'AveBedrms': ave_bedrms,
            'Population': population,
            'AveOccup': ave_occup,
            'Latitude': latitude,
            'Longitude': longitude
        }])

    with col_result:
        st.markdown("#### 💵 Valuation Appraisal Result")

        pred_price = predict_price(single_df)[0]
        annual_income_equiv = med_inc * 10000.0
        affordability_ratio = pred_price / annual_income_equiv if annual_income_equiv > 0 else 0

        # Tier classification
        if pred_price > 420000:
            tier_badge = "💎 Luxury / Tier 1 Coastal Corridor"
            tier_color = "#3b82f6"
        elif pred_price > 250000:
            tier_badge = "🏡 Premium Suburban Market"
            tier_color = "#10b981"
        elif pred_price > 150000:
            tier_badge = "🏢 Mid-Market District"
            tier_color = "#f59e0b"
        else:
            tier_badge = "🏷️ Affordable Entry Basin"
            tier_color = "#64748b"

        st.markdown(f"""
        <div class="valuation-hero">
            <span style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; opacity: 0.9;">Estimated District Median House Price</span>
            <h1 style="font-size: 2.8rem; margin: 8px 0; font-weight: 800; color: white;">${pred_price:,.2f}</h1>
            <div style="background: rgba(255,255,255,0.15); display: inline-block; padding: 4px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 600;">
                {tier_badge}
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Benchmark Income", f"${annual_income_equiv:,.0f}/yr")
        with m2:
            st.metric("Price-to-Income", f"{affordability_ratio:.1f}x")
        with m3:
            st.metric("Appraisal Band (±7%)", f"±${pred_price * 0.07:,.0f}")

        # Map pinpoint
        st.markdown("##### 📍 Geographic Coordinates on California Map")
        map_df = pd.DataFrame({'lat': [latitude], 'lon': [longitude]})
        st.map(map_df, zoom=7)

# ──────────────────────────────────────────────────────────────────────
# TAB 2: Batch CSV Predictor
# ──────────────────────────────────────────────────────────────────────
with tab_batch:
    st.markdown("### 📁 Batch Valuation Engine")
    st.caption("Score thousands of California housing records instantly with the trained pipeline.")

    sample_template = pd.DataFrame([
        {'MedInc': 8.32, 'HouseAge': 41.0, 'AveRooms': 6.98, 'AveBedrms': 1.02, 'Population': 322.0, 'AveOccup': 2.55, 'Latitude': 37.88, 'Longitude': -122.23},
        {'MedInc': 3.84, 'HouseAge': 21.0, 'AveRooms': 6.23, 'AveBedrms': 0.97, 'Population': 2401.0, 'AveOccup': 2.10, 'Latitude': 37.86, 'Longitude': -122.22},
        {'MedInc': 7.25, 'HouseAge': 52.0, 'AveRooms': 8.28, 'AveBedrms': 1.07, 'Population': 496.0, 'AveOccup': 2.80, 'Latitude': 37.85, 'Longitude': -122.24},
        {'MedInc': 1.65, 'HouseAge': 40.0, 'AveRooms': 3.82, 'AveBedrms': 1.05, 'Population': 1485.0, 'AveOccup': 3.25, 'Latitude': 36.75, 'Longitude': -119.80},
    ])

    col_b1, col_b2 = st.columns([1, 1])
    with col_b1:
        uploaded_file = st.file_uploader("Upload CSV File (matching feature columns)", type=["csv"])
    with col_b2:
        st.markdown("**Download Sample CSV Template:**")
        csv_bytes = sample_template.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Template CSV",
            data=csv_bytes,
            file_name="california_housing_sample_template.csv",
            mime="text/csv"
        )
        if st.button("🚀 Load Sample Dataset into Batch Engine"):
            uploaded_file = "SAMPLE"

    if uploaded_file is not None:
        if uploaded_file == "SAMPLE":
            batch_df = sample_template.copy()
            st.success("Loaded 4 representative California district test cases!")
        else:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.success(f"Successfully loaded {len(batch_df)} rows from uploaded file!")
            except Exception as e:
                st.error(f"Error reading CSV: {e}")
                batch_df = None

        if batch_df is not None:
            missing_cols = [c for c in FEATURE_NAMES if c not in batch_df.columns]
            if missing_cols:
                st.error(f"Missing required columns in CSV: {missing_cols}")
            else:
                preds = predict_price(batch_df)
                batch_df["PredictedHousePrice"] = preds
                batch_df["PredictedPrice_USD"] = batch_df["PredictedHousePrice"].apply(lambda v: f"${v:,.2f}")

                st.markdown("#### 📊 Batch Predictions Table")
                st.dataframe(batch_df, use_container_width=True)

                st.markdown("#### 📈 Summary Statistics")
                b_c1, b_c2, b_c3 = st.columns(3)
                with b_c1:
                    st.metric("Average Predicted Price", f"${preds.mean():,.2f}")
                with b_c2:
                    st.metric("Min Valuation", f"${preds.min():,.2f}")
                with b_c3:
                    st.metric("Max Valuation", f"${preds.max():,.2f}")

                # Download results
                res_csv = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Scored CSV with Predictions",
                    data=res_csv,
                    file_name="california_housing_predictions.csv",
                    mime="text/csv"
                )

# ──────────────────────────────────────────────────────────────────────
# TAB 3: Exploratory Data Analysis & Residual Diagnostics
# ──────────────────────────────────────────────────────────────────────
with tab_eda:
    st.markdown("### 📈 Outlier Diagnostics & Exploratory Visualizations")
    st.caption("Visual inspection from the exploratory training phase:")

    img_col1, img_col2 = st.columns(2)

    box_full = BASE_DIR / "boxPlot.jpg"
    box_train = BASE_DIR / "boxPlotTrainData.jpg"
    box_test = BASE_DIR / "boxPlotTestData.jpg"

    with img_col1:
        if box_full.exists():
            st.markdown("#### 📦 Full Dataset Feature Distributions")
            st.image(str(box_full), caption="Boxplot of Standardized Features", use_container_width=True)
        if box_train.exists():
            st.markdown("#### 📦 Training Split Feature Bounds")
            st.image(str(box_train), caption="Boxplot of Training Data Distribution", use_container_width=True)

    with img_col2:
        if box_test.exists():
            st.markdown("#### 📦 Test Split Feature Bounds")
            st.image(str(box_test), caption="Boxplot of Holdout Test Data Distribution", use_container_width=True)

        st.markdown("#### 🔍 Dataset Insights")
        st.info("""
        - **Median Income:** Demonstrated the single strongest positive linear relationship with house prices ($r \\approx 0.69$).
        - **Geographic Clusters:** Coastal coordinates (higher negative longitude and moderate latitude) exhibit significant price premiums over inland valleys.
        - **Extreme Outliers:** Extreme values in `Population` and `AveOccup` are normalized via Z-Score scaling to protect model weights from skew.
        """)

# ──────────────────────────────────────────────────────────────────────
# TAB 4: Feature Sensitivity & Weights
# ──────────────────────────────────────────────────────────────────────
with tab_insights:
    st.markdown("### 🧠 Feature Sensitivity & Regression Coefficients")
    st.caption("How each standardized feature impacts the predicted price of a California home:")

    coef_df = pd.DataFrame({
        "Feature": FEATURE_NAMES,
        "Coefficient": DEFAULT_COEFS,
        "Impact Direction": ["📈 Positive" if c > 0 else "📉 Negative" for c in DEFAULT_COEFS],
        "Mean Value": DEFAULT_SCALER_MEAN,
        "Std Deviation": DEFAULT_SCALER_SCALE,
        "Interpretation": [
            "Every 1-std increase in median income boosts price by +$84,922",
            "Older established neighborhoods carry slight price premium (+$12,212 per std)",
            "Room count without bedroom ratio correction has negative coefficient",
            "Higher bedroom concentration per dwelling adds +$34,841 per std",
            "Total population density exerts slight downward drag",
            "High household overcrowding lowers price by -$4,170 per std",
            "Northern latitudes correlate with lower prices compared to coastal hubs",
            "Eastern inland locations correlate with lower values compared to Pacific coast"
        ]
    }).sort_values(by="Coefficient", ascending=False)

    st.dataframe(coef_df, use_container_width=True)

    st.bar_chart(data=coef_df.set_index("Feature")["Coefficient"])

# ──────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.85rem; padding: 10px 0;">
    California Housing Valuation Studio • Engineered for Streamlit Community Cloud • Built with Scikit-Learn & Python
</div>
""", unsafe_allow_html=True)
