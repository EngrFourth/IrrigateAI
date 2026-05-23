# =========================================================
# IRRIGATEAI
# AI-Based Irrigation Water Requirement Forecasting System
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import joblib
import os

from datetime import datetime, timedelta

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="IrrigateAI",
    layout="wide"
)

# =========================================================
# TITLE
# =========================================================

st.title("🌱 IrrigateAI")

st.markdown(
    """
    ### AI-Based Irrigation Water Requirement Forecasting System
    """
)

# =========================================================
# MODEL DIRECTORY
# =========================================================

BASE_DIR = "."

# =========================================================
# LOAD SCALER
# =========================================================

scaler = joblib.load(

    os.path.join(
        BASE_DIR,
        "L8_Scaler.pkl"
    )
)

# =========================================================
# LOAD SVR MODELS
# =========================================================

models = {}

for i in range(1, 15):

    model_path = os.path.join(

        BASE_DIR,

        f"L8_SVR_ETo(t+{i})_Model.pkl"
    )

    models[f"day_{i}"] = (
        joblib.load(model_path)
    )

# =========================================================
# OPENWEATHERMAP API KEY
# =========================================================

API_KEY = "d3a4040d3243bab208807557d0c2f6cd"

# =========================================================
# LOCATION
# =========================================================

LAT = 14.1673
LON = 121.2436

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header(
    "Input Parameters"
)

# =========================================================
# TODAY
# =========================================================

today = datetime.today()

st.sidebar.info(
    f"Today's Date:\n{today.strftime('%B %d, %Y')}"
)

# =========================================================
# LOCATION INFO
# =========================================================

st.sidebar.subheader(
    "Forecast Location"
)

st.sidebar.write(
    "UPLB National Agrometeorological Station"
)

st.sidebar.write(
    "Los Baños, Laguna"
)

# =========================================================
# CROP INFORMATION
# =========================================================

st.sidebar.subheader(
    "Crop Information"
)

crop_type = st.sidebar.selectbox(

    "Crop Type",

    [
        "Lowland Rice"
    ]
)

# =========================================================
# GROWTH STAGE
# =========================================================

growth_stage = st.sidebar.selectbox(

    "Growth Stage (% Total Growth Duration)",

    [
        "0-20%",
        "20-40%",
        "40-70%",
        "70-90%",
        "Harvest"
    ]
)

# =========================================================
# KC VALUES
# =========================================================

kc_database = {

    "0-20%": 0.95,
    "20-40%": 1.05,
    "40-70%": 1.10,
    "70-90%": 1.10,
    "Harvest": 0.61
}

kc = kc_database[growth_stage]

# =========================================================
# SOIL TEXTURE
# =========================================================

soil_texture = st.sidebar.selectbox(

    "Soil Texture",

    [
        "Clay",
        "Silty Clay",
        "Clay Loam",
        "Silty Clay Loam",
        "Sandy Clay Loam",
        "Sandy Loam"
    ]
)

# =========================================================
# PERCOLATION VALUES
# =========================================================

percolation_database = {

    "Clay": 1.25,
    "Silty Clay": 1.50,
    "Clay Loam": 1.75,
    "Silty Clay Loam": 1.75,
    "Sandy Clay Loam": 2.00,
    "Sandy Loam": 4.00
}

percolation = percolation_database[
    soil_texture
]

# =========================================================
# EFFECTIVE RAINFALL PERCENTAGE
# =========================================================

effective_rainfall_percent = st.sidebar.slider(

    "Effective Rainfall Percentage (%)",

    min_value=0,

    max_value=100,

    value=80,

    step=5
)

effective_rainfall_factor = (
    effective_rainfall_percent / 100
)

# =========================================================
# DISPLAY VALUES
# =========================================================

st.sidebar.success(

    f"Crop Coefficient (Kc): {kc:.2f}"
)

st.sidebar.info(

    f"Percolation Rate: {percolation:.2f} mm/day"
)

# =========================================================
# ETO INPUTS
# =========================================================

st.sidebar.subheader(
    "Past 7-Day ETo Inputs"
)

eto_values = []

for i in range(8):

    current_date = (
        today - timedelta(days=(7 - i))
    )

    eto = st.sidebar.number_input(

        f"{current_date.strftime('%b %d, %Y')}",

        min_value=1.0,

        max_value=20.0,

        value=4.5,

        step=0.1,

        key=i
    )

    eto_values.append(eto)

# =========================================================
# GENERATE BUTTON
# =========================================================

generate = st.sidebar.button(
    "Generate Forecast"
)

# =========================================================
# MAIN PROCESS
# =========================================================

if generate:

    # =====================================================
    # INPUT DATAFRAME
    # =====================================================

    input_df = pd.DataFrame(

        [eto_values],

        columns=[

            "ETo(t-7)",
            "ETo(t-6)",
            "ETo(t-5)",
            "ETo(t-4)",
            "ETo(t-3)",
            "ETo(t-2)",
            "ETo(t-1)",
            "ETo(t)"
        ]
    )

    # =====================================================
    # SCALE INPUT
    # =====================================================

    input_scaled = scaler.transform(
        input_df
    )

    # =====================================================
    # PREDICT 14-DAY ETO
    # =====================================================

    eto_predictions = []

    for i in range(1, 15):

        model = models[f"day_{i}"]

        pred = model.predict(
            input_scaled
        )[0]

        eto_predictions.append(pred)

    # =====================================================
    # FETCH 5-DAY RAINFALL FORECAST
    # =====================================================

    rainfall_forecast = []

    try:

        url = (

            f"https://api.openweathermap.org/data/2.5/forecast"

            f"?lat={LAT}"

            f"&lon={LON}"

            f"&appid={API_KEY}"

            f"&units=metric"
        )

        response = requests.get(url)

        data = response.json()

        daily_rain = {}

        for item in data["list"]:

            date = item["dt_txt"].split()[0]

            rain = item.get(
                "rain",
                {}
            ).get(
                "3h",
                0
            )

            if date not in daily_rain:

                daily_rain[date] = 0

            daily_rain[date] += rain

        rainfall_forecast = list(
            daily_rain.values()
        )[:5]

    except:

        st.error(
            "Unable to retrieve rainfall forecast."
        )

        rainfall_forecast = [0] * 5

    # =====================================================
    # EFFECTIVE RAINFALL
    # =====================================================

    effective_rainfall = [

        rainfall_forecast[i]
        *
        effective_rainfall_factor

        for i in range(5)
    ]

    # =====================================================
    # COMPUTE CWR
    # =====================================================

    crop_water_requirement = [

        (
            eto_predictions[i] * kc
        )
        +
        percolation

        for i in range(5)
    ]

    # =====================================================
    # COMPUTE NIR
    # =====================================================

    net_irrigation_requirement = [

        max(

            crop_water_requirement[i]
            -
            effective_rainfall[i],

            0

        )

        for i in range(5)
    ]

    # =====================================================
    # FORECAST DATES
    # =====================================================

    forecast_dates_14 = [

        (
            today + timedelta(days=i)
        ).strftime("%b %d, %Y")

        for i in range(1, 15)
    ]

    forecast_dates_5 = [

        (
            today + timedelta(days=i)
        ).strftime("%b %d, %Y")

        for i in range(1, 6)
    ]

    # =====================================================
    # RESULTS DATAFRAME
    # =====================================================

    results_df = pd.DataFrame({

        "Forecast Date":
            forecast_dates_5,

        "Predicted ETo (mm/day)":
            np.round(
                eto_predictions[:5],
                2
            ),

        "Forecast Rainfall (mm)":
            np.round(
                rainfall_forecast,
                2
            ),

        "Effective Rainfall (mm)":
            np.round(
                effective_rainfall,
                2
            ),

        "Crop Water Requirement (mm/day)":
            np.round(
                crop_water_requirement,
                2
            ),

        "Net Irrigation Requirement (mm/day)":
            np.round(
                net_irrigation_requirement,
                2
            )
    })

    # =====================================================
    # DISPLAY TABLE
    # =====================================================

    st.subheader(
        "📊 5-Day Irrigation Water Requirement Forecast"
    )

    st.dataframe(
        results_df,
        use_container_width=True
    )

    # =====================================================
    # MAIN PLOT
    # =====================================================

    st.subheader(
        "📈 Forecast Visualization"
    )

    fig, ax = plt.subplots(
        figsize=(14,6)
    )

    # =====================================================
    # 14-DAY ETO
    # =====================================================

    ax.plot(

        forecast_dates_14,

        eto_predictions,

        marker='o',

        linewidth=2,

        label="14-Day Predicted ETo"
    )

    # =====================================================
    # FORECAST RAINFALL
    # =====================================================

    ax.plot(

        forecast_dates_5,

        rainfall_forecast,

        marker='o',

        linewidth=2,

        label="5-Day Forecast Rainfall"
    )

    # =====================================================
    # EFFECTIVE RAINFALL
    # =====================================================

    ax.plot(

        forecast_dates_5,

        effective_rainfall,

        marker='o',

        linewidth=2,

        label="5-Day Effective Rainfall"
    )

    # =====================================================
    # CWR
    # =====================================================

    ax.plot(

        forecast_dates_5,

        crop_water_requirement,

        marker='o',

        linewidth=2,

        label="5-Day Crop Water Requirement"
    )

    # =====================================================
    # NIR
    # =====================================================

    ax.plot(

        forecast_dates_5,

        net_irrigation_requirement,

        marker='o',

        linewidth=2,

        label="5-Day Net Irrigation Requirement"
    )

    ax.set_xlabel(
        "Forecast Date"
    )

    ax.set_ylabel(
        "mm/day"
    )

    ax.set_title(
        "Forecast Visualization"
    )

    ax.grid(True)

    ax.legend()

    plt.xticks(rotation=45)

    st.pyplot(fig)

    # =====================================================
    # INDIVIDUAL CHARTS
    # =====================================================

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "14-Day Predicted ETo"
        )

        st.line_chart(

            pd.DataFrame({

                "ETo":
                    eto_predictions
            },

            index=forecast_dates_14)
        )

    with col2:

        st.subheader(
            "5-Day Net Irrigation Requirement"
        )

        st.line_chart(

            pd.DataFrame({

                "Net Irrigation":
                    net_irrigation_requirement
            },

            index=forecast_dates_5)
        )

    # =====================================================
    # DOWNLOAD CSV
    # =====================================================

    csv = results_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(

        label="📥 Download Forecast CSV",

        data=csv,

        file_name="Forecast_Results.csv",

        mime="text/csv"
    )

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    Developed using:
    - Streamlit
    - Support Vector Regression (SVR)
    - OpenWeatherMap API
    - NIA Percolation Values
    - Effective Rainfall Method
    - Crop Coefficient Method
    """
)
