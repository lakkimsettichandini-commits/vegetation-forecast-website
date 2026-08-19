import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="AI Vegetation Forecast",
    page_icon="🌱",
    layout="wide"
)


# =====================================================
# TITLE
# =====================================================

st.title("🌱 AI-Based Geospatial Vegetation Forecast")

st.write(
    "AI-powered forecasting of future vegetation "
    "conditions using geospatial and environmental data."
)

st.divider()


# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():

    return pd.read_csv("vegetation_data.csv.csv")


df = load_data()


# =====================================================
# CHECK REQUIRED COLUMNS
# =====================================================

required_columns = [
    "City",
    "Latitude",
    "Longitude",
    "Year",
    "Month",
    "NDVI",
    "NDBI",
    "NDWI",
    "Elevation (m)",
    "LST (°C)"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:

    st.error(
        "Missing columns in your CSV: "
        + ", ".join(missing_columns)
    )

    st.stop()


# =====================================================
# CLEAN CITY
# =====================================================

df["City"] = (
    df["City"]
    .astype(str)
    .str.strip()
)


# =====================================================
# CLEAN YEAR
# =====================================================

df["Year"] = pd.to_numeric(
    df["Year"],
    errors="coerce"
)


# =====================================================
# CONVERT MONTH TO NUMBER
# =====================================================

def convert_month(value):

    if pd.isna(value):
        return np.nan

    # If already numeric
    try:

        number = float(value)

        if 1 <= number <= 12:
            return int(number)

    except:

        pass


    # Convert text month names
    month_text = str(value).strip().lower()

    month_map = {

        "january": 1,
        "jan": 1,

        "february": 2,
        "feb": 2,

        "march": 3,
        "mar": 3,

        "april": 4,
        "apr": 4,

        "may": 5,

        "june": 6,
        "jun": 6,

        "july": 7,
        "jul": 7,

        "august": 8,
        "aug": 8,

        "september": 9,
        "sep": 9,
        "sept": 9,

        "october": 10,
        "oct": 10,

        "november": 11,
        "nov": 11,

        "december": 12,
        "dec": 12
    }

    return month_map.get(
        month_text,
        np.nan
    )


df["Month"] = df["Month"].apply(
    convert_month
)


# =====================================================
# CONVERT NUMERIC DATA
# =====================================================

numeric_columns = [

    "Latitude",
    "Longitude",
    "Year",
    "Month",
    "NDVI",
    "NDBI",
    "NDWI",
    "Elevation (m)",
    "LST (°C)"

]

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# =====================================================
# CREATE DATE
# =====================================================

df["Date"] = pd.to_datetime(
    dict(
        year=df["Year"],
        month=df["Month"],
        day=1
    ),
    errors="coerce"
)


# =====================================================
# SORT DATA
# =====================================================

df = df.sort_values(
    ["City", "Date"],
    na_position="last"
).reset_index(drop=True)


# =====================================================
# CREATE FUTURE NDVI
# =====================================================

df["Future_NDVI"] = (
    df.groupby("City")["NDVI"]
    .shift(-1)
)


# =====================================================
# MODEL FEATURES
# =====================================================

features = [

    "Latitude",
    "Longitude",
    "NDBI",
    "NDWI",
    "Elevation (m)",
    "LST (°C)",
    "NDVI"

]


# =====================================================
# CREATE MODEL DATA
# =====================================================

df_model = df.dropna(
    subset=["Future_NDVI"]
).copy()


# =====================================================
# FILL MISSING FEATURE VALUES
# =====================================================

for column in features:

    df_model[column] = pd.to_numeric(
        df_model[column],
        errors="coerce"
    )

    median_value = (
        df_model[column].median()
    )

    if pd.isna(median_value):

        median_value = 0

    df_model[column] = (
        df_model[column]
        .fillna(median_value)
    )


# =====================================================
# CHECK MODEL DATA
# =====================================================

if len(df_model) < 10:

    st.error(
        f"""
Only {len(df_model)} valid records are available
for training.

Please check that each city has multiple NDVI
records for different months.
"""
    )

    st.write(
        "Total CSV rows:",
        len(df)
    )

    st.write(
        "Unique cities:",
        df["City"].nunique()
    )

    st.write(
        "Rows with Future NDVI:",
        df["Future_NDVI"].notna().sum()
    )

    st.stop()


# =====================================================
# FEATURES AND TARGET
# =====================================================

X = df_model[features]

y = df_model["Future_NDVI"]


# =====================================================
# TRAIN / TEST SPLIT
# =====================================================

split_point = int(
    len(df_model) * 0.80
)


X_train = X.iloc[:split_point]

X_test = X.iloc[split_point:]


y_train = y.iloc[:split_point]

y_test = y.iloc[split_point:]


# =====================================================
# RANDOM FOREST
# =====================================================

model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)


model.fit(
    X_train,
    y_train
)


# =====================================================
# PREDICTION
# =====================================================

y_pred = model.predict(
    X_test
)


# =====================================================
# MODEL PERFORMANCE
# =====================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)


# =====================================================
# FORECAST DATA
# =====================================================

forecast = X_test.copy()


forecast["City"] = df_model.loc[
    X_test.index,
    "City"
].values


forecast["Year"] = df_model.loc[
    X_test.index,
    "Year"
].values


forecast["Month"] = df_model.loc[
    X_test.index,
    "Month"
].values


forecast["Actual_Future_NDVI"] = (
    y_test.values
)


forecast["Predicted_Future_NDVI"] = (
    y_pred
)


forecast["NDVI_Change"] = (
    forecast["Predicted_Future_NDVI"]
    - forecast["NDVI"]
)


# =====================================================
# CLASSIFICATION
# =====================================================

def classify(change):

    if change > 0.05:

        return "Increasing"

    elif change < -0.05:

        return "Decreasing"

    else:

        return "Stable"


forecast["Vegetation_Status"] = (
    forecast["NDVI_Change"]
    .apply(classify)
)


# =====================================================
# SUMMARY
# =====================================================

total = len(forecast)

increasing = (
    forecast["Vegetation_Status"]
    == "Increasing"
).sum()

stable = (
    forecast["Vegetation_Status"]
    == "Stable"
).sum()

decreasing = (
    forecast["Vegetation_Status"]
    == "Decreasing"
).sum()


# =====================================================
# SUMMARY CARDS
# =====================================================

st.subheader(
    "📊 Forecast Summary"
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total",
    total
)

col2.metric(
    "🟢 Increasing",
    increasing
)

col3.metric(
    "🟡 Stable",
    stable
)

col4.metric(
    "🔴 Decreasing",
    decreasing
)


# =====================================================
# LOCATION FORECAST
# =====================================================

st.divider()

st.subheader(
    "📍 Location Forecast"
)


cities = sorted(
    df["City"]
    .dropna()
    .unique()
)


selected_city = st.selectbox(
    "Select a City",
    cities
)


# =====================================================
# CITY DATA
# =====================================================

city_data = df[
    df["City"]
    .astype(str)
    .str.strip()
    .str.lower()
    ==
    selected_city
    .strip()
    .lower()
].copy()


# =====================================================
# CITY PREDICTION
# =====================================================

if len(city_data) > 0:

    city_data = city_data.sort_values(
        "Date",
        na_position="last"
    )

    latest = city_data.iloc[-1]


    # -------------------------------------------------
    # PREPARE INPUT
    # -------------------------------------------------

    input_data = pd.DataFrame({

        "Latitude": [
            latest["Latitude"]
        ],

        "Longitude": [
            latest["Longitude"]
        ],

        "NDBI": [
            latest["NDBI"]
        ],

        "NDWI": [
            latest["NDWI"]
        ],

        "Elevation (m)": [
            latest["Elevation (m)"]
        ],

        "LST (°C)": [
            latest["LST (°C)"]
        ],

        "NDVI": [
            latest["NDVI"]
        ]

    })


    # Fill missing values
    for column in features:

        median_value = (
            df_model[column].median()
        )

        if pd.isna(median_value):

            median_value = 0

        input_data[column] = (
            pd.to_numeric(
                input_data[column],
                errors="coerce"
            )
            .fillna(median_value)
        )


    # -------------------------------------------------
    # PREDICTION
    # -------------------------------------------------

    predicted_ndvi = model.predict(
        input_data
    )[0]


    current_ndvi = latest["NDVI"]


    if pd.isna(current_ndvi):

        current_ndvi = input_data[
            "NDVI"
        ].iloc[0]


    ndvi_change = (
        predicted_ndvi
        - current_ndvi
    )


    vegetation_status = classify(
        ndvi_change
    )


    # =================================================
    # LOCATION DETAILS
    # =================================================

    st.write(
        f"**City:** {selected_city}"
    )


    if pd.notna(latest["Year"]):

        st.write(
            f"**Current Year:** "
            f"{int(latest['Year'])}"
        )

    else:

        st.write(
            "**Current Year:** Not available"
        )


    if pd.notna(latest["Month"]):

        st.write(
            f"**Current Month:** "
            f"{int(latest['Month'])}"
        )

    else:

        st.write(
            "**Current Month:** Not available"
        )


    # =================================================
    # METRICS
    # =================================================

    col1, col2, col3 = st.columns(3)


    col1.metric(
        "Current NDVI",
        round(
            current_ndvi,
            4
        )
    )


    col2.metric(
        "Predicted Future NDVI",
        round(
            predicted_ndvi,
            4
        )
    )


    col3.metric(
        "NDVI Change",
        round(
            ndvi_change,
            4
        )
    )


    # =================================================
    # STATUS
    # =================================================

    if vegetation_status == "Increasing":

        st.success(
            "🌱 Vegetation Status: Increasing"
        )

    elif vegetation_status == "Decreasing":

        st.error(
            "⚠️ Vegetation Status: Decreasing"
        )

    else:

        st.info(
            "🌿 Vegetation Status: Stable"
        )


else:

    st.warning(
        "No data found for this city."
    )


# =====================================================
# MODEL PERFORMANCE
# =====================================================

st.divider()

st.subheader(
    "🤖 AI Model Performance"
)


col1, col2, col3 = st.columns(3)


col1.metric(
    "MAE",
    round(mae, 4)
)

col2.metric(
    "RMSE",
    round(rmse, 4)
)

col3.metric(
    "R² Score",
    round(r2, 4)
)


# =====================================================
# VEGETATION BAR CHART
# =====================================================

st.divider()

st.subheader(
    "🌿 Vegetation Forecast"
)


labels = [
    "Increasing",
    "Stable",
    "Decreasing"
]


values = [
    increasing,
    stable,
    decreasing
]


fig1, ax1 = plt.subplots(
    figsize=(8, 5)
)


ax1.bar(
    labels,
    values
)


ax1.set_xlabel(
    "Vegetation Status"
)

ax1.set_ylabel(
    "Number of Observations"
)

ax1.set_title(
    "Future Vegetation Forecast"
)


st.pyplot(fig1)


# =====================================================
# MAP
# =====================================================

st.divider()

st.subheader(
    "🗺️ Geospatial Forecast Map"
)


fig2, ax2 = plt.subplots(
    figsize=(10, 7)
)


for status in [
    "Increasing",
    "Stable",
    "Decreasing"
]:

    subset = forecast[
        forecast["Vegetation_Status"]
        == status
    ]


    ax2.scatter(
        subset["Longitude"],
        subset["Latitude"],
        label=status,
        s=40,
        alpha=0.7
    )


ax2.set_xlabel(
    "Longitude"
)

ax2.set_ylabel(
    "Latitude"
)

ax2.set_title(
    "AI-Based Future Vegetation Forecast"
)

ax2.legend()

ax2.grid(True)


st.pyplot(fig2)


# =====================================================
# ACTUAL VS PREDICTED
# =====================================================

st.divider()

st.subheader(
    "📈 Actual vs Predicted Future NDVI"
)


fig3, ax3 = plt.subplots(
    figsize=(12, 5)
)


number_to_plot = min(
    100,
    len(y_test)
)


ax3.plot(
    y_test.iloc[:number_to_plot].values,
    label="Actual"
)


ax3.plot(
    y_pred[:number_to_plot],
    label="Predicted"
)


ax3.set_xlabel(
    "Test Sample"
)

ax3.set_ylabel(
    "NDVI"
)

ax3.set_title(
    "Actual vs Predicted Future NDVI"
)

ax3.legend()

ax3.grid(True)


st.pyplot(fig3)


# =====================================================
# FORECAST TABLE
# =====================================================

st.divider()

st.subheader(
    "📋 Forecast Results"
)


display_columns = [

    "City",
    "Year",
    "Month",
    "Latitude",
    "Longitude",
    "NDVI",
    "Predicted_Future_NDVI",
    "NDVI_Change",
    "Vegetation_Status"

]


table = forecast[
    display_columns
].copy()


# Make Year integer where available
table["Year"] = table["Year"].apply(
    lambda x:
    int(x)
    if pd.notna(x)
    else "N/A"
)


# Make Month integer where available
table["Month"] = table["Month"].apply(
    lambda x:
    int(x)
    if pd.notna(x)
    else "N/A"
)


st.dataframe(
    table,
    use_container_width=True
)


# =====================================================
# DOWNLOAD
# =====================================================

csv = forecast.to_csv(
    index=False
)


st.download_button(
    "⬇️ Download Forecast Results",
    csv,
    "vegetation_forecast_results.csv",
    "text/csv"
)


# =====================================================
# FINAL MESSAGE
# =====================================================

st.success(
    "🌱 AI Vegetation Forecast completed successfully!"
)