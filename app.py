import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io

# ===============================
# App Configuration
# ===============================
st.set_page_config(page_title="Solar Tilt & Azimuth Optimization", layout="wide")
st.title("Fixed Solar Panel Orientation Optimization")

# ===============================
# User Inputs
# ===============================
uploaded_file = st.file_uploader(
    "Upload Excel file with columns: Month, Day, Hour, I_tot, I_diff",
    type=["xlsx"]
)

latitude = st.number_input("Latitude (degrees)", value=30.0)
albedo = st.number_input("Ground albedo", value=0.2)

# ===============================
# Main Logic
# ===============================
if uploaded_file:

    df = pd.read_excel(uploaded_file)

    required_cols = ["Month", "Day", "Hour", "I_tot", "I_diff"]
    if not all(col in df.columns for col in required_cols):
        st.error(f"Excel file must contain: {required_cols}")
        st.stop()

    st.success("Data loaded successfully")

    # ===============================
    # Date & Solar Geometry
    # ===============================
    df["Day_of_Year"] = pd.to_datetime(
        dict(year=2024, month=df["Month"], day=df["Day"]),
        errors="coerce"
    ).dt.dayofyear

    lat_rad = np.deg2rad(latitude)

    df["Declination"] = 23.45 * np.sin(
        np.deg2rad(360 * (284 + df["Day_of_Year"]) / 365)
    )

    df["Hour_angle"] = 15 * (df["Hour"] - 12)

    dec = np.deg2rad(df["Declination"])
    h = np.deg2rad(df["Hour_angle"])

    cos_theta_z = (
        np.sin(lat_rad) * np.sin(dec)
        + np.cos(lat_rad) * np.cos(dec) * np.cos(h)
    )

    df["cos_theta_z"] = np.clip(cos_theta_z, 0.1, 1)

    # ===============================
    # Direct Normal Irradiance
    # ===============================
    df["I_dir"] = (df["I_tot"] - df["I_diff"]) / df["cos_theta_z"]
    df["I_dir"] = df["I_dir"].clip(lower=0)

    # ===============================
    # Optimization (Fixed Tilt & Azimuth)
    # ===============================
    tilt_angles = np.arange(0, 61, 5)
    azimuth_angles = np.arange(-90, 91, 10)

    best_energy = -1
    best_tilt = 0
    best_azimuth = 0

    for beta in tilt_angles:
        beta_rad = np.deg2rad(beta)
        for gamma in azimuth_angles:
            gamma_rad = np.deg2rad(gamma)

            cos_theta = (
                np.sin(dec) * np.sin(lat_rad) * np.cos(beta_rad)
                - np.sin(dec) * np.cos(lat_rad) * np.sin(beta_rad) * np.cos(gamma_rad)
                + np.cos(dec) * np.cos(lat_rad) * np.cos(h) * np.cos(beta_rad)
                + np.cos(dec) * np.sin(lat_rad) * np.cos(h) * np.sin(beta_rad) * np.cos(gamma_rad)
                + np.cos(dec) * np.sin(h) * np.sin(beta_rad) * np.sin(gamma_rad)
            )

            cos_theta = np.clip(cos_theta, 0, 1)

            r_b = cos_theta / df["cos_theta_z"]
            r_d = (1 + np.cos(beta_rad)) / 2
            r_r = albedo * (1 - np.cos(beta_rad)) / 2

            I_tilt = (
                df["I_dir"] * r_b
                + df["I_diff"] * r_d
                + (df["I_dir"] + df["I_diff"]) * r_r
            )

            energy = I_tilt.sum()

            if energy > best_energy:
                best_energy = energy
                best_tilt = beta
                best_azimuth = gamma

    st.subheader("Optimal Fixed Orientation (Whole Year)")
    st.write(f"**Tilt angle:** {best_tilt}°")
    st.write(f"**Azimuth angle:** {best_azimuth}°")
    st.write(f"**Total annual energy:** {best_energy:,.0f} W·h")

    # ===============================
    # Energy Using Optimal Angles
    # ===============================
    beta_rad = np.deg2rad(best_tilt)
    gamma_rad = np.deg2rad(best_azimuth)

    cos_theta = (
        np.sin(dec) * np.sin(lat_rad) * np.cos(beta_rad)
        - np.sin(dec) * np.cos(lat_rad) * np.sin(beta_rad) * np.cos(gamma_rad)
        + np.cos(dec) * np.cos(lat_rad) * np.cos(h) * np.cos(beta_rad)
        + np.cos(dec) * np.sin(lat_rad) * np.cos(h) * np.sin(beta_rad) * np.cos(gamma_rad)
        + np.cos(dec) * np.sin(h) * np.sin(beta_rad) * np.sin(gamma_rad)
    )

    cos_theta = np.clip(cos_theta, 0, 1)

    r_b = cos_theta / df["cos_theta_z"]
    r_d = (1 + np.cos(beta_rad)) / 2
    r_r = albedo * (1 - np.cos(beta_rad)) / 2

    df["I_tilt"] = (
        df["I_dir"] * r_b
        + df["I_diff"] * r_d
        + (df["I_dir"] + df["I_diff"]) * r_r
    )

    # ===============================
    # Monthly Daily Energy Plots
    # ===============================
    daily_energy_by_month = {}

    for m in range(1, 13):
        dmonth = df[df["Month"] == m]
        daily = dmonth.groupby("Day")["I_tilt"].sum()
        daily_energy_by_month[m] = daily.values

        fig, ax = plt.subplots()
        ax.plot(daily.index, daily.values, linewidth=2)
        ax.set_title(f"Month {m} – Daily Energy")
        ax.set_xlabel("Day")
        ax.set_ylabel("Energy (W·h)")
        ax.grid(True)
        st.pyplot(fig)

    # ===============================
    # Dynamic Multi-Month Plot
    # ===============================
    st.subheader("Dynamic Monthly Comparison")

    selected_months = st.multiselect(
        "Select months to display",
        list(range(1, 13)),
        default=list(range(1, 13))
    )

    if selected_months:
        fig, ax = plt.subplots(figsize=(14, 6))
        for m in selected_months:
            ax.plot(daily_energy_by_month[m], label=f"Month {m}", linewidth=2)

        ax.set_title("Daily Energy Comparison")
        ax.set_xlabel("Day index")
        ax.set_ylabel("Energy (W·h)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    # ===============================
    # Show Full Table
    # ===============================
    st.subheader("Full Calculation Table")
    st.dataframe(df, use_container_width=True)

    # ===============================
    # Excel Download (OPENPYXL)
    # ===============================
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Results")

    buffer.seek(0)

    st.download_button(
        label="Download Results as Excel",
        data=buffer,
        file_name="solar_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
