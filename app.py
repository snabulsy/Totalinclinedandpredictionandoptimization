import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


st.set_page_config(page_title="Fixed Solar Panel Orientation", layout="wide")
st.title("Fixed Solar Panel Tilt & Azimuth Calculator (Monthly Energy Curves)")

# --- Inputs ---
st.header("Step 1: Upload Data")
uploaded_file = st.file_uploader("Upload Excel file with columns: Day, Hour, I_tot, I_diff", type=["xlsx", "xls"])

st.header("Step 2: Enter Location Info")
latitude = st.number_input("Latitude (° N positive, S negative)", value=30.0)
albedo = st.number_input("Ground albedo (reflectivity, e.g., 0.2)", value=0.2)

# --- Process data ---
if uploaded_file:
    df = pd.read_excel(uploaded_file)
    
    required_cols = ['Day', 'Hour', 'I_tot', 'I_diff']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Excel file must contain columns: {required_cols}")
    else:
        st.success("Data loaded successfully!")

        # Add month column
        df['Month'] = pd.to_datetime(df['Day'], format='%j', errors='coerce').dt.month.fillna(1).astype(int)

        # --- Step 3: Solar geometry ---
        df['Declination'] = 23.45 * np.sin(np.deg2rad(360 * (284 + df['Day']) / 365))
        df['Hour_angle'] = 15 * (df['Hour'] - 12)

        lat_rad = np.deg2rad(latitude)
        dec_rad = np.deg2rad(df['Declination'])
        H_rad = np.deg2rad(df['Hour_angle'])

        cos_theta_z = np.sin(lat_rad)*np.sin(dec_rad) + np.cos(lat_rad)*np.cos(dec_rad)*np.cos(H_rad)
        df['cos_theta_z'] = np.clip(cos_theta_z, 0, 1)

        df['I_dir'] = (df['I_tot'] - df['I_diff']) / df['cos_theta_z']

        # --- Step 4: Optimize fixed tilt & azimuth for whole year ---
        tilt_angles = np.arange(0, 46, 5)
        azimuth_angles = np.arange(-90, 91, 10)
        max_energy = 0
        best_tilt = 0
        best_azimuth = 0

        progress_text = "Optimizing fixed tilt & azimuth for the year..."
        my_bar = st.progress(0)
        total_steps = len(tilt_angles) * len(azimuth_angles)
        step_count = 0

        for beta in tilt_angles:
            beta_rad = np.deg2rad(beta)
            for gamma in azimuth_angles:
                gamma_rad = np.deg2rad(gamma)

                cos_theta = (np.sin(dec_rad)*np.sin(lat_rad)*np.cos(beta_rad)
                             - np.sin(dec_rad)*np.cos(lat_rad)*np.sin(beta_rad)*np.cos(gamma_rad)
                             + np.cos(dec_rad)*np.cos(lat_rad)*np.cos(H_rad)*np.cos(beta_rad)
                             + np.cos(dec_rad)*np.sin(lat_rad)*np.cos(H_rad)*np.sin(beta_rad)*np.cos(gamma_rad)
                             + np.cos(dec_rad)*np.sin(H_rad)*np.sin(beta_rad)*np.sin(gamma_rad)
                            )
                cos_theta = np.clip(cos_theta, 0, 1)
                r_b = cos_theta / df['cos_theta_z']
                r_d = (1 + np.cos(beta_rad)) / 2
                r_r = albedo * (1 - np.cos(beta_rad)) / 2

                I_tilt = df['I_dir'] * r_b + df['I_diff'] * r_d + (df['I_dir'] + df['I_diff']) * r_r
                total = I_tilt.sum()

                if total > max_energy:
                    max_energy = total
                    best_tilt = beta
                    best_azimuth = gamma

                step_count += 1
                my_bar.progress(step_count / total_steps)

        st.header("Step 5: Fixed Tilt & Azimuth for the Year")
        st.write(f"✅ Fixed tilt angle: **{best_tilt}°**")
        st.write(f"✅ Fixed azimuth angle: **{best_azimuth}°**")
        st.write(f"✅ Maximum total annual energy: **{max_energy:.0f} W·h**")

        # --- Step 6: Compute monthly energy curves ---
        monthly_energy = {m: [] for m in range(1, 13)}
        best_azimuth_rad = np.deg2rad(best_azimuth)

        for beta in tilt_angles:
            beta_rad = np.deg2rad(beta)
            for month in range(1, 13):
                df_month = df[df['Month'] == month]
                cos_theta = (np.sin(np.deg2rad(df_month['Declination']))*np.sin(lat_rad)*np.cos(beta_rad)
                             - np.sin(np.deg2rad(df_month['Declination']))*np.cos(lat_rad)*np.sin(beta_rad)*np.cos(best_azimuth_rad)
                             + np.cos(np.deg2rad(df_month['Declination']))*np.cos(lat_rad)*np.cos(np.deg2rad(df_month['Hour_angle']))*np.cos(beta_rad)
                             + np.cos(np.deg2rad(df_month['Declination']))*np.sin(lat_rad)*np.cos(np.deg2rad(df_month['Hour_angle']))*np.sin(beta_rad)*np.cos(best_azimuth_rad)
                             + np.cos(np.deg2rad(df_month['Declination']))*np.sin(np.deg2rad(df_month['Hour_angle']))*np.sin(beta_rad)*np.sin(best_azimuth_rad)
                            )
                cos_theta = np.clip(cos_theta, 0, 1)
                r_b = cos_theta / df_month['cos_theta_z']
                r_d = (1 + np.cos(beta_rad)) / 2
                r_r = albedo * (1 - np.cos(beta_rad)) / 2

                I_tilt = df_month['I_dir'] * r_b + df_month['I_diff'] * r_d + (df_month['I_dir'] + df_month['I_diff']) * r_r
                monthly_energy[month].append(I_tilt.sum())

        # --- Step 7: Plot monthly curves ---
        fig, ax = plt.subplots(figsize=(10,6))
        for month in range(1, 13):
            ax.plot(tilt_angles, monthly_energy[month], marker='o', label=f'Month {month}')
        ax.set_xlabel("Tilt Angle (°)")
        ax.set_ylabel("Total Monthly Energy (W·h)")
        ax.set_title("Monthly Total Energy vs Tilt Angle (Fixed Azimuth)")
        ax.legend()
        st.pyplot(fig)
