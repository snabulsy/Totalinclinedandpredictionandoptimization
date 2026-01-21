import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# ----------------------------
# Streamlit Config
# ----------------------------
st.set_page_config(page_title="Solar Panel Fixed Tilt Optimizer", layout="wide")
st.title("📊 Solar Panel Fixed Tilt & Azimuth Optimizer (Realistic Energy Values)")

# ----------------------------
# Step 1: Upload Data
# ----------------------------
st.header("Step 1: Upload Data")
uploaded_file = st.file_uploader(
    "Upload Excel with columns: Month, Day, Hour, I_tot, I_diff",
    type=["xlsx", "xls"]
)

# ----------------------------
# Step 2: Location Info
# ----------------------------
st.header("Step 2: Enter Location Info")
latitude = st.number_input("Latitude (° N positive, S negative)", value=30.0)
albedo = st.number_input("Ground albedo (reflectivity, e.g., 0.2)", value=0.2)

# ----------------------------
# Step 3: Process Data & Optimize
# ----------------------------
if uploaded_file:
    df = pd.read_excel(uploaded_file, engine='openpyxl')
    required_cols = ['Month', 'Day', 'Hour', 'I_tot', 'I_diff']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Excel must contain columns: {required_cols}")
    else:
        st.success("Data loaded successfully!")
        
        # Ensure correct types
        df['Month'] = df['Month'].astype(int)
        df['Day_of_month'] = df['Day'].astype(int)
        df['Hour'] = df['Hour'].astype(int)
        
        # Compute day of year for declination
        day_of_year = df.groupby(['Month','Day_of_month']).ngroup() + 1
        df['Declination'] = 23.45 * np.sin(np.deg2rad(360 * (284 + day_of_year) / 365))
        df['Hour_angle'] = 15 * (df['Hour'] - 12)
        
        lat_rad = np.deg2rad(latitude)
        dec_rad = np.deg2rad(df['Declination'])
        H_rad = np.deg2rad(df['Hour_angle'])
        
        # Avoid tiny cos_theta_z that makes I_dir huge
        df['cos_theta_z'] = np.clip(np.sin(lat_rad)*np.sin(dec_rad) + np.cos(lat_rad)*np.cos(dec_rad)*np.cos(H_rad), 0.1, 1)
        
        # Compute direct normal irradiance safely
        df['I_dir'] = (df['I_tot'] - df['I_diff']) / df['cos_theta_z']
        df['I_dir'] = df['I_dir'].apply(lambda x: max(x,0))  # avoid negatives
        
        # ----------------------------
        # Optimize fixed tilt & azimuth
        # ----------------------------
        st.header("Step 3: Optimize Fixed Tilt & Azimuth")
        tilt_angles = np.arange(0, 46, 5)
        azimuth_angles = np.arange(-90, 91, 10)
        max_energy = 0
        best_tilt = 0
        best_azimuth = 0
        
        with st.spinner("🔄 Optimizing tilt & azimuth for maximum annual energy..."):
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
        
        st.success(f"✅ Optimal tilt: **{best_tilt}°**, Optimal azimuth: **{best_azimuth}°**, Max annual energy: **{max_energy:.0f} W·h**")
        
        # ----------------------------
        # Step 4: Daily Energy per Month (Separate Plots)
        # ----------------------------
        st.header("Step 4: Daily Energy Curves (Separate Plots per Month)")
        best_azimuth_rad = np.deg2rad(best_azimuth)
        
        for month in range(1, 13):
            df_month = df[df['Month'] == month].copy()
            daily_totals = []
            for day in df_month['Day_of_month'].unique():
                df_day = df_month[df_month['Day_of_month'] == day]
                beta_rad = np.deg2rad(best_tilt)
                gamma_rad = best_azimuth_rad

                cos_theta = (np.sin(np.deg2rad(df_day['Declination']))*np.sin(lat_rad)*np.cos(beta_rad)
                             - np.sin(np.deg2rad(df_day['Declination']))*np.cos(lat_rad)*np.sin(beta_rad)*np.cos(gamma_rad)
                             + np.cos(np.deg2rad(df_day['Declination']))*np.cos(lat_rad)*np.cos(np.deg2rad(df_day['Hour_angle']))*np.cos(beta_rad)
                             + np.cos(np.deg2rad(df_day['Declination']))*np.sin(lat_rad)*np.cos(np.deg2rad(df_day['Hour_angle']))*np.sin(beta_rad)*np.cos(gamma_rad)
                             + np.cos(np.deg2rad(df_day['Declination']))*np.sin(np.deg2rad(df_day['Hour_angle']))*np.sin(beta_rad)*np.sin(gamma_rad)
                            )
                cos_theta = np.clip(cos_theta, 0, 1)
                r_b = cos_theta / df_day['cos_theta_z']
                r_d = (1 + np.cos(beta_rad)) / 2
                r_r = albedo * (1 - np.cos(beta_rad)) / 2
                I_tilt = df_day['I_dir'] * r_b + df_day['I_diff'] * r_d + (df_day['I_dir'] + df_day['I_diff']) * r_r
                daily_totals.append(I_tilt.sum())
            
            x_days = np.arange(1, len(daily_totals)+1)
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(x_days, daily_totals, marker='o', color='orange', linewidth=2)
            ax.set_title(f"Month {month} - Daily Energy (Tilt={best_tilt}°, Azimuth={best_azimuth}°)")
            ax.set_xlabel("Day of Month")
            ax.set_ylabel("Total Daily Energy (W·h)")
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig, clear_figure=True)
        
        # ----------------------------
        # Step 5: Detailed Calculations Table
        # ----------------------------
        st.header("Step 5: Detailed Calculations Table")
        beta_rad = np.deg2rad(best_tilt)
        gamma_rad = np.deg2rad(best_azimuth)
        cos_theta = (np.sin(dec_rad)*np.sin(lat_rad)*np.cos(beta_rad)
                     - np.sin(dec_rad)*np.cos(lat_rad)*np.sin(beta_rad)*np.cos(gamma_rad)
                     + np.cos(dec_rad)*np.cos(lat_rad)*np.cos(H_rad)*np.cos(beta_rad)
                     + np.cos(dec_rad)*np.sin(lat_rad)*np.cos(H_rad)*np.sin(beta_rad)*np.cos(gamma_rad)
                     + np.cos(dec_rad)*np.sin(H_rad)*np.sin(beta_rad)*np.sin(gamma_rad)
                    )
        cos_theta = np.clip(cos_theta, 0, 1)
        df['r_b'] = cos_theta / df['cos_theta_z']
        df['r_d'] = (1 + np.cos(beta_rad)) / 2
        df['r_r'] = albedo * (1 - np.cos(beta_rad)) / 2
        df['I_tilt'] = df['I_dir'] * df['r_b'] + df['I_diff'] * df['r_d'] + (df['I_dir'] + df['I_diff']) * df['r_r']

        with st.expander("Show first 500 rows of calculations"):
            st.dataframe(df.head(500))
        
        # Download full table
        def to_excel(df):
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df.to_excel(writer, index=False, sheet_name='Calculations')
            return output.getvalue()

        st.download_button(
            label="📥 Download Full Calculations as Excel",
            data=to_excel(df),
            file_name="solar_calculations.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
