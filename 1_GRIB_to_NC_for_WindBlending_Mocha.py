# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 15:52:44 2024

@author: Vignesh Govindharaj
@mail: writeamailtovignesh@gmail.com
@contact: 9715306688

This Program is used to convert grib to NetCDF. At the same time it will do first day appending
"""
#%% Library
import os
import xarray as xr
import numpy as np
from datetime import datetime

#%% File Operations
# Specify the directory containing the Wind NetCDF files
#directory = r"D:/VIGNESH/WindBlending/Mocha_GRIB_and_NC"
directory = r"D:\VIGNESH\WindBlending\Data\ECMWF_Mocha_cy_8To16_may2023_mr_apr23tomay31\Mocha"
cyclone_name = "Mocha"
#cyclone_name = "Biparjoy"
# Change the directory
os.chdir(directory)

#% Log filename
logfilename = cyclone_name+"_GRIB-to-NC_Logfile"


#%% Functions
def update_log(message):
    # Get the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Format the log entry
    log_entry = f"{timestamp} - {message}\n"
    print(log_entry)
    # Open the log file in append mode and write the log entry
    with open(directory +'/'+logfilename+'.txt', 'a') as log_file:
        log_file.write(log_entry)
#%% Analysis
# List all required ECMWF date folders
date_folders = [f for f in os.listdir(directory) if len(f) == 8]
update_log("GRIB to NC")
update_log("Input Folder is "+directory)
update_log("Inside the folder, Date folder contains the GRIB files")
update_log("Date Folders :"+str(date_folders))
# Initialize lists to hold data
val_u10 = []
val_v10 = []
val_ws = []
val_pressure = []
val_time = []

# Loop for each date
for date_folder in date_folders:
    # List all files within the date folder
    date_folder_path = os.path.join(directory, date_folder)
    os.chdir(date_folder_path)
    
    # Get the first 8 files in the date folder
    files = [f for f in os.listdir(date_folder_path) if len(f) == 20][:8]
    update_log(f"Date Folder: {date_folder}")
    for file in files:
        # Read the GRIB message
        grbs = xr.open_dataset(file, engine='cfgrib')
        p = grbs.variables['msl'].values * 0.01
        u10 = grbs['u10'].values
        v10 = grbs['v10'].values
        lat = grbs.variables['latitude'].values
        lon = grbs.variables['longitude'].values
        valid_time = grbs.variables['valid_time'].values
        
        # Calculate wind speed
        wind_speed = np.sqrt(u10**2 + v10**2)
        
        # Append data to lists
        val_pressure.append(p)
        val_u10.append(u10)
        val_v10.append(v10)
        val_ws.append(wind_speed)
        val_time.append(valid_time)
        update_log(f"   File: {file} was read and converted to NC format")
     
        # Uncomment the following sections if you need to visualize the data
        # %% Quiver plot
        # fig, ax = plt.subplots(figsize=(10, 10))
        # q = ax.quiver(lon, lat, val_u10, val_v10, wind_speed, scale=300, cmap='coolwarm', width=0.0005)
        # plt.colorbar(q, ax=ax, label='Wind Speed (m/s)')
        # ax.set_xlabel('Longitude')
        # ax.set_ylabel('Latitude')
        # ax.set_title('Wind Vectors (u10, v10)')
        # plt.show()        

        # %% Contour plot
        # fig, ax = plt.subplots(figsize=(14, 9))
        # contour = ax.contourf(lon, lat, wind_speed, cmap='coolwarm')
        # plt.colorbar(contour, ax=ax, label='Wind Speed (m/s)')
        # ax.set_xlabel('Longitude')
        # ax.set_ylabel('Latitude')
        # ax.set_title('Wind Speed')
        # plt.show()

#%% Convert lists to numpy arrays for xarray Dataset
val_u10 = np.array(val_u10)
val_v10 = np.array(val_v10)
val_ws = np.array(val_ws)
val_pressure = np.array(val_pressure)
val_time = np.array(val_time, dtype='datetime64[ns]')
update_log("NC File Started Preparing & Exporting")
# Calculate time in hours since 1901-01-15
origin = np.datetime64('1901-01-15T00:00:00')
val_time_hours = (val_time - origin) / np.timedelta64(1, 'h')

# Create xarray Dataset
ds = xr.Dataset(
    {
        "u10": (["time", "latitude", "longitude"], val_u10),
        "v10": (["time", "latitude", "longitude"], val_v10),
        "ws": (["time", "latitude", "longitude"], val_ws),
        "pressure": (["time", "latitude", "longitude"], val_pressure),
    },
    coords={
        "latitude": lat,
        "longitude": lon,
        "time": val_time_hours
    }
)

# Set the units attribute for the time coordinate
ds["time"].attrs["units"] = "hours since 1901-01-15"
ds["time"].attrs["ref"] = "hours since 1901-01-15"

# Save to NetCDF file
output_file = os.path.join(directory, cyclone_name+"_wind_data.nc")
ds.to_netcdf(output_file)
update_log(output_file+"    Was Exported")
update_log("%%%%%%%%%%%%%%%%%%%   Analysis Completed   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")