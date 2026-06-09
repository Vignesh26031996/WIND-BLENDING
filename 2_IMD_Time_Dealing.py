# -*- coding: utf-8 -*-
"""
After Converting the data from .grib to nc
Inputs: 
    1. IMD Best track -  .xls format
    2. Name of the cyclone - "xxxxxx"
    3. Appended wind data - .nc format
    4. Land and Ocean Shapefile

Created on Thu Jun 20 09:58:28 2024

@author: Vignesh G
@phone: 9715306688
@mail: writeamailtovignesh@gmail.com
"""
#%% Import Library
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from shapely.geometry import Point
import geopandas as gpd
#%% Step-1: IMD Track interpolation
#%%Read Best cyclone track excel
#% Inputs were given here
data = pd.read_excel("D:/VIGNESH/WIndBlending_REDO/Data/BestTracksData(1982-2024).xls", sheet_name="2023")
cyclone_Name = 'MOCHA'
#cyclone_Name = 'BIPARJOY'
#% Appended Wind data
#appended_data = "D:/VIGNESH/WindBlending/Mocha/Mocha_wind_data.nc"
appended_data = "D:/VIGNESH/WIndBlending_REDO/Data/ECMWF_Mocha_cy_8To16_may2023_mr_apr23tomay31/Mocha/Mocha_wind_data.nc"
#% Land and Ocean shapefile
land_ocean_path = r"D:/VIGNESH/WindBlending_REDO/Land_Ocean.shp"
land_ocean = gpd.read_file(land_ocean_path)
Cyclone_data = data[data['Name'] == cyclone_Name].reset_index(drop=True)
datelist = Cyclone_data['Date(DD/MM/YYYY)']
# Convert 'Time (UTC)' column to numeric, setting errors='coerce' to handle non-numeric values
Cyclone_data['Time (UTC)'] = pd.to_numeric(Cyclone_data['Time (UTC)'], errors='coerce')
timelist = Cyclone_data['Time (UTC)']
#%% Functions
# Function to clean and convert the string to float
def clean_and_convert(value):
    try:
        # Ensure the value is a string
        if isinstance(value, str):
            # Check if the string ends with a dot and remove it
            if value.endswith('.'):
                value = value[:-1]
            # Convert to float
            return float(value)
        else:
            # If the value is not a string, try converting directly to float
            return float(value)
    except ValueError:
        # Handle the error if conversion fails
        print(f"Could not convert value to float: {value}")
        return None  # or handle it appropriately for your use case
    
# Function to draw square on top of the image
def draw_square(lat_center, lon_center, size, ax=None):
    half_size = size / 2
    
    # Calculate the coordinates of the square's corners
    lat_min = lat_center - half_size
    lat_max = lat_center + half_size
    lon_min = lon_center - half_size
    lon_max = lon_center + half_size

    # Define the corners of the square
    square_corners = [
        (lat_min, lon_min), # bottom-left
        (lat_min, lon_max), # bottom-right
        (lat_max, lon_max), # top-right
        (lat_max, lon_min), # top-left
        (lat_min, lon_min)  # close the square
    ]

    # Separate the latitudes and longitudes for plotting
    lats, lons = zip(*square_corners)

    # Plot the square
    if ax is None:
        ax = plt.gca()
    ax.plot(lons, lats, marker='o', linestyle='-', color='b', label='Square', markersize=1.5)
    
    return lat_min, lat_max, lon_min, lon_max

# Maximum wind speed of that region
def extract_ws_max_within_square(lat_min, lat_max, lon_min, lon_max, lat2d, lon2d, ws):
    # Create a mask to find the points within the square
    mask = (lat2d >= lat_min) & (lat2d <= lat_max) & (lon2d >= lon_min) & (lon2d <= lon_max)
    ECMWF_data = pd.DataFrame()
    # Extract the ws values within the square
    ECMWF_data['ws'] = ws[mask]
    ECMWF_data['lat'] = lat2d[mask]
    ECMWF_data['lon'] = lon2d[mask]
    max_ws = ECMWF_data[ECMWF_data['ws'] == ECMWF_data['ws'].max()].reset_index(drop=True)
    #min_ws = ECMWF_data[ECMWF_data['ws'] == ECMWF_data['ws'].min()].reset_index(drop=True)
    return max_ws

# Cyclone eye for that region
def extract_ws_min_within_square(lat_min, lat_max, lon_min, lon_max, lat2d, lon2d, ws, pressure):
    # Create a mask to find the points within the square
    mask = (lat2d >= lat_min) & (lat2d <= lat_max) & (lon2d >= lon_min) & (lon2d <= lon_max)
    ECMWF_data = pd.DataFrame()
    # Extract the ws values within the square
    ECMWF_data['ws'] = ws[mask]
    ECMWF_data['pressure'] = pressure[mask]
    ECMWF_data['lat'] = lat2d[mask]
    ECMWF_data['lon'] = lon2d[mask]
    #max_ws = ECMWF_data[ECMWF_data['ws'] == ECMWF_data['ws'].max()].reset_index(drop=True)
    # Check the pressure value 
    if ECMWF_data['pressure'].min() >1000: 
        min_ws = ECMWF_data[ECMWF_data['ws'] == ECMWF_data['ws'].min()].reset_index(drop=True)
    else:
        # Sorting ECMWF_data by the 'pressure' column in ascending order
        ECMWF_data = ECMWF_data.sort_values(by='pressure', ascending=True).reset_index(drop=True)
        ECMWF_data = ECMWF_data[0:10]
        min_ws = ECMWF_data[ECMWF_data['ws'] == ECMWF_data['ws'].min()].reset_index(drop=True)
    return min_ws

#%% Data Organizing (Align date in series)
#% Date Formatting
date_modified = []
for i in range(len(timelist)):
    #% Get Time Information:
    try:
        time_utc = timedelta(hours=round(timelist[i] / 100))
    except:
        time_utc = np.nan
    #% Get date 
    if not pd.isna(Cyclone_data['Date(DD/MM/YYYY)'][i]):
        date_row = Cyclone_data['Date(DD/MM/YYYY)'][i]
    if pd.isna(time_utc):
        datetime_together = np.nan
    else:
        datetime_together = date_row + time_utc
    date_modified.append(datetime_together)

#% DateTime column to CycloneData
Cyclone_data['DateTime'] = date_modified
not_time = pd.isna(Cyclone_data['DateTime'])
Cyclone_data = Cyclone_data[~not_time]

# Drop the last row if needed
Cyclone_data = Cyclone_data.drop(Cyclone_data.index[-1]).reset_index(drop=True)

#% Calculate the interval between DateTime entries
Cyclone_data['Interval'] = np.nan
for i in range(1, len(Cyclone_data)):
    interval_a = Cyclone_data['DateTime'].iloc[i - 1]
    interval_b = Cyclone_data['DateTime'].iloc[i]
    Cyclone_data.loc[i, 'Interval'] = interval_b - interval_a
    
#% Interval in Number format
Cyclone_data['Interval_hours'] = Cyclone_data['Interval'].apply(lambda x: f"{int(x.total_seconds() // 3600):02}" if pd.notna(x) else '00')

#%% Find other than 3 hours IMD data
# Identify indices where 'Interval_hours' is '06'
indices_to_insert = Cyclone_data.index[Cyclone_data['Interval_hours'] == '06'].tolist()
Cyclone_data['Remark'] = "Raw"

# Insert new rows above each identified index and Interpolation
for index in reversed(indices_to_insert):  # Reverse to maintain correct indices after insertion
    upper_part = Cyclone_data.iloc[:index]
    lower_part = Cyclone_data.iloc[index:]
    date_time = Cyclone_data['DateTime'].iloc[index - 1] + timedelta(hours=3)
    lat = (Cyclone_data['Latitude (lat)'].iloc[index - 1] + Cyclone_data['Latitude (lat)'].iloc[index ])/2
    lon = (Cyclone_data['longitude  (Long)'].iloc[index - 1] + Cyclone_data['longitude  (Long)'].iloc[index ])/2
    hPa = (Cyclone_data['Estimated Central Pressure (hPa) [or "E.C.P"]'].iloc[index -1] + Cyclone_data['Estimated Central Pressure (hPa) [or "E.C.P"]'].iloc[index ])/2
    kt = (Cyclone_data['Maximum Sustained Surface Wind (kt) '].iloc[index-1]+Cyclone_data['Maximum Sustained Surface Wind (kt) '].iloc[index ])/2
    pdrop = (Cyclone_data['Pressure Drop (hPa)[or "delta P"]'].iloc[index-1]+Cyclone_data['Pressure Drop (hPa)[or "delta P"]'].iloc[index ])/2
    grade = Cyclone_data['Grade (text)'].iloc[index-1]
    new_row = pd.Series({
        'Name': cyclone_Name,
        'Date(DD/MM/YYYY)': pd.NaT,
        'Time (UTC)': np.nan,
        'DateTime': date_time,
        'Interval': np.nan,
        'Interval_hours': "03",
        'Remark': "Modified",
        'Latitude (lat)':lat,
        'longitude  (Long)':lon,
        'Estimated Central Pressure (hPa) [or "E.C.P"]':hPa,
        'Maximum Sustained Surface Wind (kt) ':kt,
        'Pressure Drop (hPa)[or "delta P"]':pdrop,
        'Grade (text)':grade,
    })
    Cyclone_data = pd.concat([upper_part, new_row.to_frame().T, lower_part]).reset_index(drop=True)

Cyclone_data.to_csv("CycloneName_"+cyclone_Name+".csv")

#%% Step-2: ECMWF data, load datasets
#%% Load the dataset
ds = xr.open_dataset(appended_data)
# Coordinates
lat = np.array(ds['latitude'])
lon = np.array(ds['longitude'])
all_time = ds['time'].to_pandas()
#% Analysis
lat_lp= []
lon_lp= []
image_files = []
for i in range(len(all_time)):
    #% Extract the variables
    ws = np.array(ds['ws'].isel(time=i)) # Example: selecting the first time step
    u = np.array(ds['u10'].isel(time=i))  
    v = np.array(ds['v10'].isel(time=i))
    pressure = np.array(ds['pressure'].isel(time=i))  
    #% Get the Time
    time = str(ds['time'].isel(time=i).to_pandas())
    date_format = '%Y-%m-%dT%H:%M:%S'
    datetime_obj = datetime.strptime(time[:19], date_format)
    datetime_obj = datetime_obj.strftime('%Y-%m-%d %H:%M:%S')
    #% Find the time in geodatabase
    Cyclone_data['DateTime'] = Cyclone_data['DateTime'].astype(str)
    matching_row = Cyclone_data[Cyclone_data['DateTime']==datetime_obj].reset_index(drop=True)
    #% Extract latitude and longitude if there is a match
    if not matching_row.empty:
        # Apply the function to the specific column
        matching_row['Latitude (lat)'] = matching_row['Latitude (lat)'].apply(clean_and_convert)        
        # Now you can safely convert the cleaned value to float
        # lat_lp.append(matching_row.iloc[0]['Latitude (lat)'])
        # lon_lp.append(matching_row.iloc[0]['longitude  (Long)'])
        lat_lp = matching_row.iloc[0]['Latitude (lat)']
        lon_lp = matching_row.iloc[0]['longitude  (Long)']
        match = 'yes'        
        # get index to store the value
        cyc_index = Cyclone_data[Cyclone_data['DateTime']==datetime_obj].index
        Cyclone_data.loc[cyc_index,'Latitude (lat)'] = matching_row.iloc[0]['Latitude (lat)']
    else:
        print('No matching datetime found. Date:' + str(datetime_obj))
        match = 'no'
        lat_lp= []
        lon_lp= []
    #% Create a meshgrid for the latitude and longitude
    lon2d, lat2d = np.meshgrid(lon, lat)
    #%% Create figure and axes
    fig, ax = plt.subplots(figsize=(10, 10))
    # Plot the wind speed as a pcolor plot
    pc = ax.pcolor(lon2d, lat2d, ws, shading='auto',cmap = 'viridis', vmin = 0, vmax = 20)
    cbar = fig.colorbar(pc, ax=ax, label='Wind Speed (m/s)')
    ax.set_xlabel('Longitude in degree')
    ax.set_ylabel('Latitude in degree')
    ax.set_title('Wind Speed / DateTime : '+str(datetime_obj))
    ax.plot(lon_lp, lat_lp, marker='o', linestyle='-', color='r', label='Line plot',markersize=1.5)
    if match == 'yes':
        #% Decide the bounding box value
        get_grade = Cyclone_data['Grade (text)'].iloc[cyc_index].to_list()
        # Define the array and bounding box
        categories = np.array(['D', 'DD', 'CS', 'SCS', 'VSCS', 'ESCS'], dtype=object)        
        # Bounding box for Cyclone eye (values in degree)
        bounding_box_min = 2     
        # Bounding box for maximum wind speed (values in degree)
        bounding_box = np.array([2, 2.5, 3, 3.5, 4, 4.5])        
        # Define a dictionary to act as a switch-case
        category_bounding_box_map = {
            'D': bounding_box[0],
            'DD': bounding_box[1],
            'CS': bounding_box[2],
            'SCS': bounding_box[3],
            'VSCS': bounding_box[4],
            'ESCS': bounding_box[5]
        }        
        bounding_box = category_bounding_box_map[get_grade[0]]
        lat_min, lat_max, lon_min, lon_max = draw_square(lat_lp, lon_lp,bounding_box,ax=ax)
        max_ws = extract_ws_max_within_square(lat_min, lat_max, lon_min, lon_max, lat2d, lon2d, ws)
        Cyclone_data.loc[cyc_index,'ecmwf_ws_max'] = max_ws['ws'][0]
        Cyclone_data.loc[cyc_index,'ecmwf_lat_max'] = max_ws['lat'][0]
        Cyclone_data.loc[cyc_index,'ecmwf_lon_max'] = max_ws['lon'][0]
        lat_min, lat_max, lon_min, lon_max = draw_square(lat_lp, lon_lp,bounding_box_min,ax=ax)
        min_ws = extract_ws_min_within_square(lat_min, lat_max, lon_min, lon_max, lat2d, lon2d, ws, pressure)
        Cyclone_data.loc[cyc_index,'ecmwf_ws_min'] = min_ws['ws'][0]
        Cyclone_data.loc[cyc_index,'ecmwf_lat_min'] = min_ws['lat'][0]
        Cyclone_data.loc[cyc_index,'ecmwf_lon_min'] = min_ws['lon'][0]
        Cyclone_data.loc[cyc_index,'e_hpa_min'] = min_ws['pressure'][0]
        # Identify the pixels
        # Define the target latitude and longitude for IMD
        target_lat = Cyclone_data['Latitude (lat)'].iloc[cyc_index].tolist()
        target_lon = Cyclone_data['longitude  (Long)'].iloc[cyc_index].tolist()
        
        # Calculate the absolute difference and find the index of the minimum distance
        distance = np.sqrt((lat2d - target_lat)**2 + (lon2d - target_lon)**2)
        row_col_imd = np.unravel_index(np.argmin(distance), distance.shape)

        # Find the indices of the closest values in lat and lon arrays for IMD
        Cyclone_data.loc[cyc_index,'IMD_row'] = row_col_imd[0]
        Cyclone_data.loc[cyc_index,'IMD_column'] = row_col_imd[1]
        
        # Define the target latitude and longitude for ECMWF
        target_lat = min_ws['lat'][0]
        target_lon = min_ws['lon'][0]
        
        # Calculate the absolute difference and find the index of the minimum distance
        distance = np.sqrt((lat2d - target_lat)**2 + (lon2d - target_lon)**2)
        row_col_ecmwf = np.unravel_index(np.argmin(distance), distance.shape)

        # Find the indices of the closest values in lat and lon arrays for ECMWF
        Cyclone_data.loc[cyc_index,'ECMWF_row'] = row_col_ecmwf[0]
        Cyclone_data.loc[cyc_index,'ECMWF_column'] = row_col_ecmwf[1]
        Cyclone_data.loc[cyc_index,'Horizontal'] = row_col_imd[1] - row_col_ecmwf[1]
        Cyclone_data.loc[cyc_index,'Vertical'] = row_col_ecmwf[0] - row_col_imd[0] 
        imd_wind = Cyclone_data['Maximum Sustained Surface Wind (kt) '][cyc_index].values[0]
        ecmwf_wind = max_ws['ws'][0]
        if ecmwf_wind<imd_wind:
            Cyclone_data.loc[cyc_index,'Blending'] = "Yes"
        else:
            Cyclone_data.loc[cyc_index,'Blending'] = "No"
    # Overlay the quiver plot
    #qv = ax.quiver(lon2d, lat2d, u, v, scale=800, color='k')
    # Save the plot as an image
    # imagefilename = f'plot_Single{i}.png'
    # plt.savefig(imagefilename,dpi = 300)
    # image_files.append(imagefilename)    
    # # Show the plot
    plt.show()
    #plt.close()  
    
#%% Export shapefiles
Cyclone_data['Latitude (lat)'] = Cyclone_data['Latitude (lat)'].astype(np.double)
Cyclone_data['longitude  (Long)'] = Cyclone_data['longitude  (Long)'].astype(np.double)
Cyclone_data.to_csv("CycloneName_"+cyclone_Name+"with_ECMWF_Insights.csv")

# Create a geometry column using the latitude and longitude
Cyclone_data['g_imd'] = Cyclone_data.apply(lambda row: Point(row['longitude  (Long)'], row['Latitude (lat)']), axis=1)
Cyclone_data['g_ecmwf'] = Cyclone_data.apply(lambda row: Point(row['ecmwf_lon_min'], row['ecmwf_lat_min']), axis=1)
# Shorten column names to be within 10-character limit and convert datetime fields to strings
Cyclone_data = Cyclone_data.rename(columns={
    'Serial Number of system during year': 'Serial_No',
    'Basin of origin': 'Basin',
    'Date(DD/MM/YYYY)': 'Date',
    'Time (UTC)': 'Time',
    'Latitude (lat)': 'Lat',
    'longitude  (Long)': 'Lon',
    'CI No [or "T. No"]': 'CI_No',
    'Estimated Central Pressure (hPa) [or "E.C.P"]': 'ECP',
    'Maximum Sustained Surface Wind (kt) ': 'Max_Wind',
    'Pressure Drop (hPa)[or "delta P"]': 'hPa',
    'Grade (text)': 'Grade',
    'Outermost closed isobar (hPa)': 'Isobar',
    'Diameter/Size of outermost closed isobar(in degree)': 'Si_Ibar',
    'DateTime': 'D_Time',
    'Interval_hours': 'Int_hours',
    'ecmwf_ws_max': 'e_max_ws',
    'ecmwf_lat_max': 'e_max_lat',
    'ecmwf_lon_max': 'e_max_lon',
    'ecmwf_ws_min': 'e_min_ws',
    'ecmwf_lat_min': 'e_min_latl',
    'ecmwf_lon_min': 'e_min_lon'
})
# Convert the DataFrame to a GeoDataFrame
Cyclone_data = Cyclone_data.drop(columns=['Date','Int_hours','Interval'])
gdf_imd = gpd.GeoDataFrame(Cyclone_data, geometry='g_imd')
gdf_imd = gdf_imd.drop(columns='g_ecmwf')
gdf_imd.set_crs(epsg=4326, inplace=True)
#gdf_imd['DateTime'] = gdf_imd['DateTime'].astype(str)
# Perform the union operation
gdf_imd_union = gpd.overlay(gdf_imd, land_ocean, how='union')
idx = gdf_imd_union[gdf_imd_union['Part']!="Land"].index
gdf_imd_union.loc[idx,'Part'] = "Ocean"
# Define the coordinate reference system (CRS), here using WGS84
gdf_ecmwf = gpd.GeoDataFrame(Cyclone_data, geometry='g_ecmwf')
gdf_ecmwf = gdf_ecmwf.drop(columns='g_imd')
gdf_ecmwf.set_crs(epsg=4326, inplace=True)
gdf_ecmwf_union = gpd.overlay(gdf_ecmwf, land_ocean, how='union')
gdf_ecmwf_union.loc[idx,'Part'] = "Ocean"
# Export the GeoDataFrame to a shapefile

gdf_imd_union.to_file("CycloneName_"+cyclone_Name+"_imd.shp")
gdf_ecmwf_union.to_file("CycloneName_"+cyclone_Name+"_ecmwf.shp")
print("Shapefile created successfully.")
