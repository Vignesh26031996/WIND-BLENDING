
"""
After Converting the data from .grib to nc and IMD_Time_Dealing scripts
Inputs: 
    3. Appended wind data - .nc format
    5. IMD_shapefile from IMD_Time_Dealing_v2.py
    
@author: Vignesh G
@phone: 9715306688
@mail: writeamailtovignesh@gmail.com
"""
#%% Import Library
import numpy as np
import xarray as xr
import geopandas as gpd
from scipy.interpolate import interp1d
from datetime import datetime
from scipy.constants import kilo
#%% Inputs
appended_data = "D:/VIGNESH/WIndBlending_REDO/Data/ECMWF_Mocha_cy_8To16_may2023_mr_apr23tomay31/Mocha/Mocha_wind_data.nc"
#appended_data = "D:/VIGNESH/WindBlending/Mocha/Mocha_wind_data.nc"
#% Shapefile hold ECMWF shift values
#Shapefile_path = r"D:/VIGNESH/WindBlending/Mocha/CycloneName_MOCHA_imd.shp"
Shapefile_path = r"D:/VIGNESH/WIndBlending_REDO/Data/ECMWF_Mocha_cy_8To16_may2023_mr_apr23tomay31/CycloneName_MOCHA_imd.shp"
gdf_imd_union = gpd.read_file(Shapefile_path)

#%% Functions
def modify_array(ws, horizontal, vertical):
    """
    Modifies the array based on the horizontal and vertical parameters.
    
    Parameters:
    ws (numpy.ndarray): The input 2D array.
    horizontal (int): Direction and number of columns to modify (-5 to 5):
                      Negative: Adds columns on the left and removes the same number from the right.
                      Positive: Adds columns on the right and removes the same number from the left.
    vertical (int): Direction and number of rows to modify (-5 to 5):
                    Negative: Adds rows at the top and removes the same number from the bottom.
                    Positive: Adds rows at the bottom and removes the same number from the top.
                      
    Returns:
    numpy.ndarray: The modified array with the same shape as the input.
    """
    horizontal = horizontal[0]
    vertical = vertical[0]
    
    # Ensure horizontal and vertical are within the allowed range
    if not -5 <= horizontal <= 5:
        raise ValueError("Horizontal must be within the range -5 to 5.")
    if not -5 <= vertical <= 5:
        raise ValueError("Vertical must be within the range -5 to 5.")
    
    # Horizontal modification (now changes columns)
    if horizontal != 0:
        num_cols = int(abs(horizontal))
        # Choose columns for interpolation based on the direction
        if horizontal < 0:
            cols_to_interpolate_from = ws[:, :num_cols * 2]  # Use the first few columns
        else:
            cols_to_interpolate_from = ws[:, -num_cols * 2:]  # Use the last few columns
        
        # Generate y values for interpolation
        y = np.arange(cols_to_interpolate_from.shape[1])
        
        # Define positions for new columns to interpolate
        y_new = np.linspace(0, 1, num_cols, endpoint=False)
        
        # Create interpolation function
        interp_func = interp1d(y, cols_to_interpolate_from, axis=1, kind='linear')
        
        # Interpolate values for new columns
        new_cols = interp_func(y_new)*0
        new_cols[new_cols == 0] = np.nan
        
        # Modify the array based on the horizontal direction
        if horizontal < 0:
            # Add columns on the right and remove the same number from the left
            ws = np.hstack((ws, new_cols))
            ws = ws[:, num_cols:]  # Remove columns from the left
            
        else:
            # Add columns on the left and remove the same number from the right
            ws = np.hstack((new_cols, ws))
            ws = ws[:, :-num_cols]  # Remove columns from the right
            

    # Vertical modification (now changes rows)
    if vertical != 0:
        num_rows = int(abs(vertical))
        # Choose rows for interpolation based on the direction
        if vertical < 0:
            rows_to_interpolate_from = ws[:num_rows * 2, :]  # Use the first few rows
        else:
            rows_to_interpolate_from = ws[-num_rows * 2:, :]  # Use the last few rows
        
        # Generate x values for interpolation
        x = np.arange(rows_to_interpolate_from.shape[0])
        
        # Define positions for new rows to interpolate
        x_new = np.linspace(0, 1, num_rows, endpoint=False)
        
        # Create interpolation function
        interp_func = interp1d(x, rows_to_interpolate_from, axis=0, kind='linear')
        
        # Interpolate values for new rows
        new_rows = interp_func(x_new)*0
        new_rows[new_rows == 0] = np.nan
        
        # Modify the array based on the vertical direction
        if vertical < 0:
            # Add rows at the top and remove the same number from the bottom
            ws = np.vstack((new_rows, ws))
            ws = ws[:-num_rows, :]  # Remove rows from the bottom
        else:
            # Add rows at the bottom and remove the same number from the top
            ws = np.vstack((ws, new_rows))
            ws = ws[num_rows:, :]  # Remove rows from the top
    
    return ws

# Convert degrees to kilometers assuming a spherical Earth.
def deg2km(coords):
    """
    Convert degrees to kilometers assuming a spherical Earth.
    Coordinates should be tuples or lists like (longitude, latitude).
    """
    # 1 degree of latitude is approximately 111 km
    km_per_degree = 111.1949
    return np.array(coords) * km_per_degree

# Calculate the components of wind velocity (Vx, Vy).
def get_components(Vrot, O, X, Y, Rmax):
    """
    Calculate the components of wind velocity (Vx, Vy).
    
    Parameters:
    Vrot : numpy array - Rotational velocity
    O : numpy array - Center of the cyclone in kilometers (converted)
    X : numpy array - X-coordinates (longitudes converted to kilometers)
    Y : numpy array - Y-coordinates (latitudes converted to kilometers)
    Rmax : float - Radius of maximum wind in kilometers
    
    Returns:
    Vx : numpy array - Wind velocity components in the x-direction
    Vy : numpy array - Wind velocity components in the y-direction
    """
    rx = -(O[0] - X)
    ry = -(O[1] - Y)
    r = np.sqrt((O[0] - X) ** 2 + (O[1] - Y) ** 2)
    theta = np.zeros_like(X)
    
    # Conditions based on radius to calculate theta
    idx1 = r < Rmax
    theta[idx1] = 10 * r[idx1] / Rmax
    
    idx2 = (r >= Rmax) & (r < 1.2 * Rmax)
    theta[idx2] = 10 + 75 * (r[idx2] / Rmax - 1)
    
    idx3 = r >= 1.2 * Rmax
    theta[idx3] = 25

    # Calculate wind components
    Vx = Vrot * (-np.cos(np.deg2rad(theta)) * ry - np.sin(np.deg2rad(theta)) * rx) / r
    Vy = Vrot * (np.cos(np.deg2rad(theta)) * rx - np.sin(np.deg2rad(theta)) * ry) / r

    # Replace NaN values with 0
    Vx = np.nan_to_num(Vx, nan=0)
    Vy = np.nan_to_num(Vy, nan=0)
    
    return Vx, Vy

def winds(Pc, Vm, O, Lon, Lat):
    """
    Generate wind profile data using different formulations.
    
    Parameters:
    Pc : float - Cyclone central pressure in HPa
    Vm : float - Maximum sustained wind speed in knots
    O : list or tuple - Center of the cyclone (longitude, latitude)
    Lon : numpy array - Longitudes of the grid points
    Lat : numpy array - Latitudes of the grid points
    
    Returns:
    Vx : numpy array - Wind velocity components in the x-direction
    Vy : numpy array - Wind velocity components in the y-direction
    P : numpy array - Pressure at the grid points
    """
    rhoa = 1.225  # density of air
    # Convert Vm from knots to m/s
    Vm = Vm * 0.51
    # Get Rmax using Willoughby et al., 2005 formula
    Rmax = 46.4 * np.exp(-0.0155 * Vm + 0.0169 * O[1])
    
    # Convert points into kilometers
    O = deg2km(O)
    X = deg2km(Lon)
    Y = deg2km(Lat)
    r = np.sqrt((O[0] - X) ** 2 + (O[1] - Y) ** 2)
    k = Rmax / r
    
    # Get pressure
    Pn = 1013  # Ambient Pressure in HPa
    delP = Pn - Pc
    x = 0.9 * (1 - delP / 215)
    b = (Vm ** 2) * rhoa * np.exp(1) / (delP * 100)
    
    Vrot = Vm * (k ** b * np.exp(1 - k ** b)) ** x
    Vx, Vy = get_components(Vrot, O, X, Y, Rmax)
    P = Pc + delP * np.exp(-k ** b)
    
    return Vx, Vy, P, k

#% Smooth Blending
def calculate_UB(UP, UE, k):
    """
    This function calculates UB based on smooth blending of mod_wind and vir_wind,
    depending on the distance r and maximum radius Rmax.
    
    Parameters:
    mod_wind (numpy.ndarray): The modified wind value (2D array).
    vir_wind (numpy.ndarray): The vortex induced wind value (2D array).
    r (numpy.ndarray): The radial distance from the center (2D array).
    Rmax (numpy.ndarray): The maximum radius (2D array).
    
    Returns:
    numpy.ndarray: The blended UB value (2D array).
    """
    # Calculate alpha using NumPy for array-based operations
    #alpha = ((7 - r) / Rmax) / 5
    
    ck = k>=0.2 # Identify the virtual wind area
    ck_max = np.maximum(UP[ck],UE[ck])
    # Clip alpha to ensure values remain between 0 and 1 to avoid invalid values in powers
    UP[ck] = ck_max
    UB = UP
    return UB

#% Smooth Blending with u and v
def calculate_smooth_ws_u_v(mod_wind,vir_wind,k,u_m1,v_m1,Vx,Vy):
    """
    This function calculates UB based on smooth blending of mod_wind and vir_wind,
    depending on the distance r and maximum radius Rmax.
    
    Parameters:
    mod_wind (numpy.ndarray): The modified wind value (2D array).
    vir_wind (numpy.ndarray): The vortex induced wind value (2D array).
    r (numpy.ndarray): The radial distance from the center (2D array).
    Rmax (numpy.ndarray): The maximum radius (2D array).
    
    Returns:
    numpy.ndarray: The blended UB value (2D array).
    """
    # Calculate alpha using NumPy for array-based operations
    #alpha = ((7 - r) / Rmax) / 5
    
    ck = k>=0.2 # Identify the virtual wind area
    #ck_max = np.maximum(mod_wind[ck],vir_wind[ck])
    # Clip alpha to ensure values remain between 0 and 1 to avoid invalid values in powers
    #mod_wind[ck] = ck_max
    u_max= np.maximum(Vx[ck],u_m1[ck])
    v_max= np.maximum(Vy[ck],v_m1[ck])
    u_m1[ck] = u_max
    v_m1[ck] = v_max
    ws1 = np.sqrt(u_m1**2+v_m1**2)
    #chk = np.round(mod_wind,4) - np.round(ws1,4)
    
    
    return ws1, u_m1,v_m1


#%% Load the dataset
ds = xr.open_dataset(appended_data)
# Coordinates
lat = np.array(ds['latitude'])
lon = np.array(ds['longitude'])
lon2d, lat2d = np.meshgrid(lon, lat)
all_time = ds['time'].to_pandas()
#% Analysis
ws_m= []
u_m= []
v_m = []
pressure_m = []
time_m = []
virtual_wind = [] 
virtual_u= []
virtual_v = []
virtual_pressure = []
Blended_wind = []
for i in range(len(all_time)):
    #% Extract the variables
    #ws = np.array(ds['ws'].isel(time=i)) # Example: selecting the first time step
    u = np.array(ds['u10'].isel(time=i))  
    v = np.array(ds['v10'].isel(time=i))
    ws = np.sqrt(u**2+v**2)
    pressure = np.array(ds['pressure'].isel(time=i))  
    #% Get the Time
    time = str(ds['time'].isel(time=i).to_pandas())
    date_format = '%Y-%m-%dT%H:%M:%S'
    datetime_obj = datetime.strptime(time[:19], date_format)
    datetime_obj = datetime_obj.strftime('%Y-%m-%d %H:%M:%S')
    #% Find the time in geodatabase
    gdf_imd_union['D_Time'] = gdf_imd_union['D_Time'].astype(str)
    matching_row = gdf_imd_union[gdf_imd_union['D_Time']==datetime_obj].reset_index(drop=True)
    #% Extract latitude and longitude if there is a match
    if not matching_row.empty:        
        cyc_index = gdf_imd_union[gdf_imd_union['D_Time']==datetime_obj].index
        check_Ocean = (gdf_imd_union['Part'][cyc_index].values).tolist()
        if check_Ocean[0]=='Ocean':
            match = 'yes' 
            print('IMD Information is available. Date:' + str(datetime_obj))
        else:
            match = 'no' 
            print('Land Fall Occurred. Date:' + str(datetime_obj))
    else:
        print('No matching datetime found. Date:' + str(datetime_obj))
        match = 'no'
    
    if match == 'yes':
        horizontal = (gdf_imd_union['Horizontal'][cyc_index].values).tolist()
        vertical = (gdf_imd_union['Vertical'][cyc_index].values).tolist()
        mod_wind = modify_array(ws, horizontal, vertical)
        ws_m.append(mod_wind)
        u_m1 = modify_array(u, horizontal, vertical)
        v_m1 = modify_array(v, horizontal, vertical)
        pressure_m.append(modify_array(pressure, horizontal, vertical))
        time_m.append(datetime_obj)
        #%  Create Willoughby wind
        c_lat = np.double(gdf_imd_union['Lat'][cyc_index].values[0])
        c_lon = np.double(gdf_imd_union['Lon'][cyc_index].values[0])
        c_Pressure = np.double(gdf_imd_union['ECP'][cyc_index].values[0])
        c_Windmax = np.double(gdf_imd_union['Max_Wind'][cyc_index].values[0])
        [Vx, Vy, P, k] = winds(c_Pressure, c_Windmax, [c_lon, c_lat], lon2d, lat2d)
        vir_wind = np.sqrt(Vx**2+Vy**2)
        virtual_wind.append(vir_wind)
        virtual_u.append(Vx)
        virtual_v.append(Vy)
        virtual_pressure.append(P)
        
        blend_max,u_mx,v_my = calculate_smooth_ws_u_v(mod_wind.copy(),vir_wind.copy(),k.copy(),u_m1.copy(),v_m1.copy(),Vx.copy(),Vy.copy())
        Blended_wind.append(blend_max)
        u_m.append(u_mx)
        v_m.append(v_my)
    else:
        time_m.append(datetime_obj)
        u_m.append(u)
        ws_m.append(ws)
        v_m.append(v)
        pressure_m.append(pressure)
        virtual_wind.append(ws)
        virtual_u.append(u)
        virtual_v.append(v)
        virtual_pressure.append(pressure)
        Blended_wind.append(ws)
        
time_m = np.array(time_m, dtype='datetime64[ns]')

# Calculate time in hours since 1901-01-15
origin = np.datetime64('1901-01-15T00:00:00')
val_time_hours = (time_m - origin) / np.timedelta64(1, 'h')

# Create xarray Dataset
ds1 = xr.Dataset(
    {
        "u10": (["time", "latitude", "longitude"], u_m),
        "v10": (["time", "latitude", "longitude"], v_m),
        "IMD_Shifted_Wind": (["time", "latitude", "longitude"], ws_m),
        "pressure": (["time", "latitude", "longitude"], pressure_m),
        "Virtual_u10": (["time", "latitude", "longitude"], virtual_u),
        "Virtual_v10": (["time", "latitude", "longitude"], virtual_v),
        "Virtual_Wind": (["time", "latitude", "longitude"], virtual_wind),
        "Virtual_pressure": (["time", "latitude", "longitude"],virtual_pressure),
        "Blended_Wind": (["time", "latitude", "longitude"],Blended_wind),
    },
    coords={
        "latitude": lat,
        "longitude": lon,
        "time": val_time_hours
    }
)

# Set the units attribute for the time coordinate
ds1["time"].attrs["units"] = "hours since 1901-01-15"
ds1["time"].attrs["ref"] = "hours since 1901-01-15"

# Save to NetCDF file
output_file = r"D:/VIGNESH\WIndBlending_REDO/Data/ECMWF_Mocha_cy_8To16_may2023_mr_apr23tomay31/Mocha/Modified_Mocha_wind_data.nc"
#output_file = "D:/VIGNESH/WindBlending/Biparjoy/Modified_Biparjoy_wind_data.nc"
ds1.to_netcdf(output_file)
