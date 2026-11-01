import os
import pandas as pd
import json
import numpy as np


def get_Q_df_for_station(station_id):
    predecessors_df = pd.read_csv('data/ARSO_podatki_2/20250110/Hidro/aktivneVP_predhodnice.csv', sep=',', names=['aktivna', 'predhonica1','predhodnica2','predhodnica3'], header=0)
    predecessors_df = predecessors_df[predecessors_df['aktivna'] == station_id]
    relevant_stations = predecessors_df[['aktivna','predhonica1','predhodnica2','predhodnica3']].values.flatten()
    relevant_stations = relevant_stations[relevant_stations != -1]
    #print(relevant_stations)
    # before 2022
    station_data_folder = "data/ARSO_podatki_2/20250110/Hidro/pov_arhiv_Q_urni/pov_arhiv_Q_urni"
    files = os.listdir(station_data_folder)
    files = [f for f in files if any([f.startswith(str(station_id)) for station_id in relevant_stations])]
    dfs = []
    for f in files:
        df = pd.read_csv(os.path.join(station_data_folder, f), sep=';', parse_dates=[0]).rename(columns={'datum': 'date', 'vrednost': 'Q'})
        dfs.append(df)

    # after 2022
    station_data_folder_raw = "data/ARSO_podatki_2/20250110/Hidro/pov_surovi_HQ_urni/pov_surovi_HQ_urni"
    files_raw = os.listdir(station_data_folder_raw)
    files_raw = [f for f in files_raw if any([f.startswith(str(station_id)) for station_id in relevant_stations])]
    for f in files_raw:
        df = pd.read_csv(os.path.join(station_data_folder_raw, f), sep=';', parse_dates=[0]).drop(columns=['H']).rename(columns={'datum': 'date'})
        dfs.append(df)

    # combine all data
    Q_df = pd.concat(dfs)
    Q_df = Q_df.set_index('date')
    Q_df = Q_df.sort_index()
    # Drop duplicate indices (timestamps)
    Q_df = Q_df[~Q_df.index.duplicated(keep='first')]
    #print("Number of NaNs before interpolation: ", Q_df['Q'].isna().sum())



    #
    # Interpolate missing values, where the missing values are closely surrounded by valid values
    #
    date_examples_reindexed = Q_df.copy()
    date_examples_reindexed.index = pd.DatetimeIndex(date_examples_reindexed.index)
    start_time = date_examples_reindexed.index.min()
    end_time = date_examples_reindexed.index.max()

    # Create a complete hourly index
    complete_hourly_index = pd.date_range(start=start_time, end=end_time, freq='h')

    # Reindex the data to identify gaps
    filled_examples = date_examples_reindexed.reindex(complete_hourly_index)

    # Fill small gaps (max 7 h) using linear interpolation
    # Only fill gaps that are surrounded by valid values on both sides
    # First identify gaps of length 1
    # Create a mask for NaN values that are at most 3 positions away from non-NaN values
    mask = filled_examples['Q'].isna() & (
        # Check up to 3 positions before
        (filled_examples['Q'].shift(1).notna() | 
        filled_examples['Q'].shift(2).notna() | 
        filled_examples['Q'].shift(3).notna()) &
        # Check up to 3 positions after
        (filled_examples['Q'].shift(-1).notna() |
        filled_examples['Q'].shift(-2).notna() |
        filled_examples['Q'].shift(-3).notna())
    )
    # Create a temporary series with all values (including non-NaN) for interpolation
    temp_series = filled_examples['Q'].copy()
    # Apply interpolation to the entire series
    temp_interpolated = temp_series.interpolate(method='linear')
    # Only update the NaN values that match our mask criteria
    filled_examples.loc[mask, 'Q'] = temp_interpolated.loc[mask]

    filled_examples = filled_examples.dropna()
        
    

    return filled_examples

def get_hidro_station_loc(hidro_id):
    with open('data/ARSO_podatki_2/20250110/Hidro/hid_arhiv.geojson', encoding='utf-8') as f:
        hid_arhiv = json.load(f)

    hidro_station_coordinates = {}
    for feature in hid_arhiv['features']:
        hidro_station_coordinates[int(feature['properties']['id'])] = np.array(feature['geometry']['coordinates'])

    return hidro_station_coordinates[hidro_id]

def get_meteo_df_for_station(station_id):
    # before 2022
    station_data_folder = "data/ARSO_podatki_2/20250110/Meteo/padavine/rr"
    files = os.listdir(station_data_folder)
    files = [f for f in files if f == f"{station_id}_rr.txt"]
    if len(files) == 0:
        raise ValueError(f"No files found for station {station_id}")
    elif len(files) > 1:
        print(files)
        raise ValueError(f"Multiple files found for station {station_id}")
    file = files[0]
    meteo_df = pd.read_csv(os.path.join(station_data_folder, file), sep=',', parse_dates=[0], usecols=[0,1]).rename(columns={'datum': 'date', 't2m':'rainfall'})
    meteo_df = meteo_df.set_index('date')
    meteo_df = meteo_df.sort_index()

    # Interpolate missing values
    date_examples_reindexed = meteo_df.copy()
    date_examples_reindexed.index = pd.DatetimeIndex(date_examples_reindexed.index)
    start_time = date_examples_reindexed.index.min()
    end_time = date_examples_reindexed.index.max()

    # Create complete hourly index
    complete_hourly_index = pd.date_range(start=start_time, end=end_time, freq='h')

    # Reindex data to identify gaps
    filled_examples = date_examples_reindexed.reindex(complete_hourly_index)

    # Fill small gaps using linear interpolation
    for col in filled_examples.columns:
        mask = filled_examples[col].isna() & (
            (filled_examples[col].shift(1).notna() | 
            filled_examples[col].shift(2).notna() | 
            filled_examples[col].shift(3).notna()) &
            (filled_examples[col].shift(-1).notna() |
            filled_examples[col].shift(-2).notna() |
            filled_examples[col].shift(-3).notna())
        )
        temp_series = filled_examples[col].copy()
        temp_interpolated = temp_series.interpolate(method='linear')
        filled_examples.loc[mask, col] = temp_interpolated.loc[mask]

    filled_examples = filled_examples.dropna()
    
    return filled_examples

def get_all_meteo_stations_loc():
    file = pd.read_excel("data/ARSO_podatki_2/20250110/Meteo/metapodatki.xlsx")
    file.columns = file.columns.str.strip()
    file = file.rename(columns={'____ime_postaje______': 'ime_postaje'})
    file['id'] = file['id'].astype(int)
    file['ge_dol_wgs84'] = file['ge_dol_wgs84'].astype(float)
    file['ge_sir_wgs84'] = file['ge_sir_wgs84'].astype(float)

    station_data_folder = "data/ARSO_podatki_2/20250110/Meteo/padavine/rr"
    files = os.listdir(station_data_folder)
    # Get list of valid station IDs from files
    valid_station_ids = set()
    for station_id in file['id']:
        if f"{station_id}_rr.txt" in files:
            valid_station_ids.add(station_id)
    # Filter file to only include stations that have data files
    file = file[file['id'].isin(valid_station_ids)]
    file = file.drop_duplicates(subset=['id'])
    # remove duplicates
    return file

def get_meteo_station_loc(station_id):
    file = get_all_meteo_stations_loc()
    file = file[file['id'] == station_id]
    return file['ge_dol_wgs84'].values[0], file['ge_sir_wgs84'].values[0]

def get_k_closest_meteo_stations(hidro_station_id, k):
    # get location of hidro station
    hidro_station_loc = get_hidro_station_loc(hidro_station_id)
    # get all meteo stations loc
    all_meteo_stations_loc = get_all_meteo_stations_loc()
    
    # Calculate haversine distance between hidro station and all meteo stations
    R = 6371  # Earth's radius in kilometers
    
    # Convert latitude/longitude to radians
    lat1 = np.radians(hidro_station_loc[1])
    lon1 = np.radians(hidro_station_loc[0])
    lat2 = np.radians(all_meteo_stations_loc['ge_sir_wgs84'])
    lon2 = np.radians(all_meteo_stations_loc['ge_dol_wgs84'])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    all_meteo_stations_loc['distance'] = R * c
    
    # get k closest meteo stations
    # Sort stations by distance
    # Sort stations by distance and take top 3
    closest_meteo_stations = all_meteo_stations_loc.sort_values(by='distance').head(k+10)
    # Get data for each station and concatenate horizontally
    station_dfs = []
    for station_id in closest_meteo_stations['id']:
        df = get_meteo_df_for_station(station_id)
        station_dfs.append(df.rename(columns={'rainfall': f'rainfall_{station_id}'}))
    # Concatenate all station data
    df = pd.concat(station_dfs, axis=1)['2011':]
    
    # Create empty dataframe for first k non-NaN values per row
    result = pd.DataFrame(index=df.index, columns=[f'rainfall_{i+1}' for i in range(k)])
    # Get first k non-NaN values for each row using numpy operations
    nan_mask = df.isna()
    non_nan_indices = nan_mask.values.argsort(axis=1, kind='stable')[:, :k]
    row_indices = np.arange(len(df))[:, None]
    non_nan_values = df.values[row_indices, non_nan_indices]
    
    # Assign values to result columns
    for i in range(k):
        result[f'rainfall_{i+1}'] = non_nan_values[:, i]
    closest_meteo_stations = result
    # Forward fill NaN values
    closest_meteo_stations = closest_meteo_stations.ffill(axis=1)

    return closest_meteo_stations#[['id', 'distance','ge_sir_wgs84', 'ge_dol_wgs84', 'ime_postaje']]

def get_station_and_river_name(station_id:int):
    with open('data/ARSO_podatki_2/20250110/Hidro/hid_arhiv.geojson', encoding='utf-8') as f:
        hid_arhiv = json.load(f)
    for feature in hid_arhiv['features']:
        if feature['properties']['id'] == str(station_id):
            return feature['properties']['name'], feature['properties']['river']
    return None, None







