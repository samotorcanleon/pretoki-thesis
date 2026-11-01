import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import hidro_and_meteo_helper_functions as sf
import xarray as xr
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import logging
from multiprocess import Pool
from sklearn.preprocessing import MaxAbsScaler
import gin
# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Merged ALADIN forecasts, produced by process_aladin_files.ipynb
ALADIN_PATH = "data/ALADIN/aladin_merged_ds.nc"

@gin.configurable
class TOKDataset(Dataset):

    aladin_raw_latitude = np.array([45.202, 45.242, 45.282, 45.322, 45.362, 45.402, 45.442, 45.482, 45.522,
                                        45.562, 45.602, 45.642, 45.682, 45.722, 45.762, 45.802, 45.842, 45.882,
                                        45.922, 45.962, 46.002, 46.042, 46.082, 46.122, 46.162, 46.202, 46.242,
                                        46.282, 46.322, 46.362, 46.402, 46.442, 46.482, 46.522, 46.562, 46.602,
                                        46.642, 46.682, 46.722, 46.762, 46.802, 46.842, 46.882, 46.922, 46.962,
                                        47.002, 47.042])
    aladin_raw_longitude = np.array([13.05 , 13.107, 13.164, 13.221, 13.278, 13.335, 13.392, 13.449, 13.506,
                                        13.563, 13.62 , 13.677, 13.734, 13.791, 13.848, 13.905, 13.962, 14.019,
                                        14.076, 14.133, 14.19 , 14.247, 14.304, 14.361, 14.418, 14.475, 14.532,
                                        14.589, 14.646, 14.703, 14.76 , 14.817, 14.874, 14.931, 14.988, 15.045,
                                        15.102, 15.159, 15.216, 15.273, 15.33 , 15.387, 15.444, 15.501, 15.558,
                                        15.615, 15.672, 15.729, 15.786, 15.843, 15.9  , 15.957, 16.014, 16.071,
                                        16.128, 16.185, 16.242, 16.299, 16.356, 16.413, 16.47 , 16.527, 16.584,
                                        16.641, 16.698, 16.755, 16.812, 16.869, 16.926, 16.983, 17.04 ])
    """
    Dataset class for TOK model based on DATA_linreg structure.
    """
    def model_data_load(self):
        """Load ALADIN meteorological model data"""
        long, lat = sf.get_hidro_station_loc(self.hidro_station_id)

        # Find closest longitude and latitude
        lon_idx = np.abs(self.aladin_raw_longitude - long).argmin()
        lat_idx = np.abs(self.aladin_raw_latitude - lat).argmin()

        # Take (5,5) area around it
        min_lat = lat_idx - self.expand_lat_lon_area[0]
        max_lat = lat_idx + self.expand_lat_lon_area[0] + 1
        min_lon = lon_idx - self.expand_lat_lon_area[1]
        max_lon = lon_idx + self.expand_lat_lon_area[1] + 1

        self.ALADIN = xr.open_dataset(ALADIN_PATH,
                                     decode_timedelta=True
                                     ).isel(lat=slice(min_lat, max_lat),
                                           lon=slice(min_lon, max_lon),
                                           step=slice(0, 24)
                                           ).compute()
        self.ALADIN = self.ALADIN.sel(time=slice(f"{self.from_year}-01-01", f"{self.up_to_year}-01-01"))
        self.ALADIN = self.ALADIN.drop_duplicates('time')

        if 'timedelta' not in str(self.ALADIN.step.dtype):
            self.ALADIN = self.ALADIN.assign_coords(step=pd.to_timedelta(self.ALADIN.step, unit='h'))
        self.DATA = self.ALADIN

        self.DATA = self.DATA.sortby('time')
        
        return self.DATA
    
    def meteo_data_load(self):
        """Load meteorological station data"""
        self.meteo_data = sf.get_k_closest_meteo_stations(self.hidro_station_id, self.num_closest_meteo_stations)
        return self.meteo_data

    @staticmethod
    def process_item(args):
        """Process a single item to check validity"""
        import numpy as np
        self_obj, date, hour_offset = args
        item = self_obj.get_item_by_date_and_prediction_hour_offset(date, hour_offset)
        if item is None:
            return None
        for key, value in item.items():
            if np.isnan(value).any():
                self_obj.logger.debug(f"Invalid item: {date}, {key} is nan")
                return None
        return item

    def __init__(self, station_id, from_year, up_to_year,
                 meteo_past_hours=48, 
                 river_history_hours=4,
                 prediction_hours=18, 
                 expand_lat_lon_area=(2, 2),
                 reduce_features_over_lat_lon=False,
                 generate_extra_data_points=False,
                 run_parallel=True,
                 predefined_scalers=None,
                 num_closest_meteo_stations=3):
        """
        Initialize the TOK dataset.
        
        Args:
            station_id (int): ID of the hydrological station
            from_year (int): Start year for data
            up_to_year (int): End year for data
            meteo_past_hours (int): Number of past hours for meteorological data
            river_history_hours (int): Number of past hours for river flow data
            prediction_hours (int): Number of hours to predict into the future
            expand_lat_lon_area (tuple): Area to expand around the station (lat, lon)
            reduce_features_over_lat_lon (bool): Whether to reduce features over lat/lon
            generate_extra_data_points (bool): Whether to generate extra data points
            run_parallel (bool): Whether to run processing in parallel
            predefined_scalers: Predefined scalers for normalization
            num_closest_meteo_stations (int): Number of closest meteorological stations to use
        """
        self.hidro_station_id = station_id
        self.from_year = from_year
        self.up_to_year = up_to_year
        self.meteo_past_hours = meteo_past_hours
        self.river_history_hours = river_history_hours
        self.prediction_hours = prediction_hours
        self.expand_lat_lon_area = expand_lat_lon_area
        self.reduce_features_over_lat_lon = reduce_features_over_lat_lon
        self.generate_extra_data_points = generate_extra_data_points
        self.run_parallel = run_parallel
        self.num_closest_meteo_stations = num_closest_meteo_stations
        self.logger = logger
        print("Dataset Parameters:")
        print(f"Station ID: {self.hidro_station_id}")
        print(f"Time Range: {self.from_year} to {self.up_to_year}")
        print(f"Meteo Past Hours: {self.meteo_past_hours}")
        print(f"River History Hours: {self.river_history_hours}")
        print(f"Prediction Hours: {self.prediction_hours}")
        print(f"Expand Lat/Lon Area: {self.expand_lat_lon_area}")
        print(f"Reduce Features Over Lat/Lon: {self.reduce_features_over_lat_lon}")
        print(f"Generate Extra Data Points: {self.generate_extra_data_points}")
        print(f"Run Parallel: {self.run_parallel}")
        print(f"Number of Closest Meteo Stations: {self.num_closest_meteo_stations}")

        if self.expand_lat_lon_area != (0,0) and not self.reduce_features_over_lat_lon:
            self.logger.info("Features are kept for each lat and lon, if you want to reduce features over lat and lon, set reduce_features_over_lat_lon to True")

        # Load data in appropriate format
        self.meteo_data = self.meteo_data_load()
        self.DATA = self.model_data_load()
        self.station_Q = sf.get_Q_df_for_station(self.hidro_station_id).sort_index()

        # Normalize data
        if predefined_scalers is None:
            self.scalers = {}
            self.scalers['tp'] = MinMaxScaler().fit(self.DATA.tp.values.reshape(-1,1))
            self.DATA.tp.values = self.scalers['tp'].transform(self.DATA.tp.values.reshape(-1,1)).reshape(self.DATA.tp.values.shape)
            # Normalize river flow data (Q) to scale between [0,1] using MaxAbsScaler
            self.scalers['Q'] = MaxAbsScaler().fit(self.station_Q['Q'].values.reshape(-1,1))
            self.station_Q['Q'] = self.scalers['Q'].transform(self.station_Q['Q'].values.reshape(-1,1)).reshape(-1)
            print(self.scalers['Q'].scale_[0])
        else:
            self.scalers = predefined_scalers
            self.DATA.tp.values = self.scalers['tp'].transform(self.DATA.tp.values.reshape(-1,1)).reshape(self.DATA.tp.values.shape)
            self.station_Q['Q'] = self.scalers['Q'].transform(self.station_Q['Q'].values.reshape(-1,1)).reshape(-1)

        # Training example is defined by hour at which we want to predict
        potential_items = self.DATA.time.values
        
        # Check if we can create a training example for each time, and save training examples into a list
        self.valid_items = []
        
        # Define hour range based on whether we want to generate extra data points
        hour_range = range(8) if self.generate_extra_data_points else [7]

        # Process items to check validity
        items_to_process = [(self, date, hour) for date in potential_items for hour in hour_range]
        if self.run_parallel:
            num_processes = 8
            with Pool(num_processes) as pool:
                results = list(tqdm(pool.imap(TOKDataset.process_item, items_to_process, 
                                             chunksize=min(1000, len(items_to_process)//num_processes+1)), 
                                   total=len(items_to_process), 
                                   desc="Checking valid items"))
        else:
            results = [TOKDataset.process_item(item) for item in tqdm(items_to_process, desc="Checking valid items")]
        
        self.valid_items = [item for item in results if item is not None]
        
        self.logger.info(f"Tested {len(potential_items)*len(hour_range)} items, {len(self.valid_items)} are valid, which is {len(self.valid_items) / (len(potential_items)*len(hour_range)) * 100:.2f}%")
        
        # Get station and river name for reference
        self.station_name, self.river_name = sf.get_station_and_river_name(station_id)
        
        print(f"Dataset created for {self.station_name} on {self.river_name}")

    def __len__(self):
        return len(self.valid_items)

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        return self.valid_items[idx]
    
    def get_model_data_values(self, date, prediction_hour_offset=7):
        """Get model data values for the specified date and prediction hour offset."""
        import pandas as pd
        import numpy as np
        
        def get_field_data(field_name, field_data):
            """Helper function to get data for a specific field."""
            try:
                prediction_day_data = field_data.sel(time=date).isel(step=slice(0, 24+min(0, prediction_hour_offset-7)))[field_name].values
            except KeyError:
                self.logger.debug(f'No prediction day data for {date} and field {field_name}')
                return None
            try:
                previous_day_data = field_data.sel(time=date - pd.Timedelta(days=1)).isel(step=slice(0+max(0,prediction_hour_offset-7), 24))[field_name].values
            except KeyError:
                self.logger.debug(f'No previous day data for {date} and field {field_name}')
                return None
            
            field_values = np.concatenate([previous_day_data, prediction_day_data], axis=0)
            field_values = field_values.reshape(-1, *field_values.shape[-2:]) # flatten time and step keep lat and lon
            
            if prediction_hour_offset > 7:
                try:
                    next_day_data = field_data.sel(time=date + pd.Timedelta(days=1)).isel(step=slice(0, prediction_hour_offset-7))[field_name].values
                except KeyError:
                    self.logger.debug(f'No next day data for {date} and field {field_name}')
                    return None       
                next_day_data = next_day_data.reshape(-1, *next_day_data.shape[-2:]) # flatten time and step keep lat and lon
                field_values = np.concatenate([field_values, next_day_data], axis=0)
            elif prediction_hour_offset < 7:
                # Take data from 2 days before for offsets < 7
                try:
                    two_days_before = field_data.sel(time=date - pd.Timedelta(days=2)).isel(step=slice(24+(prediction_hour_offset-7), 24))[field_name].values
                except KeyError:
                    self.logger.debug(f'No two days before data for {date} and field {field_name}')
                    return None
                two_days_before = two_days_before.reshape(-1, *two_days_before.shape[-2:]) # flatten time and step keep lat and lon
                field_values = np.concatenate([two_days_before, field_values], axis=0)    

            _, lat, lon = field_values.shape
            if field_values.shape != (48, lat, lon):
                self.logger.debug(f'Field {field_name} data values shape is not (48, lat, lon), but {field_values.shape}')
                return None
            
            return field_values
        
        # Get data for all fields in the dataset
        all_field_data = {}
        for field_name in self.DATA.data_vars:
            field_data = get_field_data(field_name, self.DATA)
            if field_data is None:
                return None, None
            all_field_data[field_name] = field_data
        
        # Split each field into past and future data
        past_model_data = {}
        future_model_data = {}
        for field_name, field_values in all_field_data.items():
            past_model_data[field_name] = field_values[:24, :, :]
            future_model_data[field_name] = field_values[24:, :, :]
        
        return past_model_data, future_model_data


    def get_item_by_date_and_prediction_hour_offset(self, date, prediction_hour_offset=7):
        """
        Generate item by date and prediction hour offset.
        Default prediction hour offset is 7 hours, which is what time ARSO predicts for.
        But we can generate more data by also taking into account data around that time.
        """
        import pandas as pd
        import numpy as np
        
        past_model_data, future_model_data = self.get_model_data_values(date, prediction_hour_offset)
        if past_model_data is None or future_model_data is None:
            return None

        # Instead of keeping every lat and lon, we sum over them with one value
        #if self.reduce_features_over_lat_lon:
        #    past_model_data = past_model_data.sum(axis=(-1,-2)).reshape(-1,1,1)
        #    future_model_data = future_model_data.sum(axis=(-1,-2)).reshape(-1,1,1)

        # Flow we want to predict - date is at 00:00, to match ARSO predictions, first prediction is at 07:00
        first_prediction_hour = date + pd.Timedelta(hours=prediction_hour_offset)
        future_Q = self.station_Q.loc[first_prediction_hour: first_prediction_hour + pd.Timedelta(hours=self.prediction_hours-1)]
        
        # Flow for past hours
        past_Q = self.station_Q.loc[first_prediction_hour - pd.Timedelta(hours=self.river_history_hours): first_prediction_hour - pd.Timedelta(hours=1)]
        
        # Meteo data features
        meteo_data_past_24h = self.meteo_data.loc[first_prediction_hour - pd.Timedelta(hours=12): first_prediction_hour - pd.Timedelta(hours=1)]
        if meteo_data_past_24h.isna().values.any() or meteo_data_past_24h.shape[0] != 12:
            self.logger.info(f"Meteo data past 24h is not the same shape as 24, {meteo_data_past_24h.shape}")
            return None

        # Flatten future_Q and past_Q
        future_Q = future_Q.values.flatten()
        past_Q = past_Q.values.flatten()

        if len(future_Q) != self.prediction_hours or len(past_Q) != self.river_history_hours:
            if len(future_Q) != self.prediction_hours:
                self.logger.debug('future Q is not the same length as prediction hours')
            if len(past_Q) != self.river_history_hours:
                self.logger.debug('past Q is not the same length as flow past hours')
            return None

        # Prepare data in TOK format
        # 1. River history data
        river_history = torch.tensor(past_Q, dtype=torch.float32).reshape(1, -1)
        
        # 2. Target data
        target = torch.tensor(future_Q, dtype=torch.float32).reshape(1, -1)
        
        # 3. Meteorological data (keeping station data and model data separate)
        # Extract data for each station and reshape
        # Convert station data directly to tensor
        station_data_values = meteo_data_past_24h.values  # shape: (24, num_stations)
        station_data_values = station_data_values.transpose()  # shape: (num_stations, 24)
        meteorological_station_data = torch.tensor(station_data_values, dtype=torch.float32)

        return {
            'meteorological_station_data': meteorological_station_data,
            'past_model_data_tp': torch.tensor(past_model_data['tp'], dtype=torch.float32),
            'future_model_data_tp': torch.tensor(future_model_data['tp'], dtype=torch.float32),
            'past_model_data_t': torch.tensor(past_model_data['t2m'], dtype=torch.float32),
            'future_model_data_t': torch.tensor(future_model_data['t2m'], dtype=torch.float32),
            'past_model_data_r': torch.tensor(past_model_data['r'], dtype=torch.float32),
            'future_model_data_r': torch.tensor(future_model_data['r'], dtype=torch.float32),
            'past_model_data_u10': torch.tensor(past_model_data['u10'], dtype=torch.float32),
            'future_model_data_u10': torch.tensor(future_model_data['u10'], dtype=torch.float32),
            'past_model_data_v10': torch.tensor(past_model_data['v10'], dtype=torch.float32),
            'future_model_data_v10': torch.tensor(future_model_data['v10'], dtype=torch.float32),
            'past_model_data_diffT500hPa': torch.tensor(past_model_data['diffT500hPa'], dtype=torch.float32),
            'future_model_data_diffT500hPa': torch.tensor(future_model_data['diffT500hPa'], dtype=torch.float32),
            'river_history': river_history,
            'target': target,
            'prediction_timestamp': pd.Timestamp(first_prediction_hour - pd.DateOffset(hours=1)).timestamp()
        }

    def get_scalers(self):
        """Get a copy of the scalers used for normalization."""
        return self.scalers.copy()



if __name__ == "__main__":
    1+1
    dd = TOKDataset(4200, from_year=2020, up_to_year=2021, prediction_hours=18, expand_lat_lon_area=(1, 1), reduce_features_over_lat_lon=False, generate_extra_data_points=False, run_parallel=False)
    1+1


def create_tok_dataloaders(station_id, batch_size=64, 
                          train_up_to_year=2020,
                          meteo_past_hours=48, prediction_hours=18, 
                          river_history_hours=12,
                          num_closest_meteo_stations=3):
    """
    Create train, validation and test dataloaders for the TOK model.
    
    Args:
        station_id (int): ID of the hydrological station
        batch_size (int): Batch size for dataloaders
        train_up_to_year (int): Last year to include in training set (exclusive)
        from_year (int): Start year for training data
        meteo_past_hours (int): Number of past hours for meteorological data
        prediction_hours (int): Number of hours to predict into the future
        river_history_hours (int): Number of past hours for river flow data
        num_closest_meteo_stations (int): Number of closest meteorological stations to use
        
    Returns:
        tuple: (train_dataloader, val_dataloader, test_dataloader)
    """
    from_year = 2012
    # Fixed values for validation and test years
    from_year_validation = train_up_to_year
    from_year_test = train_up_to_year + 1
    
    # Create datasets for each split
    common_params = {
        'station_id': station_id,
        'meteo_past_hours': meteo_past_hours,
        'prediction_hours': prediction_hours,
        'river_history_hours': river_history_hours,
        'num_closest_meteo_stations': num_closest_meteo_stations
    }
    
    train_dataset = TOKDataset(
        from_year=from_year,
        up_to_year=train_up_to_year,
        **common_params
    )
    
    val_dataset = TOKDataset(
        from_year=from_year_validation,
        up_to_year=from_year_validation + 1,
        predefined_scalers=train_dataset.scalers,
        **common_params
    )
    
    test_dataset = TOKDataset(
        from_year=2021,
        up_to_year=2024,
        predefined_scalers=train_dataset.scalers,
        **common_params
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_dataloader, val_dataloader, test_dataloader



def create_cv_dataloaders(station_id, batch_size=64,
                          train_up_to_year=2020,
                          meteo_past_hours=48, prediction_hours=18, 
                          river_history_hours=12,
                          num_closest_meteo_stations=3, 
                          generate_extra_data_points=False):
    """
    Create train, test model split for cross validation
    """
    common_params = {
        'meteo_past_hours': meteo_past_hours,
        'prediction_hours': prediction_hours,
        'river_history_hours': river_history_hours,
        'num_closest_meteo_stations': num_closest_meteo_stations
    }

    dataset_train_up_to_2021 = TOKDataset(station_id, 2012, up_to_year=2021, **common_params,
                                            generate_extra_data_points=generate_extra_data_points)
    dataset_test_2021 = TOKDataset(station_id, 2021, up_to_year=2022, **common_params,
                                    generate_extra_data_points=False, predefined_scalers=dataset_train_up_to_2021.get_scalers())
    dataset_train_up_to_2022 = TOKDataset(station_id, 2012, up_to_year=2022, **common_params,
                                           generate_extra_data_points=generate_extra_data_points)
    dataset_test_2022 = TOKDataset(station_id, 2022, up_to_year=2023, **common_params,
                                    generate_extra_data_points=False, predefined_scalers=dataset_train_up_to_2022.get_scalers())
    dataset_train_up_to_2023 = TOKDataset(station_id, 2012, up_to_year=2023, **common_params,
                                           generate_extra_data_points=generate_extra_data_points)
    dataset_test_2023 = TOKDataset(station_id, 2023, up_to_year=2024, **common_params,
                                    generate_extra_data_points=False, predefined_scalers=dataset_train_up_to_2023.get_scalers())

    # Create dataloaders
    dataloader_train_up_to_2021 = DataLoader(
        dataset_train_up_to_2021, 
        batch_size=batch_size, 
        shuffle=True
    )
    dataloader_test_2021 = DataLoader(
        dataset_test_2021, 
        batch_size=batch_size, 
        shuffle=False
    )
    dataloader_train_up_to_2022 = DataLoader(
        dataset_train_up_to_2022, 
        batch_size=batch_size, 
        shuffle=True
    )
    dataloader_test_2022 = DataLoader(
        dataset_test_2022,  
        batch_size=batch_size, 
        shuffle=False
    )
    dataloader_train_up_to_2023 = DataLoader(
        dataset_train_up_to_2023, 
        batch_size=batch_size, 
        shuffle=True
    )
    dataloader_test_2023 = DataLoader(
        dataset_test_2023, 
        batch_size=batch_size, 
        shuffle=False
    )
    cv_splits = [(dataloader_train_up_to_2021, dataloader_test_2021),
             (dataloader_train_up_to_2022, dataloader_test_2022),
             (dataloader_train_up_to_2023, dataloader_test_2023)]
    return cv_splits
    
    
