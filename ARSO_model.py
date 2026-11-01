import os
import pandas as pd
import numpy as np

def get_Q_ARSO_sim_for_station(station_id):

    station_data_folder = "data/ARSO_podatki_2/20250110/Hidro/simHPS_Q_2021-23/"
    files_raw = os.listdir(station_data_folder)
    files_raw = [f for f in files_raw if any([f.startswith(str(station_id))])]
    dfs = []
    for f in files_raw:
        df = pd.read_csv(os.path.join(station_data_folder, f), sep=';', parse_dates=[0]).rename(columns={'tos': 'date','fh':'step' , 'value': 'Q'})
        dfs.append(df)

    # combine all data
    Q_df = pd.concat(dfs)
    Q_df['date'] = pd.to_datetime(Q_df['date'], format="%d.%m.%Y %H:%M")
    
    print(f"Number of NaNs for arso station {station_id}", Q_df['Q'].isna().sum())
    Q_df = Q_df.dropna()
    Q_df = Q_df.sort_values(by=['date', 'step'])
    Q_df = Q_df.set_index('date')
    Q_df = Q_df.sort_index()

    Q_df.index = Q_df.index - pd.DateOffset(hours=1)
    Q_df['step'] = Q_df['step'] + 1
    Q_df['forecast_time'] = Q_df.index + pd.to_timedelta(Q_df['step'], unit='h') # Column containing times for which the prediction values are for the future
    Q_df.index.name = 'prediction_date'  # when was the prediction made

    Q_df['Q'] = Q_df['Q'].str.replace(',', '.').astype(float)

    return Q_df

class ARSO_model:
    def __init__(self, station_id, prediction_num_of_hours=18):
        self.station_id = station_id
        self.data = get_Q_ARSO_sim_for_station(self.station_id)
        self.prediction_num_of_hours = prediction_num_of_hours

    def get_prediction_for_date(self, date):
        # For current purpose the date should always be year-month-day 06:00:00
        if self.data is not None:
            return self.data.loc[date].query(f'step <= @self.prediction_num_of_hours')
        else:
            raise ValueError(f"Data for {date} does not exist")
        

    def get_prediction_for_dates(self, dates):
        predictions = np.array([self.get_prediction_for_date(date)['Q'].values for date in dates])
        return predictions
    
    def get_valid_dates(self):
        return self.data.index.unique()
