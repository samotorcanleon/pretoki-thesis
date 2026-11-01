# Pretoki: hourly river discharge forecasting for Slovenian rivers

Code for the master's thesis on forecasting river discharge (*pretok*) at ARSO gauging stations.

The TOK model predicts hourly discharge at one station for the next 18 hours. Its inputs are:

- the ALADIN weather forecast for the 5×5 grid cells around the station, both the current and the previous day's forecast: precipitation, relative humidity, 10 m wind, and the temperature difference between 2 m and 500 hPa
- the last 12 hours of rainfall at the nearest meteorological station
- the last 12 hours of discharge at the station

Each ALADIN field has its own convolutional encoder, which processes the 24 forecast hours in 4-hour blocks. The encoded fields are combined with the station rainfall and discharge history in fully connected layers.

A separate model is trained for each station. It is evaluated with three expanding-window folds: training starts in 2012, and the test years are 2021, 2022 and 2023. The comparison is against:

- ARSO's operational hydrological forecasts
- a per-station linear regression baseline

## Contents

| File | Purpose |
| --- | --- |
| `TOK/TOK.py` | The model |
| `TOK_dataset.py` | Builds training samples from ARSO and ALADIN data; train/test splits |
| `TOK_train.py` | Training entry point, W&B logging, station list, sweep configuration |
| `TOK_cv_train.py` | Cross-validation trainer used by `TOK_train.py` |
| `TOK_config.gin` | Training and model hyperparameters |
| `parameter_sweep.py` | Joins an existing W&B hyperparameter sweep as an agent |
| `hidro_and_meteo_helper_functions.py` | Loading of ARSO discharge, rainfall and station metadata |
| `ARSO_model.py` | Loads ARSO's own discharge forecasts for comparison |
| `plotting_and_evaluation_functions.py` | Metrics (NSE, Pearson, dQs, RMSE) and evaluation plots |
| `process_aladin_files.ipynb` | Raw ALADIN GRIB → regridded `data/ALADIN/aladin_merged_ds.nc` |
| `process_arso_vv_xml.ipynb` | `arso_vv.xml` → `arso_vv.parquet` (high-water discharge thresholds) |
| `pretoki_data_exploration.ipynb` | Exploration of the ARSO discharge data |
| `linearna_regresija_final.ipynb` | Linear regression baseline |
| `model_explanation.ipynb` | SHAP explanation of a trained model |
| `performance_graphs.ipynb` | Results and comparison figures, built from the W&B runs |
| `slike_magisterij.ipynb` | Other figures used in the thesis |
| `arso_vv.xml`, `arso_vv.parquet`, `hid_arhiv.geojson` | ARSO station metadata used by the notebooks |

## Setup

```bash
conda env create -f environment.yml
conda activate pretoki
wandb login
```

The thesis runs used Python 3.12 and a PyTorch 2.5 nightly build with CUDA 12.4. Training falls back to the CPU if no GPU is available.

## Data

The data comes from ARSO (Slovenian Environment Agency) and is not included in this repository. To get access to the data used in this study, contact the author, Leon Samotorcan (samotorcan.leon@gmail.com).

All code reads the data from a `data/` folder in the repository root. If you keep the data elsewhere, make `data` a link to it:
- Windows: `mklink /J data D:\path\to\data`
- Linux/macOS: `ln -s /path/to/data data`

```
data/
├── ARSO_podatki_2/20250110/
│   ├── Hidro/
│   │   ├── aktivneVP_predhodnice.csv               predecessor station IDs per active station (records are merged)
│   │   ├── hid_arhiv.geojson                       station locations, names and rivers
│   │   ├── pov_arhiv_Q_urni/pov_arhiv_Q_urni/      hourly discharge archive (before 2022)
│   │   ├── pov_surovi_HQ_urni/pov_surovi_HQ_urni/  raw hourly water level and discharge (2022 onwards)
│   │   └── simHPS_Q_2021-23/                       ARSO hydrological model forecasts, 2021–2023
│   └── Meteo/
│       ├── metapodatki.xlsx                        meteorological station metadata
│       ├── padavine/rr/                            hourly rainfall per station (<id>_rr.txt)
│       └── temperature/temperature/                hourly temperature per station (figures only)
├── reke/                                           older ARSO export, used by the exploration and figure notebooks
│   ├── aktivneVP_predhodnice.csv
│   └── POV/POV/<station>/{H,Q}/{urne,dnevne}/
├── ALADIN_COMPLETE/<year>/<month>/ashz_*.grb       raw ALADIN forecasts
└── ALADIN/                                         written by process_aladin_files.ipynb
    └── aladin_merged_ds.nc                         ~8.4 GB, the file training reads
```

## Running

1. **Preprocess ALADIN.** Run `process_aladin_files.ipynb` to create `data/ALADIN/aladin_merged_ds.nc`. The notebook's first cell explains the order the cells were run in.
2. **Train TOK.** Run these from the repository root:

   ```bash
   python TOK_train.py TOK_config.gin              # station from Config.station_id
   python TOK_train.py TOK_config.gin 4222 4230    # given stations
   python TOK_train.py TOK_config.gin all          # all 177 stations from the thesis
   ```

   Each station runs all three folds and logs the results to the W&B project set by `Config.wandb_project`. After each fold, the model is saved to `modelsSweep/final_model_<run_id>.pth`. Each fold overwrites the previous file, so the saved model is the one from the last fold.
3. **Train the baseline.** Run `linearna_regresija_final.ipynb`.
4. **Analyse the results.**
   - `performance_graphs.ipynb` reads the W&B projects `final_experiment_CNN` (TOK) and `Final Linear Regression` (baseline). To use your own runs, change `my_project` in the notebook. Figures are written to `figures/`.
   - `model_explanation.ipynb` explains one saved model. Set the station and the W&B run ID in its first cell.

## License

The code is released under the MIT License (see `LICENSE`). The ARSO and ALADIN data are not covered by this license.
