import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error 
import pandas as pd

def plot_predictions_analysis(y_test, y_pred, arso_pred=None, title=None):
    # Calculate RMSE for each hour
    num_hours = y_pred.shape[1]  # Assuming y_pred is 2D with shape (num_samples, num_hours)

    hourly_rmse = []
    max_diff = []  # List to store max differences for each hour
    
    # For ARSO predictions if provided
    arso_hourly_rmse = []
    arso_max_diff = []

    for hour in range(num_hours):
        rmse = root_mean_squared_error(y_test[:, hour], y_pred[:, hour])
        hourly_rmse.append(rmse)
        
        # Calculate max difference between prediction and ground truth for the current hour
        max_difference = np.max(np.abs(y_test[:, hour] - y_pred[:, hour]))
        max_diff.append(max_difference)
        
        # Calculate ARSO metrics if provided
        if arso_pred is not None:
            arso_rmse = root_mean_squared_error(y_test[:, hour], arso_pred[:, hour])
            arso_hourly_rmse.append(arso_rmse)
            
            arso_max_difference = np.max(np.abs(y_test[:, hour] - arso_pred[:, hour]))
            arso_max_diff.append(arso_max_difference)

    # Calculate maximum flow value for ground truth
    max_flow = y_test.max(axis=0)

    # Plotting Hourly RMSE, Maximum Flow Value, and Max Differences in subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))  # Made the plot slightly larger

    # Plot RMSE
    axs[0].plot(range(1, num_hours + 1), hourly_rmse, marker='o', linestyle='-', color='b', label='Model RMSE')
    if arso_pred is not None:
        axs[0].plot(range(1, num_hours + 1), arso_hourly_rmse, marker='s', linestyle='--', color='orange', label='ARSO RMSE')
    axs[0].set_title('Hourly Root Mean Squared Error')
    axs[0].set_xlabel('Hour')
    axs[0].set_ylabel('RMSE')
    axs[0].set_xticks(range(1, num_hours + 1))
    axs[0].set_ylim(0, max(hourly_rmse + (arso_hourly_rmse if arso_pred is not None else [])) * 1.1)
    axs[0].grid()
    axs[0].legend()

    # Plot Max Flow Value
    axs[1].plot(range(1, num_hours + 1), max_flow, marker='x', linestyle='--', color='r', label='Max Flow Value')
    axs[1].set_title('Maximum Flow Value')
    axs[1].set_xlabel('Hour')
    axs[1].set_ylabel('Max Flow Value')
    axs[1].set_xticks(range(1, num_hours + 1))
    axs[1].set_ylim(0, max_flow.max() * 1.1)
    axs[1].grid()
    axs[1].legend()

    # Plot Max Differences
    axs[2].plot(range(1, num_hours + 1), max_diff, marker='o', linestyle='-', color='g', label='Model Max Difference')
    if arso_pred is not None:
        axs[2].plot(range(1, num_hours + 1), arso_max_diff, marker='s', linestyle='--', color='purple', label='ARSO Max Difference')
    axs[2].set_title('Max Differences Between Predictions and Ground Truth')
    axs[2].set_xlabel('Hour')
    axs[2].set_ylabel('Max Difference')
    axs[2].set_xticks(range(1, num_hours + 1))
    axs[2].set_ylim(0, max(max_diff + (arso_max_diff if arso_pred is not None else [])) * 1.1)
    axs[2].grid()
    axs[2].legend()

    if title:
        fig.suptitle(title, fontsize=16)
        
    fig.tight_layout()
    return fig


def plot_biggest_miss(y_test, y_pred, prediction_timestamp_test, arso_pred=None, title=None):
    num_hours = y_test.shape[1]
    # Calculate the absolute errors
    absolute_errors = np.abs(y_test - y_pred)

    # Find the index of the biggest miss
    biggest_miss_index = np.argmax(absolute_errors)

    # Get the corresponding timestamp for the biggest miss
    timestamp_index = biggest_miss_index // num_hours
    prediction_timestamp = prediction_timestamp_test[timestamp_index]

    # Plotting the biggest miss
    fig = plt.figure(figsize=(12, 6))
    plt.plot(range(num_hours), y_test[timestamp_index], marker='o', linestyle='-', label='True Value')
    plt.plot(range(num_hours), y_pred[timestamp_index], marker='x', linestyle='--', label='Model Prediction')
    
    if arso_pred is not None:
        plt.plot(range(num_hours), arso_pred[timestamp_index], marker='s', linestyle='-.', label='ARSO Prediction')

    plt.title('Biggest Miss of the Model' if title is None else title)
    plt.xlabel('Hour')
    plt.ylabel('Value')
    plt.xticks(range(num_hours), [pd.to_datetime(prediction_timestamp) + pd.Timedelta(hours=i) for i in range(1,num_hours+1)], rotation=45)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    return fig


# Calculate NS (Nash-Sutcliffe efficiency), Per (Pearson Correlation Coefficient), and dQs (difference in predicted vs ground truth) per hour
def calculate_metrics(y_true, y_pred):
    # Calculate Nash-Sutcliffe efficiency
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((y_true - np.mean(y_true)) ** 2)
    ns = 1 - (numerator / denominator) if denominator != 0 else np.nan

    # Calculate Pearson Correlation Coefficient
    # Using numpy's corrcoef function which returns the Pearson correlation coefficient
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    per = correlation  # This is already the Pearson correlation coefficient

    # Calculate dQs (difference in predicted vs ground truth)
    dQs = np.sum(y_pred) / np.sum(y_true)

    return ns, per, dQs

import pandas as pd

def plot_metrics(y_test, y_pred, arso_pred=None, show_plots=True, title=None):
    # Initialize lists to store metrics
    ns_per_hour = []
    per_corr_per_hour = []
    dQs_per_hour = []
    
    # For ARSO predictions if provided
    arso_ns_per_hour = []
    arso_per_corr_per_hour = []
    arso_dQs_per_hour = []

    # Extract num_hours from the shape of y_test
    num_hours = y_test.shape[1]

    # Calculate metrics for each hour
    for hour in range(num_hours):
        y_true_hour = y_test[:, hour]
        y_pred_hour = y_pred[:, hour]
        
        ns, per, dQs = calculate_metrics(y_true_hour, y_pred_hour)
        ns_per_hour.append(ns)
        per_corr_per_hour.append(per)
        dQs_per_hour.append(dQs)
        
        # Calculate ARSO metrics if provided
        if arso_pred is not None:
            arso_pred_hour = arso_pred[:, hour]
            arso_ns, arso_per, arso_dQs = calculate_metrics(y_true_hour, arso_pred_hour)
            arso_ns_per_hour.append(arso_ns)
            arso_per_corr_per_hour.append(arso_per)
            arso_dQs_per_hour.append(arso_dQs)

    # Convert to numpy arrays for easier handling
    ns_per_hour = np.array(ns_per_hour)
    per_corr_per_hour = np.array(per_corr_per_hour)
    dQs_per_hour = np.array(dQs_per_hour)
    
    if arso_pred is not None:
        arso_ns_per_hour = np.array(arso_ns_per_hour)
        arso_per_corr_per_hour = np.array(arso_per_corr_per_hour)
        arso_dQs_per_hour = np.array(arso_dQs_per_hour)

    # Create a DataFrame to return the metrics
    metrics_df = pd.DataFrame({
        'Hour': np.arange(1, num_hours + 1),
        'Nash-Sutcliffe Efficiency': ns_per_hour,
        'Pearson Correlation Coefficient': per_corr_per_hour,
        'dQs': dQs_per_hour
    })
    
    if arso_pred is not None:
        metrics_df['ARSO Nash-Sutcliffe Efficiency'] = arso_ns_per_hour
        metrics_df['ARSO Pearson Correlation Coefficient'] = arso_per_corr_per_hour
        metrics_df['ARSO dQs'] = arso_dQs_per_hour

    # Define maximum values for capping
    max_ns = 1.0
    max_per = 1.0

    # Plotting the metrics if show_plots is True
    if show_plots:
        fig = plt.figure(figsize=(12, 8))

        # Nash-Sutcliffe Efficiency
        plt.subplot(3, 1, 1)
        plt.plot(np.arange(1, num_hours + 1), ns_per_hour, marker='o', linestyle='-', label='Model NSE')
        if arso_pred is not None:
            plt.plot(np.arange(1, num_hours + 1), arso_ns_per_hour, marker='s', linestyle='--', label='ARSO NSE')
        plt.title('Nash-Sutcliffe Efficiency per Hour')
        plt.xlabel('Hour')
        plt.ylabel('NSE')
        plt.xticks(np.arange(1, num_hours + 1))
        
        # Calculate min/max for y-axis limits
        all_ns = ns_per_hour.tolist() + (arso_ns_per_hour.tolist() if arso_pred is not None else [])
        min_ns = min(0.6, np.floor(min(all_ns) * 10) / 10) if all_ns else 0.6
        max_ns_val = min(max_ns, np.ceil(max(all_ns) * 10) / 10) + 0.04 if all_ns else 1.0
        
        plt.ylim(min_ns, max_ns_val)
        plt.grid()
        plt.legend()

        # Pearson Correlation Coefficient
        plt.subplot(3, 1, 2)
        plt.plot(np.arange(1, num_hours + 1), per_corr_per_hour, marker='o', linestyle='-', label='Model Pearson', color='orange')
        if arso_pred is not None:
            plt.plot(np.arange(1, num_hours + 1), arso_per_corr_per_hour, marker='s', linestyle='--', label='ARSO Pearson', color='green')
        plt.title('Pearson Correlation Coefficient per Hour')
        plt.xlabel('Hour')
        plt.ylabel('Pearson Correlation Coefficient')
        plt.xticks(np.arange(1, num_hours + 1))
        
        # Calculate min/max for y-axis limits
        all_per = per_corr_per_hour.tolist() + (arso_per_corr_per_hour.tolist() if arso_pred is not None else [])
        min_per = min(0.6, np.floor(min(all_per) * 10) / 10) if all_per else 0.6
        max_per_val = min(max_per, np.ceil(max(all_per) * 10) / 10) + 0.04 if all_per else 1.0
        
        plt.ylim(min_per, max_per_val)
        plt.grid()
        plt.legend()

        # Difference in Predicted vs Ground Truth (dQs)
        plt.subplot(3, 1, 3)
        plt.plot(np.arange(1, num_hours + 1), dQs_per_hour, marker='o', linestyle='-', label='Model dQs', color='purple')
        if arso_pred is not None:
            plt.plot(np.arange(1, num_hours + 1), arso_dQs_per_hour, marker='s', linestyle='--', label='ARSO dQs', color='brown')
        plt.title('Difference in Predicted vs Ground Truth (dQs) per Hour')
        plt.xlabel('Hour')
        plt.ylabel('dQs')
        plt.xticks(np.arange(1, num_hours + 1))
        
        # Calculate min/max for y-axis limits
        all_dqs = dQs_per_hour.tolist() + (arso_dQs_per_hour.tolist() if arso_pred is not None else [])
        min_dqs = np.floor(min(all_dqs) * 100) / 100 if all_dqs else 0
        max_dqs = np.ceil(max(all_dqs) * 100) / 100 if all_dqs else 1
        
        plt.ylim(min_dqs, max_dqs)
        plt.grid()
        plt.legend()

        if title:
            plt.suptitle(title, fontsize=16)  # Add title to the entire figure if provided

        plt.tight_layout()  # Adjust layout to prevent clipping of tick-labels
    return fig, metrics_df


def plot_common(station_id, y_test, y_pred, prediction_timestamp_test, show_plots=False):
    import ARSO_model
    arso_model = ARSO_model.ARSO_model(station_id)
    arso_prediction_dates = arso_model.get_valid_dates()
    # Find common dates between ARSO predictions and the test timestamps
    evaluation_dates = sorted(list(set(arso_prediction_dates).intersection(set(prediction_timestamp_test))))

    arso_model_predictions = arso_model.get_prediction_for_dates(evaluation_dates)
    y_test_filtered = y_test[np.isin(prediction_timestamp_test, evaluation_dates)]
    y_pred_filtered = y_pred[np.isin(prediction_timestamp_test, evaluation_dates)]
    
    # Plot with both model and ARSO predictions
    fig_predictions_analysis = plot_predictions_analysis(y_test_filtered, y_pred_filtered, arso_model_predictions, 
                                   title=f"Station {station_id} - Model vs ARSO")
    if show_plots:
        plt.show()
    
    # Plot metrics with both model and ARSO predictions
    fig_metrics, metrics_df = plot_metrics(y_test_filtered, y_pred_filtered, arso_model_predictions, 
                                  show_plots=True, title=f"Station {station_id} - Model vs ARSO")
    if show_plots:
        plt.show()
    
    # Plot biggest miss with both model and ARSO predictions
    fig_biggest_miss = plot_biggest_miss(y_test_filtered, y_pred_filtered, 
                           [prediction_timestamp_test[i] for i in range(len(prediction_timestamp_test)) 
                            if prediction_timestamp_test[i] in evaluation_dates],
                           arso_model_predictions, title=f"Station {station_id} - Biggest Miss")
    if show_plots:
        plt.show()

    arso_predictions_df = pd.DataFrame(arso_model_predictions, index=evaluation_dates).reset_index()
    model_predictions_df = pd.DataFrame(y_pred_filtered, index=evaluation_dates).reset_index()
    actuals_df = pd.DataFrame(y_test_filtered, index=evaluation_dates).reset_index()
    
    return metrics_df, fig_predictions_analysis, fig_metrics, fig_biggest_miss, (arso_predictions_df, model_predictions_df, actuals_df)




