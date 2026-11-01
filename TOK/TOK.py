import torch
import torch.nn as nn
import torch.nn.functional as F
import gin

@gin.configurable
class SimpleEncoder(nn.Module):
    """A simplified encoder for processing 2D meteorological data"""
    def __init__(self, input_channels, hidden_dim=128, output_dim=256, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim*2, output_dim, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # x shape: [batch_size, channels, height, width]
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        return x.view(x.size(0), -1)  # Flatten to [batch_size, output_dim]

@gin.configurable
class TemporalBlockEncoder(nn.Module):
    """A simplified temporal-aware encoder that processes weather data in meaningful time blocks."""
    def __init__(self, input_channels, block_hours=4, hidden_dim=128, output_dim=256, dropout_rate=0.1):
        super().__init__()
        self.block_hours = block_hours
        # input_channels refers to total hours for this modality (e.g., 24 hours for ALADIN data)
        if input_channels % block_hours != 0:
            raise ValueError(f"input_channels ({input_channels}) must be divisible by block_hours ({block_hours}).")
        self.num_blocks = input_channels // block_hours
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Spatial processing for each block
        # Input: [B*num_blocks, block_hours, H, W]
        # Output: [B*num_blocks, hidden_dim, 1, 1]
        self.block_spatial_processor = nn.Sequential(
            nn.Conv2d(self.block_hours, self.hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Temporal processing across blocks
        # Input to Conv1d: [B, hidden_dim, num_blocks]
        # Output of Conv1d: [B, output_dim, num_blocks]
        self.temporal_conv = nn.Conv1d(self.hidden_dim, self.output_dim, kernel_size=3, padding=1)
        self.temporal_dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # x shape: [batch_size, total_hours, height, width]
        batch_size, total_hours, height, width = x.shape
        
        if total_hours != self.num_blocks * self.block_hours:
            raise ValueError(
                f"Input tensor total_hours ({total_hours}) does not match "
                f"num_blocks ({self.num_blocks}) * block_hours ({self.block_hours}) based on init params. "
                f"Ensure input_channels used during init corresponds to total_hours."
            )

        # Reshape for block-wise spatial processing:
        # From [B, total_hours, H, W] to [B, num_blocks, block_hours, H, W]
        x_reshaped = x.view(batch_size, self.num_blocks, self.block_hours, height, width)
        
        # Combine batch and num_blocks dimensions for batched spatial processing:
        # [B * num_blocks, block_hours, H, W]
        x_for_spatial_proc = x_reshaped.reshape(-1, self.block_hours, height, width)

        # Apply spatial processor to all blocks:
        # Output: [B * num_blocks, hidden_dim, 1, 1]
        block_features_pooled = self.block_spatial_processor(x_for_spatial_proc)
        
        # Flatten pooled features: [B * num_blocks, hidden_dim]
        block_features_flat = block_features_pooled.view(batch_size * self.num_blocks, -1)
        
        # Reshape for temporal processing: [B, num_blocks, hidden_dim]
        temporal_input = block_features_flat.view(batch_size, self.num_blocks, self.hidden_dim)
        
        # Transpose for Conv1d (expects channels as the second dimension):
        # [B, hidden_dim, num_blocks]
        temporal_input_for_conv1d = temporal_input.transpose(1, 2)
        
        # Apply temporal convolution:
        # Output: [B, output_dim, num_blocks] (padding=1 and kernel_size=3 maintain sequence length)
        processed_temporal = F.relu(self.temporal_conv(temporal_input_for_conv1d))
        processed_temporal = self.temporal_dropout(processed_temporal)
        
        # Global average pooling over the temporal dimension (num_blocks):
        # Output: [B, output_dim]
        final_features = processed_temporal.mean(dim=2)
        
        return final_features

@gin.configurable
class WindEncoder(nn.Module):
    """A specialized encoder for processing both U and V wind components together.
    This makes sense because U and V wind components are physically coupled - they represent
    the horizontal wind vector components and should be processed together to capture
    wind patterns, direction changes, and vector magnitude information."""
    def __init__(self, input_channels, block_hours=4, hidden_dim=128, output_dim=256, dropout_rate=0.1):
        super().__init__()
        self.block_hours = block_hours
        # input_channels refers to total hours for each wind component (e.g., 24 hours)
        if input_channels % block_hours != 0:
            raise ValueError(f"input_channels ({input_channels}) must be divisible by block_hours ({block_hours}).")
        self.num_blocks = input_channels // block_hours
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Spatial processing for each block with 2 channels (U and V components)
        # Input: [B*num_blocks, 2*block_hours, H, W] (2 channels for U and V)
        # Output: [B*num_blocks, hidden_dim, 1, 1]
        self.block_spatial_processor = nn.Sequential(
            nn.Conv2d(2 * self.block_hours, self.hidden_dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Temporal processing across blocks
        # Input to Conv1d: [B, hidden_dim, num_blocks]
        # Output of Conv1d: [B, output_dim, num_blocks]
        self.temporal_conv = nn.Conv1d(self.hidden_dim, self.output_dim, kernel_size=3, padding=1)
        self.temporal_dropout = nn.Dropout(dropout_rate)
        
    def forward(self, u_component, v_component):
        # u_component and v_component shapes: [batch_size, total_hours, height, width]
        batch_size, total_hours, height, width = u_component.shape
        
        if total_hours != self.num_blocks * self.block_hours:
            raise ValueError(
                f"Input tensor total_hours ({total_hours}) does not match "
                f"num_blocks ({self.num_blocks}) * block_hours ({self.block_hours}) based on init params."
            )

        # Combine U and V components along the channel dimension
        # Stack: [batch_size, 2, total_hours, height, width]
        combined_wind = torch.stack([u_component, v_component], dim=1)
        
        # Reshape for block-wise processing:
        # From [B, 2, total_hours, H, W] to [B, 2, num_blocks, block_hours, H, W]
        combined_wind_reshaped = combined_wind.view(batch_size, 2, self.num_blocks, self.block_hours, height, width)
        
        # Rearrange to [B, num_blocks, 2*block_hours, H, W] for spatial processing
        combined_wind_blocks = combined_wind_reshaped.permute(0, 2, 1, 3, 4, 5).contiguous()
        combined_wind_blocks = combined_wind_blocks.view(batch_size, self.num_blocks, 2 * self.block_hours, height, width)
        
        # Combine batch and num_blocks dimensions for batched spatial processing:
        # [B * num_blocks, 2*block_hours, H, W]
        x_for_spatial_proc = combined_wind_blocks.reshape(-1, 2 * self.block_hours, height, width)

        # Apply spatial processor to all blocks:
        # Output: [B * num_blocks, hidden_dim, 1, 1]
        block_features_pooled = self.block_spatial_processor(x_for_spatial_proc)
        
        # Flatten pooled features: [B * num_blocks, hidden_dim]
        block_features_flat = block_features_pooled.view(batch_size * self.num_blocks, -1)
        
        # Reshape for temporal processing: [B, num_blocks, hidden_dim]
        temporal_input = block_features_flat.view(batch_size, self.num_blocks, self.hidden_dim)
        
        # Transpose for Conv1d (expects channels as the second dimension):
        # [B, hidden_dim, num_blocks]
        temporal_input_for_conv1d = temporal_input.transpose(1, 2)
        
        # Apply temporal convolution:
        # Output: [B, output_dim, num_blocks] (padding=1 and kernel_size=3 maintain sequence length)
        processed_temporal = F.relu(self.temporal_conv(temporal_input_for_conv1d))
        processed_temporal = self.temporal_dropout(processed_temporal)
        
        # Global average pooling over the temporal dimension (num_blocks):
        # Output: [B, output_dim]
        final_features = processed_temporal.mean(dim=2)
        
        return final_features

@gin.configurable
class TOK(nn.Module):
    """Simplified TOK (Time-series Observation and Knowledge) model"""
    def __init__(self, hidden_dim=256,
                 prediction_layer_hidden_dim=1024, dropout_rate=0.1,
                 use_temporal_blocks=True, block_hours=4):
        super().__init__()
        self.prediction_hours = 18
        self.dropout_rate = dropout_rate
        self.use_temporal_blocks = use_temporal_blocks
        print(f"hidden_dim: {hidden_dim}, prediction_layer_hidden_dim: {prediction_layer_hidden_dim}, dropout_rate: {dropout_rate}, use_temporal_blocks: {use_temporal_blocks}, block_hours: {block_hours}")
        
        # Choose encoder type based on configuration
        if use_temporal_blocks:
            # Temporal-aware encoders for different data types
            self.past_model_encoder_tp = TemporalBlockEncoder(24, block_hours=block_hours, hidden_dim=256, output_dim=128)
            self.future_model_encoder_tp = TemporalBlockEncoder(24, block_hours=block_hours)
            self.past_model_encoder_r = TemporalBlockEncoder(24, block_hours=block_hours, output_dim=32, hidden_dim=64)
            self.future_model_encoder_r = TemporalBlockEncoder(24, block_hours=block_hours, output_dim=32, hidden_dim=64)
            # Combined wind encoders for U and V components
            self.past_model_encoder_wind = WindEncoder(24, block_hours=block_hours, output_dim=32, hidden_dim=64)
            self.future_model_encoder_wind = WindEncoder(24, block_hours=block_hours, output_dim=32, hidden_dim=64)
            self.past_model_encoder_diffT500hPa = TemporalBlockEncoder(24, block_hours=block_hours, output_dim=32, hidden_dim=64)
            self.future_model_encoder_diffT500hPa = TemporalBlockEncoder(24, block_hours=block_hours, output_dim=32, hidden_dim=64)
        else:
            # Original simple encoders (fallback)
            self.past_model_encoder_tp = SimpleEncoder(24) # 24 h of ALADIN predictions
            self.future_model_encoder_tp = SimpleEncoder(24) # 24 h of past ALADIN predictions
            self.past_model_encoder_r = SimpleEncoder(24) # 24 h of ALADIN predictions
            self.future_model_encoder_r = SimpleEncoder(24) # 24 h of past ALADIN predictions
            self.past_model_encoder_u10 = SimpleEncoder(24) # 24 h of ALADIN predictions
            self.future_model_encoder_u10 = SimpleEncoder(24) # 24 h of past ALADIN predictions
            self.past_model_encoder_v10 = SimpleEncoder(24) # 24 h of ALADIN predictions
            self.future_model_encoder_v10 = SimpleEncoder(24) # 24 h of past ALADIN predictions
            self.past_model_encoder_diffT500hPa = SimpleEncoder(24) # 24 h of ALADIN predictions
            self.future_model_encoder_diffT500hPa = SimpleEncoder(24) # 24 h of past ALADIN predictions

        # We'll dynamically create the fusion layer in forward() after we know the exact dimensions
        self.hidden_dim = hidden_dim
        self.fusion_layer = None
        
        # Prediction layers
        self.prediction_layer1 = nn.Sequential(
            nn.Linear(hidden_dim+1, prediction_layer_hidden_dim),
            nn.ReLU()
        )
        
        self.prediction_layer2 = nn.Linear(prediction_layer_hidden_dim + 1, self.prediction_hours)
        
    def forward(self, data):
        batch_size = data['river_history'].shape[0]
        
        # Debug prints to see data shapes
        #for key in data:
        #    if isinstance(data[key], torch.Tensor):
        #        print(f"{key}: {data[key].shape}")
        
        # Process meteorological station data if available
        if 'meteorological_station_data' in data:
            meteo_data = data['meteorological_station_data']
            # Simple processing for meteo station data - flatten to [batch_size, stations*features]
            meteo_features = meteo_data.view(batch_size, -1)
        else:
            print("meteorological_station_data is none")
            meteo_features = torch.zeros(batch_size, 24).to(data['river_history'].device)
        
        # Process model data
        # Check and process past model data for each variable
        past_features = []
        if 'past_model_data_tp' in data:
            past_features.append(self.past_model_encoder_tp(data['past_model_data_tp']))
            
        if 'past_model_data_t2m' in data:
            # Sum average of temperatures in 3 8-hour blocks over whole area
            t2m_data = data['past_model_data_t2m']  # Shape: [batch_size, 24, lat, lon]
            # Reshape to [batch_size, 3, 8, lat, lon] and take mean over 8-hour blocks and spatial dimensions
            t2m_reshaped = t2m_data.view(batch_size, 3, 8, *t2m_data.shape[2:])
            t2m_block_means = t2m_reshaped.mean(dim=(2, 3, 4))  # Mean over 8-hour, lat, and lon dimensions
            # Result shape: [batch_size, 3] - one value per 8-hour block
            past_features.append(t2m_block_means)
            
        if 'past_model_data_r' in data:
            past_features.append(self.past_model_encoder_r(data['past_model_data_r']))
            
        # Process wind components together if both are available
        if 'past_model_data_u10' in data and 'past_model_data_v10' in data:
            if self.use_temporal_blocks:
                wind_features = self.past_model_encoder_wind(data['past_model_data_u10'], data['past_model_data_v10'])
                past_features.append(wind_features)
            else:
                # Fallback to separate processing for simple encoders
                past_features.append(self.past_model_encoder_u10(data['past_model_data_u10']))
                past_features.append(self.past_model_encoder_v10(data['past_model_data_v10']))

            
        if 'past_model_data_diffT500hPa' in data:
            past_features.append(self.past_model_encoder_diffT500hPa(data['past_model_data_diffT500hPa']))
            
        past_model_features = torch.cat(past_features, dim=1)
        
        # Check and process future model data for each variable
        future_features = []
        if 'future_model_data_tp' in data:
            future_features.append(self.future_model_encoder_tp(data['future_model_data_tp']))
            
        if 'future_model_data_t2m' in data:
            # Sum average of temperatures in 3 8-hour blocks over whole area
            t2m_data = data['future_model_data_t2m']  # Shape: [batch_size, 24, lat, lon]
            # Reshape to [batch_size, 3, 8, lat, lon] and take mean over 8-hour blocks and spatial dimensions
            t2m_reshaped = t2m_data.view(batch_size, 3, 8, *t2m_data.shape[2:])
            t2m_block_means = t2m_reshaped.mean(dim=(2, 3, 4))  # Mean over 8-hour, lat, and lon dimensions
            # Result shape: [batch_size, 3] - one value per 8-hour block
            future_features.append(t2m_block_means)
            
        if 'future_model_data_r' in data:
            future_features.append(self.future_model_encoder_r(data['future_model_data_r']))
            
        # Process wind components together if both are available
        if 'future_model_data_u10' in data and 'future_model_data_v10' in data:
            if self.use_temporal_blocks:
                wind_features = self.future_model_encoder_wind(data['future_model_data_u10'], data['future_model_data_v10'])
                future_features.append(wind_features)
            else:
                # Fallback to separate processing for simple encoders
                future_features.append(self.future_model_encoder_u10(data['future_model_data_u10']))
                future_features.append(self.future_model_encoder_v10(data['future_model_data_v10']))
            
        if 'future_model_data_diffT500hPa' in data:
            future_features.append(self.future_model_encoder_diffT500hPa(data['future_model_data_diffT500hPa']))
            
        future_model_features = torch.cat(future_features, dim=1)
        
        # Use river history directly as a vector
        river_features = data['river_history'].view(batch_size, -1)  # Flatten
        river_history_last_value = river_features[:,-1].unsqueeze(1)
        # Concatenate all features
        combined_features = torch.cat([
            past_model_features, 
            future_model_features, 
            river_features,
            meteo_features
        ], dim=1)
        
        #print(f"combined_features: {combined_features.shape}")
        
        # Create fusion layer dynamically if it doesn't exist or if input dimension has changed
        fusion_input_dim = combined_features.shape[1]
        if self.fusion_layer is None:
            self.fusion_layer = nn.Sequential(
                nn.Linear(fusion_input_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate)
            ).to(combined_features.device)
        
        # Fusion
        fused_features = self.fusion_layer(combined_features)

        # Add the last river history value before final prediction
        fused_features = torch.cat([fused_features, river_history_last_value], dim=1)
        # Prediction - split into two steps
        intermediate_features = self.prediction_layer1(fused_features)
        # Add the last river history value before final prediction
        intermediate_with_river = torch.cat([intermediate_features, river_history_last_value], dim=1)
        predictions = self.prediction_layer2(intermediate_with_river)
        predictions = predictions.view(batch_size, 1, self.prediction_hours)  # Single station
        
        # Ensure predictions are non-negative using ReLU
        if not self.training:
            predictions = torch.relu(predictions)
        return predictions


    def scaled_predictions(self,predictions):
        predictions = predictions.cpu().numpy()
        shape = predictions.shape
        predictions = predictions.reshape(-1,1)
        predictions = self.output_scaler.inverse_transform(predictions)
        predictions = predictions.reshape(shape)
        return predictions
