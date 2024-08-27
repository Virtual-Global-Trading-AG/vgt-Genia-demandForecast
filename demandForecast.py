import torch
from torch.utils.tensorboard import SummaryWriter
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class Preprocessor:
    """
    Splits data into training samples and generates categorical features.
    """
    columns_to_normalize = [
        'value_kwh',
        'dayOfYear',
        'dayOfWeek_0',
        'dayOfWeek_1',
        'dayOfWeek_2',
        'dayOfWeek_3',
        'dayOfWeek_4',
        'dayOfWeek_5',
        'dayOfWeek_6',
        'dayOfMonth',
        'timeOfDay'
    ]

    feature_ranges = {
        'dayOfYear': (1, 366),
        'dayOfMonth': (1, 31),
        'timeOfDay': (0, 24)
    }

    scaler = MinMaxScaler()

    def __init__(self, identifier:str, splits:tuple=(0.5, 0.25, 0.25)):
        assert isinstance(splits, tuple)
        assert sum(splits) == 1
        self.splits = splits

        assert isinstance(identifier, str)
        self.identifier = identifier

    def store_scaler_state(self):
        """Stores state of scaler in minimal dictionary."""
        scaler_state = {
            'min_': self.scaler.min_,
            'scale_': self.scaler.scale_,
            'data_min_': self.scaler.data_min_,
            'data_max_': self.scaler.data_max_
        }

        with open(f'{self.identifier}_state.pkl', 'wb') as f:
            pickle.dump(scaler_state, f)

    def load_scaler_state(self):
        """Loads scaler state."""
        with open(f'{self.identifier}_state.pkl', 'rb') as f:
            scaler_state = pickle.load(f)
        
        self.scaler.min_ = scaler_state['min_']
        self.scaler.scale_ = scaler_state['scale_']
        self.scaler.data_min_ = scaler_state['data_min_']
        self.scaler.data_max_ = scaler_state['data_max_']

    def preprocess(self, data:pd.DataFrame, fit:bool=True):
        """Preprocesses dataframe and optionally refits scaler."""
        assert isinstance(data, pd.DataFrame)
        assert set(['timestamp', 'value_kwh']) == set(data.columns)
    
        # generate categorical data
        data['dayOfYear'] = data.timestamp.dt.day_of_year
        data['dayOfWeek'] = data.timestamp.dt.day_of_week
        data['dayOfMonth'] = data.timestamp.dt.day
        data['timeOfDay'] = data.timestamp.dt.hour + (data.timestamp.dt.minute / 60)
        data = pd.get_dummies(data=data, columns=['dayOfWeek'])

        if fit:
            # set predefined min/max values for each column
            data_min, data_max = [], []

            for column in self.columns_to_normalize:
                if column in self.feature_ranges:
                    min_val, max_val = self.feature_ranges[column]
                else:
                    min_val, max_val = data[column].min(), data[column].max()

                data_min.append(min_val)
                data_max.append(max_val)
            
            # Manually set the scaler's internal attributes
            self.scaler.fit(data[self.columns_to_normalize]) # fitting just to avoid warning
            self.scaler.data_min_ = np.array(data_min)
            self.scaler.data_max_ = np.array(data_max)
            self.scaler.data_range_ = self.scaler.data_max_ - self.scaler.data_min_
            self.scaler.scale_ = (self.scaler.feature_range[1] - self.scaler.feature_range[0]) / self.scaler.data_range_
            self.scaler.min_ = self.scaler.feature_range[0] - self.scaler.data_min_ * self.scaler.scale_

        else:
            self.scaler.fit(data[self.columns_to_normalize]) # fitting just to avoid warning
            self.load_scaler_state()

        # Apply scaling
        data[self.columns_to_normalize] = self.scaler.transform(data[self.columns_to_normalize])

        # generate training, validation and test split
        end_of_training = int(data.shape[0] * self.splits[0])
        end_of_validation = int(data.shape[0] * (self.splits[0] + self.splits[1]))
        self.training = data.iloc[:end_of_training,:].copy().reset_index(drop=True)
        self.validation = data.iloc[end_of_training:end_of_validation, :].copy().reset_index(drop=True)
        self.test = data.iloc[end_of_validation:, :].copy().reset_index(drop=True)

        # remove timestamp columns from splits
        del self.training['timestamp']
        del self.validation['timestamp']
        del self.test['timestamp']

        assert self.validation.shape[0] > 21 * 96 and self.test.shape[0] > 21 * 96, "Dataset is too small for training!"

        # store state of Scaler
        self.store_scaler_state()
        
        return data
    
class Dataset(torch.utils.data.Dataset):
    def __init__(self, data:pd.DataFrame):
        assert isinstance(data, pd.DataFrame), "Data mus be a pd.DataFrame"
        assert set(['value_kwh', 'dayOfYear', 'dayOfMonth', 'timeOfDay',
       'dayOfWeek_0', 'dayOfWeek_1', 'dayOfWeek_2', 'dayOfWeek_3',
       'dayOfWeek_4', 'dayOfWeek_5', 'dayOfWeek_6']) == set(data.columns)
        self.data = data
        self.sample_length = (7 * 96)
        self.prediction_length = (14 * 96)
        self.n_samples = self.data.shape[0] - (self.prediction_length + self.sample_length)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index:int):
        # select data
        sample = self.data.iloc[index:index+self.sample_length,:].values
        target = self.data.loc[index+self.sample_length:index+self.sample_length+self.prediction_length-1, 'value_kwh'].values
        exogenous_features = self.data.loc[index+self.sample_length:index+self.sample_length+self.prediction_length-1, ['dayOfYear', 'dayOfMonth', 'timeOfDay',
       'dayOfWeek_0', 'dayOfWeek_1', 'dayOfWeek_2', 'dayOfWeek_3',
       'dayOfWeek_4', 'dayOfWeek_5', 'dayOfWeek_6']].values

        # transform to tensor
        sample = torch.tensor(sample, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        exogenous_features = torch.tensor(exogenous_features, dtype=torch.float32)
        
        return sample, exogenous_features, target

class PredictionModel(torch.nn.Module):
    """
    Uses numerical and categorical features before current timestep to generate a prediction.
    """
    def __init__(self, layers:torch.nn.Sequential):
        super(PredictionModel, self).__init__()
        assert isinstance(layers, torch.nn.Sequential)
        self.layers = layers
    
    def forward(self, samples:torch.tensor):
        return self.layers(samples)

class CorrectionModel(torch.nn.Module):
    """Uses prediction from PredictionModel and known categorical features of future timesteps to generate a prediction."""
    def __init__(self, layers:torch.nn.Sequential):
        super(CorrectionModel, self).__init__()
        assert isinstance(layers, torch.nn.Sequential)
        self.layers = layers

    def forward(self, samples:torch.tensor):
        return self.layers(samples)

class DemandForecast(torch.nn.Module):
    """
    Combines Prediction Model and Correction Model to generate a demand forecast
    """
    def __init__(
            self,
            prediction_model:PredictionModel,
            correction_model:CorrectionModel,
            optimizer:str,
            lr:float=1e-4
            ):
        super(DemandForecast, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        assert isinstance(prediction_model, PredictionModel)
        assert isinstance(correction_model, CorrectionModel)
        self.prediction_model = prediction_model
        self.correction_model = correction_model
        self.to(device=self.device)

        assert isinstance(optimizer, str)
        assert isinstance(lr, float) and lr > 0
        self.lr = lr

        if optimizer == 'adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            raise ValueError("Optimizer not implemented")
        
    def forward(self, samples:torch.tensor, exogenous_features:torch.tensor):
        samples = samples.unsqueeze(1).to(device=self.device)
        exogenous_features = exogenous_features.to(device=self.device)
        output = self.prediction_model.forward(samples)
        output = self.correction_model.forward(torch.concat((exogenous_features, output.unsqueeze(2)), dim=2).unsqueeze(1))
        return output
    
    def learn(self,
          max_epochs: int,
          training_loader: torch.utils.data.DataLoader,
          validation_loader: torch.utils.data.DataLoader,
          test_loader: torch.utils.data.DataLoader,
          lr: float = 1e-4,
          log_dir: str = './runs'):
        """Execute training with given training-, validation- and testLoader and log with TensorBoard"""
        
        # Define loss criterion
        criterion = torch.nn.MSELoss()  # Assuming we are using Mean Squared Error Loss

        # Define optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        
        # Move model to the appropriate device
        self.to(self.device)
        
        # Set up TensorBoard writer
        writer = SummaryWriter(log_dir=log_dir)

        # Training loop
        for epoch in range(max_epochs):
            self.train()  # Set model to training mode
            total_train_loss = 0

            for samples, exogenous_features, target in training_loader:
                # Zero the parameter gradients
                self.optimizer.zero_grad()

                # Forward pass
                output = self.forward(samples, exogenous_features)

                # Compute loss
                loss = criterion(output, target.to(self.device))
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(training_loader)
            print(f"Epoch [{epoch + 1}/{max_epochs}], Training Loss: {avg_train_loss:.4f}")

            # Log the training loss
            writer.add_scalar('Loss/Train', avg_train_loss, epoch + 1)

            # Validation step
            self.eval()  # Set model to evaluation mode
            total_val_loss = 0
            with torch.no_grad():
                for samples, exogenous_features, target in validation_loader:
                    output = self.forward(samples, exogenous_features)
                    loss = criterion(output, target.to(self.device))
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(validation_loader)
            print(f"Epoch [{epoch + 1}/{max_epochs}], Validation Loss: {avg_val_loss:.4f}")

            # Log the validation loss
            writer.add_scalar('Loss/Validation', avg_val_loss, epoch + 1)

            # Early stopping or other stopping criteria could be implemented here

        # After training, evaluate on the test set
        self.eval()
        total_test_loss = 0
        with torch.no_grad():
            for samples, exogenous_features, target in test_loader:                
                output = self.forward(samples, exogenous_features)
                loss = criterion(output, target.to(self.device))
                total_test_loss += loss.item()

        avg_test_loss = total_test_loss / len(test_loader)
        print(f"Test Loss: {avg_test_loss:.4f}")

        # Log the test loss
        writer.add_scalar('Loss/Test', avg_test_loss, max_epochs)

        # Close the writer
        writer.close()

        return avg_train_loss, avg_val_loss, avg_test_loss