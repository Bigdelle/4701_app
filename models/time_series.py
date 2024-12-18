import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from typing import List, Optional, Tuple, Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import yfinance as yf
import pickle
import os


class StockDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """
        Args:
            X: Input sequences of shape (n_samples, look_back, n_features)
            y: Target values of shape (n_samples,)
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int = 1,
                 dense_layers: List[int] = [128, 64], dropout: float = 0.4):
        """
        Initialize LSTM model with dense layers.

        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
            dense_layers: List of units in dense layers after LSTM
            dropout: Dropout rate for dense layers
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True)

        # Build dense layers
        layers = []
        # Calculate input dimension for first dense layer:
        # hidden_size * num_directions(1) * num_layers
        lstm_output_size = hidden_size * num_layers
        input_dim = lstm_output_size

        for units in dense_layers:
            layers.extend([
                nn.Linear(input_dim, units),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            input_dim = units

        # Final output layer
        layers.append(nn.Linear(input_dim, 1))

        self.dense_stack = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # Get batch size
        batch_size = x.size(0)

        # Process through LSTM
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        # hn shape: (num_layers, batch_size, hidden_size)
        _, (hn, _) = self.lstm(x)

        # Reshape hidden states:
        # First transpose to (batch_size, num_layers, hidden_size)
        # Then reshape to (batch_size, num_layers * hidden_size)
        hn = hn.transpose(0, 1).contiguous()
        hn = hn.reshape(batch_size, self.num_layers * self.hidden_size)

        # Process through dense layers
        out = self.dense_stack(hn)
        return out


class StockPredictor:
    """A general-purpose time series prediction model using LSTM."""

    def __init__(self, n_features: int):
        """
        Initialize StockPredictor.

        Args:
            n_features: Number of input features
        """

        self.n_features = n_features
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Initialize empty attributes
        self.model: Optional[LSTMModel] = None
        self.scaler: Optional[MinMaxScaler] = None
        self.y_scaler: Optional[MinMaxScaler] = None
        self.normalize: bool = False
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        self.X_val: Optional[np.ndarray] = None
        self.y_val: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

    def setup_model(self, hidden_size: int = 64, num_layers: int = 3, normalize: bool = False) -> None:
        """
        Set up the LSTM model with specified parameters.

        Args:
            hidden_size: Number of hidden units in LSTM
            num_layers: Number of LSTM layers
            normalize: Whether to normalize input and target data
        """
        self.model = LSTMModel(
            input_size=self.n_features,
            hidden_size=hidden_size,
            num_layers=num_layers
        ).to(self.device)

        self.normalize = normalize
        if normalize:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
            self.y_scaler = MinMaxScaler(feature_range=(0, 1))

    @staticmethod
    def validate_sequence_data(X: np.ndarray, y: np.ndarray) -> None:
        """
        Validate the shape and content of sequence data.

        Args:
            X: Input sequences of shape (n_samples, look_back, n_features)
            y: Target values of shape (n_samples,)

        Raises:
            ValueError: If data doesn't match expected shape or contains invalid values
        """
        if len(X.shape) != 3:
            raise ValueError(
                f"Expected X to have 3 dimensions (n_samples, look_back, n_features), got shape {X.shape}")

        if len(y.shape) != 1:
            raise ValueError(
                f"Expected y to have 1 dimension (n_samples,), got shape {y.shape}")

        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"Number of samples in X ({X.shape[0]}) and y ({y.shape[0]}) must match")

        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError("Data contains NaN values")

        if np.isinf(X).any() or np.isinf(y).any():
            raise ValueError("Data contains infinite values")

    def normalize_data(self, X: np.ndarray, y: np.ndarray, fit: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize input and target data.

        Args:
            X: Input sequences of shape (n_samples, look_back, n_features)
            y: Target values of shape (n_samples,)
            fit: Whether to fit the scalers on this data

        Returns:
            Tuple of normalized X and y arrays
        """
        if not self.normalize:
            return X, y

        # Reshape X to 2D for scaling
        n_samples = X.shape[0] * X.shape[1]
        X_flat = X.reshape(n_samples, X.shape[2])

        if fit:
            X_norm = self.scaler.fit_transform(X_flat)
            y_norm = self.y_scaler.fit_transform(y.reshape(-1, 1))
        else:
            X_norm = self.scaler.transform(X_flat)
            y_norm = self.y_scaler.transform(y.reshape(-1, 1))

        # Reshape X back to 3D
        X_norm = X_norm.reshape(X.shape)
        y_norm = y_norm.ravel()

        return X_norm, y_norm

    def load_training_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Load training data.

        Args:
            X: Training sequences of shape (n_samples, look_back, n_features)
            y: Target values of shape (n_samples,)
        """
        self.validate_sequence_data(X, y)

        if X.shape[2] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {X.shape[2]}")

        # Normalize data if enabled
        X_processed, y_processed = self.normalize_data(X, y, fit=True)

        self.X_train = X_processed
        self.y_train = y_processed

    def load_validation_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Load validation data.

        Args:
            X: Validation sequences of shape (n_samples, look_back, n_features)
            y: Target values of shape (n_samples,)
        """
        self.validate_sequence_data(X, y)

        if X.shape[2] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {X.shape[2]}")

        # Normalize data if enabled
        X_processed, y_processed = self.normalize_data(X, y, fit=False)

        self.X_val = X_processed
        self.y_val = y_processed

    def load_test_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Load test data.

        Args:
            X: Test sequences of shape (n_samples, look_back, n_features)
            y: Target values of shape (n_samples,)
        """
        self.validate_sequence_data(X, y)

        if X.shape[2] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {X.shape[2]}")

        # Normalize data if enabled
        X_processed, y_processed = self.normalize_data(X, y, fit=False)

        self.X_test = X_processed
        self.y_test = y_processed

    def plot_training_history(self, history: Dict[str, List[float]], early_stopping_epoch: Optional[int] = None) -> None:
        """
        Plot training and validation loss history.

        Args:
            history: Dictionary containing training and validation losses
            early_stopping_epoch: Epoch where early stopping occurred (if applicable)
        """
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(history['train_loss']) + 1)

        plt.plot(epochs, history['train_loss'], 'b-',
                 label='Training Loss', linewidth=2)
        plt.plot(epochs, history['val_loss'], 'r-',
                 label='Validation Loss', linewidth=2)

        if early_stopping_epoch:
            plt.axvline(x=early_stopping_epoch, color='g', linestyle='--',
                        label=f'Early Stopping (epoch {early_stopping_epoch})')

        plt.title('Training and Validation Loss Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add min loss markers
        min_val_loss = min(history['val_loss'])
        min_val_epoch = history['val_loss'].index(min_val_loss) + 1
        plt.plot(min_val_epoch, min_val_loss, 'rv', label='Best Model')

        plt.tight_layout()
        plt.show()

    def train(self,
              batch_size: int = 16,
              num_epochs: int = 10,
              learning_rate: float = 0.001,
              early_stopping: bool = True,
              patience: int = 5,
              min_delta: float = 1e-4,
              plot_losses: bool = True) -> Dict[str, List[float]]:
        """
        Train the model and return training history.

        Args:
            batch_size: Number of samples per batch
            num_epochs: Maximum number of training epochs
            learning_rate: Learning rate for optimizer
            early_stopping: Whether to use early stopping
            patience: Number of epochs to wait for improvement before stopping
            min_delta: Minimum change in validation loss to qualify as an improvement
            plot_losses: Whether to plot training and validation losses

        Returns:
            Dictionary containing training history
        """
        if self.model is None:
            raise ValueError("Model not set up. Call setup_model() first.")
        if self.X_train is None or self.y_train is None:
            raise ValueError(
                "Training data not loaded. Call load_training_data() first.")
        if self.X_val is None or self.y_val is None:
            raise ValueError(
                "Validation data not loaded. Call load_validation_data() first.")

        history = {
            'train_loss': [],
            'val_loss': []
        }

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        train_dataset = StockDataset(self.X_train, self.y_train)
        val_dataset = StockDataset(self.X_val, self.y_val)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Early stopping variables
        best_val_loss = float('inf')
        best_model_state = None
        patience_counter = 0
        early_stopping_epoch = None

        for epoch in tqdm(range(num_epochs)):
            # Training phase
            self.model.train()
            train_losses = []
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(
                    self.device), batch_y.to(self.device)

                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            # Validation phase
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(
                        self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X)
                    val_loss = criterion(outputs.squeeze(), batch_y)
                    val_losses.append(val_loss.item())

            avg_train_loss = np.mean(train_losses)
            avg_val_loss = np.mean(val_losses)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)

            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {avg_train_loss:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}')

            # Early stopping check
            if early_stopping:
                if avg_val_loss < best_val_loss - min_delta:
                    best_val_loss = avg_val_loss
                    best_model_state = self.model.state_dict().copy()
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(
                            f'\nEarly stopping triggered after {epoch + 1} epochs')
                        early_stopping_epoch = epoch + 1
                        # Restore best model
                        self.model.load_state_dict(best_model_state)
                        break

        # Plot training history if requested
        if plot_losses:
            self.plot_training_history(history, early_stopping_epoch)

        return history

    def predict(self) -> np.ndarray:
        """
        Make predictions on the loaded test data.

        Returns:
            Predicted values of shape (n_samples,)
        """
        X = self.X_test
        if self.model is None:
            raise ValueError("Model not set up. Call setup_model() first.")

        self.validate_sequence_data(
            X, np.zeros(X.shape[0]))  # Validate X shape

        # Normalize input data if enabled
        # X_processed, _ = self.normalize_data(X, np.zeros(X.shape[0]), fit=False)
        X_processed = X

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_processed).to(self.device)
            predictions = self.model(X_tensor)
            predictions = predictions.cpu().numpy()

        # Inverse transform predictions if normalization is enabled
        if self.normalize:
            predictions = self.y_scaler.inverse_transform(predictions)

        return predictions.ravel()

    def evaluate_test_set(self, display_plots: bool = True) -> Dict[str, float]:
        """
        Evaluate the model on the test set and display results.

        Args:
            display_plots: Whether to display visualization plots

        Returns:
            Dictionary containing test metrics
        """
        if self.model is None:
            raise ValueError("Model not set up. Call setup_model() first.")
        if self.X_test is None or self.y_test is None:
            raise ValueError(
                "Test data not loaded. Call load_test_data() first.")

        # Get predictions
        y_pred = self.predict()
        y_true = self.y_test
        if self.normalize:
            y_true = self.y_scaler.inverse_transform(
                y_true.reshape(-1, 1)).ravel()

        print(y_pred)
        # Calculate metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        # Calculate percentage errors
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

        # Direction accuracy
        direction_correct = np.sum(np.sign(y_true[1:] - y_true[:-1]) ==
                                   np.sign(y_pred[1:] - y_pred[:-1]))
        direction_accuracy = direction_correct / (len(y_true) - 1) * 100

        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'direction_accuracy': direction_accuracy
        }

        # Print metrics
        print("\nTest Set Evaluation Metrics:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"MAPE: {mape:.2f}%")
        print(f"Direction Accuracy: {direction_accuracy:.2f}%")

        if display_plots:
            # Create figure with subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

            # Prediction vs Actual plot
            ax1.plot(y_true, label='Actual', color='blue', alpha=0.7)
            ax1.plot(y_pred, label='Predicted', color='red', alpha=0.7)
            ax1.set_title('Actual vs Predicted Values')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Value')
            ax1.legend()
            ax1.grid(True)

            # Scatter plot
            ax2.scatter(y_true, y_pred, alpha=0.5)
            ax2.plot([y_true.min(), y_true.max()],
                     [y_true.min(), y_true.max()],
                     'r--', lw=2)
            ax2.set_title('Prediction vs Actual Scatter Plot')
            ax2.set_xlabel('Actual Values')
            ax2.set_ylabel('Predicted Values')
            ax2.grid(True)

            plt.tight_layout()
            plt.show()

            # Residual plot
            residuals = y_true - y_pred
            plt.figure(figsize=(12, 5))
            plt.scatter(y_pred, residuals, alpha=0.5)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.title('Residual Plot')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.grid(True)
            plt.show()

            # Error distribution
            plt.figure(figsize=(10, 5))
            plt.hist(residuals, bins=50, alpha=0.75)
            plt.title('Error Distribution')
            plt.xlabel('Prediction Error')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.show()

        return metrics

    def save_model(self, model_path: str, scaler_path: str = None) -> None:
        """
        Save the model and scalers to separate files.

        Args:
            model_path: Path to save the model (e.g., 'model.pt')
            scaler_path: Path to save the scalers (e.g., 'scalers.pkl'). 
                        If None, will use model_path with '_scalers.pkl' suffix
        """
        if self.model is None:
            raise ValueError("Model not initialized. Nothing to save.")

        # Save model to CPU before saving
        self.model = self.model.to('cpu')

        # Save just the state dict
        torch.save(self.model.state_dict(), model_path)

        # Move model back to original device
        self.model = self.model.to(self.device)

        # Save scalers if normalization is enabled
        if self.normalize and self.scaler is not None and self.y_scaler is not None:
            if scaler_path is None:
                scaler_path = model_path.rsplit('.', 1)[0] + '_scalers.pkl'

            import pickle
            scalers = {
                'feature_scaler': self.scaler,
                'target_scaler': self.y_scaler
            }
            with open(scaler_path, 'wb') as f:
                pickle.dump(scalers, f)

    def load_model(self, model_path: str, scaler_path: str = None) -> None:
        """
        Load model and scalers from files.

        Args:
            model_path: Path to the saved model file
            scaler_path: Path to the saved scalers file. 
                        If None, will look for model_path with '_scalers.pkl' suffix
        """
        # Make sure model is initialized before loading weights
        if self.model is None:
            raise ValueError("Model not set up. Call setup_model() first.")

        # Load the state dict
        state_dict = torch.load(model_path, weights_only=False)
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Try to load scalers if path provided or if default path exists
        if scaler_path is None:
            scaler_path = model_path.rsplit('.', 1)[0] + '_scalers.pkl'

        if os.path.exists(scaler_path):
            import pickle
            with open(scaler_path, 'rb') as f:
                scalers = pickle.load(f)
                self.scaler = scalers['feature_scaler']
                self.y_scaler = scalers['target_scaler']
                self.normalize = True
        else:
            print("No scalers found. Model loaded without normalization settings.")


def fetch_stock_data(ticker: str, start_date: str, end_date: str, interval: str = '1d'):
    """
    Fetch historical stock data from Yahoo Finance.

    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Start date in 'YYYY-MM-DD' format
        end_date (str): End date in 'YYYY-MM-DD' format
        interval (str): Data frequency ('1d' for daily, '1h' for hourly)

    Returns:
        pd.DataFrame: Stock data
    """
    stock = yf.Ticker(ticker)
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval)
    return df


def time_series_split(X: np.ndarray, y: np.ndarray, validation_size: int = 0.2, test_size: int = 0.2):
    """
    Split time series data maintaining temporal order.

    Args:
        X: Input sequences of shape (n_samples, look_back, n_features)
        y: Target values
        test_size: Proportion of data to use for testing (default: 0.2)

    Returns:
        X_train, X_test, y_train, y_test
    """
    # Calculate split index
    split_idx = int(len(X) * (1 - test_size))
    split_val_idx = int(len(X) * (1 - test_size - validation_size))

    # Split maintaining temporal order
    X_train = X[:split_idx]
    X_val = X[split_val_idx:split_idx]
    X_test = X[split_idx:]
    y_train = y[:split_idx]
    y_val = y[split_val_idx:split_idx]
    y_test = y[split_idx:]

    return X_train, X_val, X_test, y_train, y_val,  y_test


def prepare_sequence_data(data: np.ndarray,
                          look_back: int,
                          look_forward: int,
                          target_idx: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequence data from raw time series data.

    Args:
        data: Raw time series data of shape (n_timestamps, n_features)
        look_back: Number of time steps to look back
        look_forward: Number of time steps to predict ahead
        target_idx: Index of the target feature

    Returns:
        Tuple of (X, y) where:
            X: Sequences of shape (n_samples, look_back, n_features)
            y: Target values of shape (n_samples,)
    """
    if len(data.shape) != 2:
        raise ValueError(f"Expected 2D input array, got shape {data.shape}")

    if target_idx >= data.shape[1]:
        raise ValueError(
            f"target_idx {target_idx} must be less than number of features {data.shape[1]}")

    X, y = [], []
    # Hope this +1 I just added doesn't break anything
    for i in range(len(data) - look_back - look_forward + 1):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back + look_forward - 1, target_idx])

    return np.array(X), np.array(y)