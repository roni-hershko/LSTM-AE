import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import os.path as path
from os import getcwd
from torch.nn.utils import clip_grad_norm_
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_data():
    # Load the dataset in Google Colab
    df = pd.read_csv(path.join(getcwd(), 'SP_500_Stock_Prices_2014-2017.csv'))
    # Convert the 'date' column to datetime format
    df['date'] = pd.to_datetime(df['date'])
    # Filter data for AMZN and GOOGL
    amzn = df[df['symbol'] == 'AMZN']
    googl = df[df['symbol'] == 'GOOGL']
    aapl = df[df['symbol'] == 'AAPL']

    return amzn, googl, aapl

# Data preprocessing functions
def preprocess_data(data, sequence_length=50):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.values.reshape(-1, 1))

    # Create sequences for input and output
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data_scaled[i:i+sequence_length])
        y.append(data_scaled[i+1:i+sequence_length+1])  # Next step for reconstruction

    # Convert data to PyTorch tensors 
    x_stock= torch.tensor(np.array(X), dtype=torch.float32)
    y_stock= torch.tensor(np.array(y), dtype=torch.float32)
    
    # Create DataLoader
    train_data = TensorDataset(x_stock, y_stock)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    return x_stock, train_loader, scaler

def preprocess_for_prediction(data, sequence_length=50):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.values.reshape(-1, 1))
    # Create sequences for input and output
    X = []
    X.append(data_scaled[0:sequence_length])
    # Convert data to PyTorch tensors 
    x_stock_sample= torch.tensor(np.array(X), dtype=torch.float32)
    return x_stock_sample, scaler


# Define the LSTM 
# reconstruct model
class LSTM_AE(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM_AE, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_enc = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.lstm_dec = nn.LSTM(hidden_size, input_size, batch_first=True)

    def forward(self, x):
        enc_out, (enc_hn, enc_cn) = self.lstm_enc(x)
        dec_output, (dec_hn, dec_cn)= self.lstm_dec(enc_out)
        return dec_output    
# predict model
class LSTM_AE_PRED(nn.Module):    
    def __init__(self, input_size=1, hidden_size=16):
        super(LSTM_AE_PRED, self).__init__()
        self.reconstructor_ae = LSTM_AE(input_size, hidden_size)
        self.prediction_ae = LSTM_AE(input_size, hidden_size)
        
    def forward(self, x):
        dec = self.reconstructor_ae(x)
        prediction = self.prediction_ae(x)
        return dec, prediction


# Training and eval functions    
def train_for_recon(num_epochs, model, train_loader, criterion, optimizer, clip_value, title):
    # Training loop for AMZN
    for epoch in range(num_epochs):
        model.train()
        for data in train_loader:
            X_batch, y_batch = data
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            reconstructed= model(X_batch)
            loss = criterion(reconstructed, X_batch)
            loss.backward()
            loss = clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()
        print(f" Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

def train_and_eval_for_pred(num_epochs, model, train_loader, criterion, optimizer, clip_value, title, training_losses, scaler, x_tensor, recon_losses, prediction_losses, amzn):
    for epoch in range(num_epochs):
        model.train()
        epoch_training_loss = 0
        for data in train_loader:
            X_batch, y_batch = data
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            recon, pred = model(X_batch)
            recon_loss = criterion(recon, X_batch)
            pred_loss = criterion(pred, y_batch)

            loss = recon_loss + pred_loss
            loss.backward()
            loss = clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()

            epoch_training_loss += loss.item()

        epoch_training_loss /= len(train_loader)
        training_losses.append(epoch_training_loss)
        print(f"AMZN Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_training_loss:.4f}")

        model.eval()
        with torch.no_grad():
            reocon_amzn, predicted_amzn = model(x_tensor.to(device))
            reocon_amzn, predicted_amzn = reocon_amzn.cpu().numpy().reshape(reocon_amzn.shape), predicted_amzn.cpu().numpy().reshape(predicted_amzn.shape)
            reocon_amzn, predicted_amzn = scaler.inverse_transform(reocon_amzn.reshape(-1, 1)).reshape(reocon_amzn.shape), scaler.inverse_transform(predicted_amzn.reshape(-1, 1)).reshape(predicted_amzn.shape)
            # Flattening predictions
            recon_amzn_flat, predicted_amzn_flat = reocon_amzn[:, -1, 0] ,predicted_amzn[:, -1, 0]  
            #real values
            actual_amzn_values = scaler.inverse_transform((amzn['high']).values.reshape(-1, 1))
            # Calculate the prediction loss (MSE)
            recon_loss_amzn, prediction_loss_amzn = mean_squared_error(actual_amzn_values[-len(recon_amzn_flat):], recon_amzn_flat), mean_squared_error(actual_amzn_values[-len(predicted_amzn_flat):], predicted_amzn_flat)
            recon_losses.append(recon_loss_amzn)
            prediction_losses.append(prediction_loss_amzn)
    return predicted_amzn_flat, actual_amzn_values[-len(predicted_amzn_flat):]

def eval_for_recon(model, x_tensor, scaler):
    model.eval()
    with torch.no_grad():
        x_tensor = x_tensor.to(device)
        reconstructed = model(x_tensor)
    # Inverse transform to get actual values for AMZN
    reconstructed = reconstructed.cpu().numpy().reshape(reconstructed.shape)
    reconstructed = scaler.inverse_transform(reconstructed.reshape(-1, 1)).reshape(reconstructed.shape)
    # Flatten the predicted values for AMZN
    reconstructed_flat = reconstructed[:, -1, 0]  # Taking the last value of each sequence for plotting
    return reconstructed_flat

def eval_for_pred(num_epochs, model, x_sample_tensor, scaler, amzn, recon_losses, prediction_losses):
    # for epoch in range(num_epochs):
        model.eval()
        predictions = []
        with torch.no_grad():
            reocon_amzn, predicted_amzn = model(x_sample_tensor.to(device))
            predictions.extend(predicted_amzn.squeeze(0).squeeze(1).detach().tolist())
            for i in range(25):
                reocon_amzn, predicted_amzn = model(predicted_amzn)
                predictions.append(predicted_amzn.squeeze(0).squeeze(1).detach().tolist()[-1])
            predictions = np.array(predictions)
            # Flattening predictions
            # predicted_amzn_flat = predictions[:, -1, 0].flatten()             
            # Calculate the prediction loss (MSE)
            predicted_actual_amzn_values = scaler.inverse_transform((predictions).reshape(-1, 1)).tolist()
        
        return predicted_actual_amzn_values
        
        
# Plots
def Plot_daily_max (data, title, label):
    plt.figure(figsize=(10, 5))
    plt.plot(data['date'], data['high'], label=label, color='blue')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Daily Maximum Value')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
        
def plot_reconstruction(data, reconstructed_flat, title):
    plt.figure(figsize=(10, 5))
    plt.plot(data['date'][-len(reconstructed_flat):], data['high'][-len(reconstructed_flat):], label='Actual Prices', color='blue')
    plt.plot(data['date'][-len(reconstructed_flat):], reconstructed_flat, label='Predicted Prices', color='red')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_loss(num_epochs, training_losses_amzn, prediction_losses_amzn, recon_losses_amzn):
    # Plot Training Loss vs. Time (Epoch)
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs), training_losses_amzn, label='Training Loss AMZN', color='blue')
    plt.title('AMZN Training Loss vs. Time')
    plt.xlabel('time')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plot Prediction Loss vs. Time (Epoch)
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs), prediction_losses_amzn, label='Prediction Loss AMZN', color='red')
    plt.title('AMZN Prediction Loss (Prediction vs Actual) vs. Time')
    plt.xlabel('time')
    plt.ylabel('Prediction Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    # Plot reconstructed Loss vs. Time (Epoch)
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_epochs), recon_losses_amzn, label='reconstracted Loss AMZN', color='red')
    plt.title('AMZN reconstructed Loss (Prediction vs Actual) vs. Time')
    plt.xlabel('time')
    plt.ylabel('recon Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_prediction(pred, gt, title1):
    plt.figure(figsize=(10, 5))
    plt.plot([i for i in range (len(gt))],gt, label='Actual Prices', color='blue')
    plt.plot([i for i in range (len(gt))],pred, label='Predicted Prices', color='red')
    plt.title(title1)
    plt.xlabel('time')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


'''questions'''      
def q_1_daily_maximum(amzn, googl):
    # Plot the daily maximum values for AMZN
    Plot_daily_max(amzn, 'AMZN Daily Maximum Stock Prices','AMZN Daily Maximum')
    # Plot the daily maximum values for GOOGL
    Plot_daily_max(googl, 'GOOGL Daily Maximum Stock Prices','GOOGL Daily Maximum')

def q_2_reconstruct(amzn, googl, aapl):
    input_size = 1
    hidden_size = 64
    batch_size = 128
    num_epochs = 100
    lr = 1e-4
    clip_value = 1.0
    sequence_length = 50
    criterion= nn.MSELoss()

    # Preprocessing
    X_amzn_tensor, train_loader_amzn, scaler_amzn= preprocess_data(amzn['high'])
    X_googl_tensor, train_loader_googl, scaler_googl= preprocess_data(googl['high'])
    x_aapl_tensor, train_loader_aapl, scaler_aapl= preprocess_data(aapl['high'])
    
    # Initialize the model for both AMZN and GOOGL (same model)
    model_amzn = LSTM_AE(input_size, hidden_size).to(device)
    model_googl = LSTM_AE(input_size, hidden_size).to(device)
    model_aapl = LSTM_AE(input_size, hidden_size).to(device)
    
    # Optimizer for both models (same optimizer)
    optimizer_amzn = torch.optim.Adam(model_amzn.parameters(), lr=0.001)
    optimizer_googl = torch.optim.Adam(model_googl.parameters(), lr=0.001)
    optimizer_aapl = torch.optim.Adam(model_aapl.parameters(), lr=0.001)
    
    # Training loop
    train_for_recon(num_epochs, model_amzn, train_loader_amzn, criterion, optimizer_amzn, clip_value, 'AMZN')
    train_for_recon(num_epochs, model_googl, train_loader_googl, criterion, optimizer_googl, clip_value, 'GOOGL')
    train_for_recon(num_epochs, model_aapl, train_loader_aapl, criterion, optimizer_aapl, clip_value, 'AAPL')
    
    # Reconstruct and predict
    reconstructed_flat_amzn= eval_for_recon(model_amzn, X_amzn_tensor, scaler_amzn)
    reconstructed_flat_googl= eval_for_recon(model_googl, X_googl_tensor, scaler_googl)
    reconstructed_flat_aapl= eval_for_recon(model_aapl, x_aapl_tensor, scaler_aapl)
    
    # Plot the predicted values
    plot_reconstruction(amzn, reconstructed_flat_amzn, 'AMZN Stock Price Reconstruction')
    plot_reconstruction(googl, reconstructed_flat_googl, 'GOOGL Stock Price Reconstruction')
    plot_reconstruction(aapl, reconstructed_flat_aapl, 'AAPL Stock Price Reconstruction')
 
def q_3_predict(amzn):
    input_size = 1
    hidden_size = 64
    batch_size = 128
    num_epochs = 100
    lr = 1e-4
    clip_value = 1.0
    sequence_length = 50
    num_epochs = 50
    criterion= nn.MSELoss()
    
    # Lists to store losses
    training_losses_amzn = []
    recon_losses_amzn = []
    prediction_losses_amzn = []
    
    #Preprocessing
    X_amzn_tensor, train_loader_amzn, scaler_amzn= preprocess_data(amzn['high'])
    # Initialize the model for both AMZN and GOOGL (same model)
    model_amzn = LSTM_AE_PRED(input_size, hidden_size).to(device)
    # Optimizer for both models (same optimizer)
    optimizer_amzn = torch.optim.Adam(model_amzn.parameters(), lr=0.001)   
    # Training loop and evaluation
    predicted_flat, actual_amzn_values = train_and_eval_for_pred(num_epochs, model_amzn, train_loader_amzn, criterion, optimizer_amzn, clip_value, 'AMZN', training_losses_amzn, scaler_amzn, X_amzn_tensor, recon_losses_amzn, prediction_losses_amzn, amzn)  
    
    predicted_flat = scaler_amzn.inverse_transform(predicted_flat.reshape(-1, 1)).flatten()
    actual_amzn_values = actual_amzn_values.flatten()  # Ensure actual values are also flattened if necessary

    # Ensure both predicted and actual values have the same length
    predicted_flat = predicted_flat[:len(actual_amzn_values)]  # Truncate if necessary
    actual_amzn_values = actual_amzn_values[-len(predicted_flat):]  # Trunca
    # Plot the predicted values
    plot_loss(num_epochs, training_losses_amzn, prediction_losses_amzn, recon_losses_amzn)
    plot_prediction(predicted_flat, actual_amzn_values, 'AMZN Stock Price Prediction')
    return model_amzn

def q_4_predict_multistep(amzn, model_amzn):
    input_size = 1
    hidden_size = 64
    batch_size = 128
    num_epochs = 100
    lr = 1e-4
    clip_value = 1.0
    sequence_length = 50
    num_epochs = 50
    criterion= nn.MSELoss()
    
    # Lists to store losses
    recon_losses_amzn = []
    prediction_losses_amzn = []
    #Preprocessing
    X_amzn_sample_50, scaler_amzn= preprocess_for_prediction(amzn['high'])
    X_amzn_tensor_25 = X_amzn_sample_50[:,:25,:]
    # X_amzn_sample_50_actual = scaler_amzn.inverse_transform(X_amzn_sample_50.numpy().reshape(-1, 1)).reshape(X_amzn_sample_50.shape)[1:]
    X_amzn_sample_50_actual = scaler_amzn.inverse_transform(X_amzn_sample_50.numpy().reshape(-1, 1))

    # Initialize the model for both AMZN and GOOGL (same model)
    # need to train model, fixxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
    # model_amzn = LSTM_AE_PRED(input_size, hidden_size, sequence_length=25).to(device)
    # evaluate the model
    predicted_amzn_flat= eval_for_pred(num_epochs, model_amzn, X_amzn_tensor_25, scaler_amzn, amzn, recon_losses_amzn, prediction_losses_amzn)
  #plot_prediction(predicted_amzn_flat, X_amzn_sample_50_actual.squeeze(0).squeeze(1).tolist(), #X_amzn_sample_50_actual.shape= (0, 50, 1),
            #        'AMZN Stock Price pred', 'AMZN Stock Price prediction')
    # plot_reconstruction(amzn, 'AMZN Stock Price Prediction')
    plot_prediction(predicted_amzn_flat, X_amzn_sample_50_actual, 'AMZN Stock Price prediction')



def main():
    amzn, googl, aapl= load_data()
    # q_1_daily_maximum(amzn, googl)
    # q_2_reconstruct(amzn, googl, aapl)
    model_amzn = q_3_predict(amzn)
    q_4_predict_multistep(amzn, model_amzn)
    
    
if __name__ == '__main__':
    main()

