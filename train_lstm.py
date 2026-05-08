import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader


# Hiperparametros de la red y setup

SEQ_LENGTH = 5
HIDDEN_SIZE = 64
NUM_LAYERS = 2
EPOCHS = 150
LEARNING_RATE = 0.001
TRAIN_SPLIT_YEAR = 2015  # Train: 1990-2014 | Validate: 2015-2019
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Prepraracion de datos y scaling para streamlit

def load_and_scale_data(filepath='coffee_db.parquet'):
    df = pd.read_parquet(filepath)
    year_cols = [col for col in df.columns if '/' in col]
    
    df_long = df.melt(id_vars=['Country', 'Coffee type'], value_vars=year_cols, 
                      var_name='Year', value_name='Consumption')
    df_long['Year'] = df_long['Year'].str.split('/').str[0].astype(int)
    df_long = df_long.sort_values(by=['Country', 'Coffee type', 'Year']).reset_index(drop=True)
    
    # Global Scaler: Fit on ALL data to ensure uniform magnitude scaling
    scaler = MinMaxScaler()
    df_long['Scaled_Consumption'] = scaler.fit_transform(df_long[['Consumption']])
    joblib.dump(scaler, 'consumption_scaler.gz')
    
    #aqui se entrena hsta 2014
    df_train = df_long[df_long['Year'] < TRAIN_SPLIT_YEAR].copy()
    
    # Validacion de pesos con datos de 2014 a 2019
    df_val = df_long[df_long['Year'] >= (TRAIN_SPLIT_YEAR - SEQ_LENGTH)].copy()
    
    return df_train, df_val, scaler

#Creador de datasets secuenciales para la validacion
class CoffeeSequenceDataset(Dataset):
    def __init__(self, df, seq_length):
        self.X, self.y = [], []
        for _, group in df.groupby(['Country', 'Coffee type']):
            values = group['Scaled_Consumption'].values
            if len(values) > seq_length:
                for i in range(len(values) - seq_length):
                    self.X.append(values[i : i + seq_length])
                    self.y.append(values[i + seq_length])
                    
        self.X = torch.tensor(np.array(self.X), dtype=torch.float32).unsqueeze(-1)
        self.y = torch.tensor(np.array(self.y), dtype=torch.float32).unsqueeze(-1)

    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

#Arquitectura de la LSTM en pytorch con los hiperparametros definidos anteriormente
class GlobalLSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super(GlobalLSTMForecaster, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]) 

#Entrenamiento y validacion
if __name__ == "__main__":
    print(f"Entrenando en dispositivo: {DEVICE}")
    df_train, df_val, _ = load_and_scale_data()
    
    # CataLoaders
    train_dataset = CoffeeSequenceDataset(df_train, SEQ_LENGTH)
    val_dataset = CoffeeSequenceDataset(df_val, SEQ_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False) 
    
    model = GlobalLSTMForecaster(hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        #validacion
        model.eval() 
        total_val_loss = 0
        with torch.no_grad(): 
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                predictions = model(batch_X)
                val_loss = criterion(predictions, batch_y)
                total_val_loss += val_loss.item()
                
        avg_val_loss = total_val_loss / len(val_loader)
        
        #checkpoints del modelo para seleccionar los mejores pesos
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'global_lstm_best.pth')
            saved_indicator = "💾 Modelo guardado!"
        else:
            saved_indicator = ""
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1:03d}/{EPOCHS}] | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f} {saved_indicator}")
            
    print(f"\nTEntrenamiento completado, Mejor perdida: {best_val_loss:.5f}")
    print("Mejores pesos guardados como 'global_lstm_best.pth'")