import pandas as pd
import numpy as np
import glob, os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import torch    

# Scalers
scaler_x = StandardScaler()
scaler_y = StandardScaler()
device = "cuda" if torch.cuda.is_available() else "cpu"
def prepare_kpi_data(base_dir, file_slice=(30, 40), seq_len=50):
    # --- Load CSVs ---
    csv_files = glob.glob(os.path.join(base_dir, "group_*.csv"))
    csv_files = csv_files[file_slice[0]:file_slice[1]]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {base_dir} with pattern 'group_*.csv'")
    
    df = pd.concat([pd.read_csv(f, sep=",", low_memory=False) for f in csv_files], ignore_index=True)
    df.columns = df.columns.str.strip()
    df["Time"] = pd.to_datetime(df["Time"], errors="coerce")

    # --- Features & target ---
    target_col = "L.Thrp.bits.UL(bit)"
    features_cols = [
        "L.Thrp.bits.DL(bit)",
        "L.E-RAB.AttEst",
        "L.E-RAB.SuccEst",
        "L.S1Sig.ConnEst.Att",
        "L.S1Sig.ConnEst.Succ",
        target_col
    ]

    # Convert numeric
    for col in df.columns:
        if col not in ['Time','eNodeB Name','Cell Name','eNodeB Function Name','Cell FDD TDD Indication']:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Encode Cell Names
    cell_names = df["Cell Name"].unique().tolist()
    cell_to_id = {name: idx for idx, name in enumerate(cell_names)}

    # Encode Time
    unique_times = sorted(df["Time"].dropna().unique())
    time_to_id = {t: i for i, t in enumerate(unique_times)}

    # Fit scalers on full dataset
    df_clean = df.dropna(subset=features_cols)
    scaler_x.fit(df_clean[features_cols].values)
    scaler_y.fit(df_clean[[target_col]].values)

    # Build sequences
    all_seq = []
    for cell in cell_names[:100]:
        cell_df = df[df["Cell Name"] == cell]
        cell_df = cell_df[["Time"] + features_cols].dropna()

        features = scaler_x.transform(cell_df[features_cols].values)
        target = scaler_y.transform(cell_df[[target_col]]).flatten()
        dates = pd.to_datetime(cell_df["Time"].values)

        for i in range(len(features) - seq_len):
            x = features[i:i+seq_len]
            y = target[i+seq_len]
            time_ids = [time_to_id[date] for date in dates[i:i+seq_len]]
            cell_id = cell_to_id[cell]
            time_ids = np.array(time_ids).reshape(-1, 1)
            x_with_time = np.concatenate([x, time_ids], axis=1)
            all_seq.append((x_with_time, cell_id, y))

    return all_seq
from modules import KPIGPTDataset 
from torch.utils.data import DataLoader
 
def predict_future_kpi(model,all_seq):
    
    dataset = KPIGPTDataset(all_seq)
    loader = DataLoader(dataset,batch_size=20, shuffle=False, drop_last=True)
    model.eval()
    all_preds, all_true = [], []

    with torch.no_grad():
        for x_feat,  cell_id, y in loader:
            x_feat  = x_feat.to(device)
            cell_id = cell_id.to(device)
            y = y.to(device)

            y_hat = model(x_feat, cell_id)  # Shape [batch_size, 1] or [batch_size]
            all_preds.append(y_hat.cpu().numpy())
            all_true.append(y.cpu().numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds).reshape(-1, 1)
    all_true  = np.concatenate(all_true).reshape(-1, 1)

    # Inverse transform (to original CDR values)
    all_preds = scaler_y.inverse_transform(all_preds)
    all_true  = scaler_y.inverse_transform(all_true) 
    
    return all_preds, all_true


