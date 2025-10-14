"""
anomaly_detection.py
Transformer-based anomaly detection for vehicle telemetry.
"""

import numpy as np
import torch
import torch.nn as nn
import pandas as pd


class TransformerAutoencoder(nn.Module):
    def __init__(self, n_features, d_model=64, nhead=8, num_layers=3):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.fc_in = nn.Linear(n_features, d_model)
        self.fc_out = nn.Linear(d_model, n_features)

    def forward(self, x):
        x = self.fc_in(x)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded, encoded)
        out = self.fc_out(decoded)
        return out


def detect_anomalies_transformer(df: pd.DataFrame, feature_cols, threshold_factor=2.0, epochs=25):
    """
    Train a Transformer Autoencoder to detect anomalies across given features.

    Adds:
        - anomaly_score: reconstruction error
        - is_anomaly: binary flag (1 if anomaly, else 0)
    """
    df = df.copy()
    df_feat = df[feature_cols].fillna(0).astype(np.float32)

    # Normalize features internally for stable training
    means = df_feat.mean(axis=0)
    stds = df_feat.std(axis=0).replace(0, 1e-6)
    df_norm = (df_feat - means) / stds

    # Convert to tensor
    X = torch.tensor(df_norm.values, dtype=torch.float32).unsqueeze(0)  # (batch=1, seq_len, features)
    model = TransformerAutoencoder(n_features=len(feature_cols))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    # ---------------- Train ----------------
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        recon = model(X)
        loss = criterion(recon, X)
        loss.backward()
        optimizer.step()

    # ---------------- Evaluate ----------------
    model.eval()
    with torch.no_grad():
        recon = model(X)
        errors = torch.mean((recon - X) ** 2, dim=2).squeeze().numpy()

    # Smooth errors
    smooth_errors = pd.Series(errors).rolling(window=2, min_periods=1, center=True).mean().values
    df["anomaly_score"] = smooth_errors

    # ---------------- Per-Vehicle Thresholding ----------------
    df["is_anomaly"] = 0
    if "vehicle_id" in df.columns:
        for veh, g in df.groupby("vehicle_id"):
            sub_err = df.loc[g.index, "anomaly_score"].values
            thr = np.mean(sub_err) + threshold_factor * np.std(sub_err)
            df.loc[g.index, "is_anomaly"] = (sub_err > thr).astype(int)
    else:
        # Global threshold (fallback)
        threshold = np.mean(smooth_errors) + threshold_factor * np.std(smooth_errors)
        df["is_anomaly"] = (df["anomaly_score"] > threshold).astype(int)

    print("Anomaly detection completed using Transformer Autoencoder.")
    return df
