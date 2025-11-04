#!/usr/bin/env python3

import sys
import math
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ------------------ CONFIG ------------------
CSV_FILE = "multi_ap_rssi_log.csv"
AP_POS_JSON = "ap_positions.json"
OUT_HTML_TEMPLATE = "heatmap_mode_{}.html"

DEFAULT_MODE = "median"
GRID_STEP = 0.1       # grid resolution in meters

# ------------------ LOAD AP POSITIONS ------------------
try:
    with open(AP_POS_JSON, "r") as f:
        KNOWN_AP_POSITIONS = json.load(f)
        # normalize keys to lowercase
        KNOWN_AP_POSITIONS = {k.lower(): tuple(v) for k, v in KNOWN_AP_POSITIONS.items()}
    print(f"[i] Loaded {len(KNOWN_AP_POSITIONS)} AP positions from JSON.")
except FileNotFoundError:
    print("[!] AP positions file not found.")
    sys.exit(1)

# ------------------ HELPER FUNCTIONS ------------------

def load_and_prepare(csv_file):
    df = pd.read_csv(csv_file)
    df = df.dropna(subset=["bssid", "distance", "dbm", "ssid"]).copy()
    df["bssid_label"] = df["bssid"].str.replace(r"[()\s]+", "", regex=True).str.lower()
    df = df[df["bssid_label"].isin(KNOWN_AP_POSITIONS.keys())]
    if df.empty:
        print("[!] No valid AP data after filtering.")
    return df

# ------------------ POSITION ESTIMATION ------------------
def estimate_position(ap_positions, distances):
    if len(distances) < 3:
        return None

    bs, ds = [], []
    for bssid, (x, y) in ap_positions.items():
        if bssid in distances:
            bs.append((x, y))
            ds.append(distances[bssid])
    if len(bs) < 3:
        return None

    x1, y1 = bs[0]
    d1 = ds[0]
    A, b = [], []
    for i in range(1, len(bs)):
        x2, y2 = bs[i]
        d2 = ds[i]
        A.append([2*(x2-x1), 2*(y2-y1)])
        b.append(d1**2 - d2**2 + x2**2 - x1**2 + y2**2 - y1**2)

    try:
        sol, *_ = np.linalg.lstsq(np.array(A), np.array(b), rcond=None)
        return (float(sol[0]), float(sol[1]))
    except:
        return None

# ------------------ HEATMAP ------------------
def compute_heatmap(df, ap_positions, est_pos=None):
    # Compute median RSSI per AP
    median_dbm = df.groupby('bssid_label')['dbm'].median().to_dict()
    # Map BSSID -> SSID
    bssid_to_ssid = df.groupby('bssid_label')['ssid'].first().to_dict()

    # Compute grid bounds
    xs = [x for x, y in ap_positions.values()]
    ys = [y for x, y in ap_positions.values()]
    if est_pos is not None:
        xs.append(est_pos[0])
        ys.append(est_pos[1])

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    x_margin = max(1.0, (x_max - x_min) * 0.3)
    y_margin = max(1.0, (y_max - y_min) * 0.3)
    x_min, x_max = x_min - x_margin, x_max + x_margin
    y_min, y_max = y_min - y_margin, y_max + y_margin

    xi_vals = np.arange(x_min, x_max + GRID_STEP, GRID_STEP)
    yi_vals = np.arange(y_min, y_max + GRID_STEP, GRID_STEP)
    xi, yi = np.meshgrid(xi_vals, yi_vals)
    zi = np.zeros_like(xi, dtype=float)

    # Automatic sigma based on average AP spacing
    positions = np.array(list(ap_positions.values()))
    if len(positions) > 1:
        dists = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
        sigma = np.mean(dists[dists > 0]) / 2
    else:
        sigma = 1.0
    sigma = max(sigma, 0.5)

    # Compute weighted contributions from each AP
    for bssid, (x_ap, y_ap) in ap_positions.items():
        dbm = float(median_dbm.get(bssid, -90))
        dist_grid = np.hypot(xi - x_ap, yi - y_ap)
        zi += dbm * np.exp(-0.5 * (dist_grid / sigma) ** 2)

    # Normalize heatmap to -100..-30
    zi_min, zi_max = zi.min(), zi.max()
    if zi_max > zi_min:
        zi = (zi - zi_min) / (zi_max - zi_min)
        zi = zi * 70 - 100
    else:
        zi[:] = -80

    return xi, yi, zi, bssid_to_ssid

# ------------------ PLOT ------------------
def plot_and_save(xi, yi, zi, ap_positions, bssid_to_ssid, estimated_pos=None):
    fig = go.Figure(
        data=go.Heatmap(
            x=xi[0],
            y=yi[:, 0],
            z=zi,
            colorscale='Viridis',
            colorbar=dict(title='RSSI (dBm)')
        )
    )

    for bssid, (x, y) in ap_positions.items():
        ssid = bssid_to_ssid.get(bssid, bssid)
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            text=[ssid],
            textposition='top center',
            marker=dict(color='red', size=10, symbol='x'),
            name=f'AP: {ssid}'
        ))

    if estimated_pos is not None:
        fig.add_trace(go.Scatter(
            x=[estimated_pos[0]], y=[estimated_pos[1]],
            mode='markers+text',
            text=['Device'],
            textposition='bottom center',
            marker=dict(color='orange', size=12, symbol='circle'),
            name='Device'
        ))

    fig.update_layout(
        title='Wi-Fi RSSI Heatmap',
        xaxis_title='X (m)',
        yaxis_title='Y (m)',
        yaxis=dict(scaleanchor='x', scaleratio=1),
        legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5)
    )

    outname = OUT_HTML_TEMPLATE.format("ssid")
    fig.write_html(outname)
    print(f"[+] Saved heatmap to {outname}")
    fig.show()

# ------------------ MAIN ------------------
def main(argv):
    df = load_and_prepare(CSV_FILE)
    if df.empty:
        print("[!] No data to process.")
        return

    distances = df.groupby('bssid_label')['distance'].median().to_dict()
    est_pos = estimate_position(KNOWN_AP_POSITIONS, distances)
    if est_pos:
        print(f"[i] Estimated device position: {est_pos}")
    else:
        print("[i] Not enough APs for trilateration")

    xi, yi, zi, bssid_to_ssid = compute_heatmap(df, KNOWN_AP_POSITIONS, est_pos)
    plot_and_save(xi, yi, zi, KNOWN_AP_POSITIONS, bssid_to_ssid, estimated_pos=est_pos)

if __name__ == '__main__':
    main(sys.argv)
