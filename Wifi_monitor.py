#!/usr/bin/env python3
import subprocess
import time
import threading
import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import numpy as np
from tkinter import Tk, Label, Button, Listbox, MULTIPLE, END, simpledialog, messagebox
import json

# ---------- Config ----------
DEFAULT_IFACE = "wlp2s0"
CSV_OUTPUT = "multi_ap_rssi_log.csv"
AP_POS_JSON = "ap_positions.json"

# ---------- Helpers ----------
def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, shell=False).stdout.strip()
    except Exception as e:
        print(f"[run] command failed {cmd}: {e}")
        return ""

def rssi_to_distance(rssi_dbm, tx_power=-50.0, path_loss_exp=2.5):
    return 10.0 ** ((tx_power - rssi_dbm) / (10.0 * path_loss_exp))

def scan_aps(iface=DEFAULT_IFACE):
    """Scan Wi-Fi APs using iw"""
    cmd = ["sudo", "iw", "dev", iface, "scan"]
    out = run(cmd)
    aps = []

    if not out:
        return aps

    blocks = out.split("BSS ")
    for block in blocks[1:]:
        lines = block.splitlines()
        bssid = lines[0].strip().split()[0]
        ssid = ""
        rssi = None
        freq = None

        for line in lines[1:]:
            line = line.strip()
            if line.startswith("signal:"):
                try:
                    rssi = float(line.split()[1])
                except:
                    rssi = None
            elif line.startswith("SSID:"):
                ssid = line[6:].strip()
            elif line.startswith("freq:"):
                try:
                    freq = int(line.split()[1])
                except:
                    freq = None

        if bssid and rssi is not None:
            aps.append({"ssid": ssid, "bssid": bssid, "freq": freq, "dbm": rssi})

    return aps

def get_signal_for_bssid(bssid, iface=DEFAULT_IFACE):
    target = bssid.lower()
    aps = scan_aps(iface)
    for ap in aps:
        if ap["bssid"].lower() == target:
            dbm = ap["dbm"]
            dist = rssi_to_distance(dbm)
            return dbm, dist
    return None

# ---------- GUI for AP selection ----------
class APSelector:
    def __init__(self, iface=DEFAULT_IFACE):
        self.iface = iface
        self.selected_aps = []
        self.aps = []
        self.listbox = None

    def run(self):
        root = Tk()
        root.title("Select APs to Monitor")
        Label(root, text="Available Wi-Fi APs:").pack(pady=(8, 4))
        self.listbox = Listbox(root, selectmode=MULTIPLE, width=70, height=14)
        self.listbox.pack(padx=10, pady=6)

        self.aps = scan_aps(self.iface)
        if not self.aps:
            messagebox.showwarning("Scan", "No APs found.\nCheck Wi-Fi or interface name.")
        for ap in self.aps:
            self.listbox.insert(END, f"{ap['ssid']}  [{ap['bssid']}]  {ap['dbm']} dBm")

        Button(root, text="Select APs", command=root.quit).pack(pady=(6, 10))
        root.mainloop()

        indices = self.listbox.curselection()
        self.selected_aps = [self.aps[i] for i in indices] if indices else []
        root.destroy()

        if not self.selected_aps:
            print("[!] No APs selected, exiting.")
            raise SystemExit(1)
        print(f"[+] Selected {len(self.selected_aps)} AP(s)")

# ---------- Real-time monitor ----------
class MultiAPMonitor:
    def __init__(self, aps, iface=DEFAULT_IFACE, interval=1.0):
        self.aps = aps
        self.iface = iface
        self.interval = float(interval)
        self.data_log = {ap['bssid'].lower(): [] for ap in aps}
        self.latest_signals = {ap['bssid'].lower(): None for ap in aps}
        self.lock = threading.Lock()
        self.stop_flag = False
        self.scan_thread = threading.Thread(target=self.scan_loop, daemon=True)

        rows = len(aps)
        self.fig, self.axes = plt.subplots(rows, 2, figsize=(11, 4 * rows), squeeze=False)
        self.lines = []

        for i, ap in enumerate(aps):
            (line_rssi,) = self.axes[i][0].plot([], [], label="RSSI (dBm)")
            self.axes[i][0].set_title(f"RSSI: {ap['ssid']}  [{ap['bssid']}]")
            self.axes[i][0].set_xlabel("Time (s)")
            self.axes[i][0].set_ylabel("dBm")
            self.axes[i][0].grid(True)
            self.axes[i][0].legend(loc="upper right")

            (line_dist,) = self.axes[i][1].plot([], [], label="Distance (m)")
            self.axes[i][1].set_title(f"Distance: {ap['ssid']}")
            self.axes[i][1].set_xlabel("Time (s)")
            self.axes[i][1].set_ylabel("m")
            self.axes[i][1].grid(True)
            self.axes[i][1].legend(loc="upper right")

            self.lines.append((line_rssi, line_dist))

        self.ani = None
        self.start = None

    def scan_loop(self):
        """Background scanning using iw"""
        while not self.stop_flag:
            for ap in self.aps:
                sig = get_signal_for_bssid(ap['bssid'], iface=self.iface)
                with self.lock:
                    self.latest_signals[ap['bssid'].lower()] = sig
            time.sleep(self.interval)

    def save_csv_periodically(self, interval=5):
        """Save CSV every few seconds"""
        while not self.stop_flag:
            time.sleep(interval)
            self.save_csv()

    def save_csv(self):
        combined = []
        for ap in self.aps:
            key = ap['bssid'].lower()
            for row in self.data_log.get(key, []):
                combined.append({
                    "ssid": ap['ssid'],
                    "bssid": ap['bssid'],
                    "time": row.get("time"),
                    "dbm": row.get("dbm"),
                    "distance": row.get("distance")
                })
        if combined:
            df = pd.DataFrame(combined)
            df.to_csv(CSV_OUTPUT, index=False)
            print(f"[+] Data saved to {CSV_OUTPUT} ({len(df)} rows, {df['bssid'].nunique()} APs)")

    def update(self, _frame):
        ts = time.time() - self.start
        for i, ap in enumerate(self.aps):
            bssid_key = ap['bssid'].lower()
            with self.lock:
                sig = self.latest_signals[bssid_key]

            if sig:
                dbm, dist = sig
                self.data_log[bssid_key].append({"time": ts, "dbm": dbm, "distance": dist})
                print(f"[{ts:6.1f}s] {ap['ssid']}: {dbm:6.1f} dBm → {dist:6.2f} m")
            else:
                self.data_log[bssid_key].append({"time": ts, "dbm": None, "distance": None})
                print(f"[{ts:6.1f}s] {ap['ssid']}: not visible")

            df = pd.DataFrame(self.data_log[bssid_key]).dropna()
            if not df.empty:
                self.lines[i][0].set_data(df["time"].values, df["dbm"].values)
                self.axes[i][0].relim()
                self.axes[i][0].autoscale_view()
                self.lines[i][1].set_data(df["time"].values, df["distance"].values)
                self.axes[i][1].relim()
                self.axes[i][1].autoscale_view()

        return [ln for pair in self.lines for ln in pair]

    def trilaterate_last_position(self, ap_positions):
        distances = {}
        for ap in self.aps:
            bssid_key = ap['bssid'].lower()
            if self.data_log[bssid_key]:
                last_entry = self.data_log[bssid_key][-1]
                dist = last_entry.get("distance")
                if dist is not None:
                    distances[bssid_key] = dist

        if len(distances) < 3:
            print("[!] Not enough APs for trilateration")
            return None

        keys = list(distances.keys())[:3]
        try:
            (x1, y1), (x2, y2), (x3, y3) = [ap_positions[k] for k in keys]
        except KeyError:
            print("[!] Missing AP positions in dictionary")
            return None

        r1, r2, r3 = [distances[k] for k in keys]
        A = np.array([[2*(x2-x1), 2*(y2-y1)],
                      [2*(x3-x1), 2*(y3-y1)]])
        b = np.array([r1**2 - r2**2 - x1**2 + x2**2 - y1**2 + y2**2,
                      r1**2 - r3**2 - x1**2 + x3**2 - y1**2 + y3**2])
        try:
            pos = np.linalg.lstsq(A, b, rcond=None)[0]
            return tuple(pos)
        except np.linalg.LinAlgError:
            print("[!] Cannot solve trilateration")
            return None

    def run(self):
        self.start = time.time()
        self.scan_thread.start()
        csv_thread = threading.Thread(target=self.save_csv_periodically, args=(5,), daemon=True)
        csv_thread.start()
        try:
            self.ani = animation.FuncAnimation(self.fig, self.update,
                                               interval=int(self.interval * 1000),
                                               blit=False, cache_frame_data=False)
            plt.tight_layout()
            plt.show()
        finally:
            self.stop_flag = True
            self.scan_thread.join()
            csv_thread.join()
            self.save_csv()

# ---------- Main ----------
if __name__ == "__main__":
    # Step 1: Select APs
    selector = APSelector(iface=DEFAULT_IFACE)
    selector.run()

    # Step 2: Sampling interval
    _root = Tk()
    _root.withdraw()
    interval = simpledialog.askfloat("Sampling interval", "Enter sampling interval (s):", minvalue=0.2, initialvalue=1.0, parent=_root)
    _root.destroy()
    if interval is None:
        interval = 1.0

    # Step 3: Enter AP coordinates and save JSON immediately
    AP_POSITIONS = {}
    for ap in selector.selected_aps:
        _root = Tk()
        _root.withdraw()
        x = simpledialog.askfloat("AP Position", f"Enter X coordinate for {ap['ssid']} [{ap['bssid']}]:", parent=_root)
        y = simpledialog.askfloat("AP Position", f"Enter Y coordinate for {ap['ssid']} [{ap['bssid']}]:", parent=_root)
        _root.destroy()
        if x is not None and y is not None:
            AP_POSITIONS[ap['bssid'].lower()] = (x, y)
            # Save JSON immediately
            with open(AP_POS_JSON, "w") as f:
                json.dump({bssid: list(pos) for bssid, pos in AP_POSITIONS.items()}, f, indent=4)
            print(f"[+] Saved AP positions to {AP_POS_JSON}")

    # Step 4: Start monitoring
    monitor = MultiAPMonitor(selector.selected_aps, iface=DEFAULT_IFACE, interval=interval)
    monitor.run()

    # Step 5: Trilateration
    device_pos = monitor.trilaterate_last_position(AP_POSITIONS)
    if device_pos:
        print(f"[+] Estimated device position: {device_pos}")
    else:
        print("[!] Could not estimate device position (need 3+ APs with coordinates).")
