#!/usr/bin/env python3
import subprocess
import time
import matplotlib
matplotlib.use("Qt5Agg")  # keep your Qt backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from tkinter import Tk, Label, Button, Listbox, MULTIPLE, END, simpledialog, messagebox


# ---------- Config ----------
DEFAULT_IFACE = "wlp2s0"
# ---------- Helpers ----------
def run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, shell=False).stdout.strip()
    except Exception as e:
        print(f"[run] command failed {cmd}: {e}")
        return ""

def percent_to_dbm(pct):
    # heuristic mapping when nmcli exposes only Signal %
    return (pct / 2.0) - 100.0

def rssi_to_distance(rssi_dbm, tx_power=-50.0, path_loss_exp=2.5):
    # log-distance path loss model
    return 10.0 ** ((tx_power - rssi_dbm) / (10.0 * path_loss_exp))

# ---------- Robust nmcli scanners ----------
def scan_aps(iface=DEFAULT_IFACE):
    """
    Return a list of visible APs with robust parsing against MAC colons.
    Each item: {'ssid','bssid','pct','dbm'}
    """
    cmd = ["nmcli", "-t", "-f", "IN-USE,SSID,BSSID,SIGNAL,CHAN,SECURITY", "device", "wifi", "list"]
    if iface:
        cmd += ["ifname", iface]
    out = run(cmd)

    aps = []
    for raw in out.splitlines():
        if not raw:
            continue
        parts = raw.split(":")
        # need at least the tail-4 fields to exist
        if len(parts) < 6:
            # unexpected line; skip safely
            # print(f"[nmcli] skipped: {raw}")
            continue
        bssid, signal, chan, security = parts[-4], parts[-3], parts[-2], parts[-1]
        # SSID may contain colons; reconstruct from the middle
        ssid = ":".join(parts[1:-4]).strip()
        if not ssid or ssid == "--":
            # hidden SSID — skip for selection UI
            continue
        try:
            pct = int((signal or "").strip())
        except Exception:
            continue
        dbm = percent_to_dbm(pct)
        aps.append({"ssid": ssid, "bssid": (bssid or "").strip(), "pct": pct, "dbm": dbm})
    return aps

def get_signal_for_bssid(bssid, iface=DEFAULT_IFACE):
    """
    Return (pct, dbm, dist) for an exact BSSID (case-insensitive). None if not found.
    Uses the robust tail parsing again.
    """
    cmd = ["nmcli", "-t", "-f", "IN-USE,SSID,BSSID,SIGNAL,CHAN,SECURITY", "device", "wifi", "list"]
    if iface:
        cmd += ["ifname", iface]
    out = run(cmd)
    target = (bssid or "").lower()
    for raw in out.splitlines():
        if not raw:
            continue
        parts = raw.split(":")
        if len(parts) < 6:
            continue
        b, signal = parts[-4], parts[-3]
        if (b or "").lower().strip() == target:
            try:
                pct = int((signal or "").strip())
            except Exception:
                return None
            dbm = percent_to_dbm(pct)
            dist = rssi_to_distance(dbm)
            return pct, dbm, dist
    return None

# ---------- GUI to select multiple APs ----------
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
            messagebox.showwarning("Scan", "No APs found.\n• Check iface (e.g., wlp2s0)\n• Ensure Wi-Fi is on\n• Try closer to APs")
        for ap in self.aps:
            self.listbox.insert(END, f"{ap['ssid']}  [{ap['bssid']}]  {ap['pct']}%")

        Button(root, text="Select APs", command=root.quit).pack(pady=(6, 10))
        root.mainloop()

        indices = self.listbox.curselection()
        self.selected_aps = [self.aps[i] for i in indices] if indices else []
        root.destroy()

        if not self.selected_aps:
            print("[!] No APs selected, exiting.")
            raise SystemExit(1)
        print(f"[+] Selected {len(self.selected_aps)} AP(s)")

# ---------- Real-time multi-AP monitor (one window, side-by-side subplots) ----------
class MultiAPMonitor:
    def __init__(self, aps, iface=DEFAULT_IFACE, interval=1.0):
        self.aps = aps
        self.iface = iface
        self.interval = float(interval)
        self.data_log = {ap['bssid'].lower(): [] for ap in aps}

        # One figure; N rows (APs) x 2 cols (RSSI, Distance)
        rows = len(aps)
        self.fig, self.axes = plt.subplots(rows, 2, figsize=(11, 4 * rows), squeeze=False)
        self.lines = []  # [(line_rssi, line_dist), ...]

        for i, ap in enumerate(aps):
            # RSSI axis
            (line_rssi,) = self.axes[i][0].plot([], [], label="RSSI (dBm)")
            self.axes[i][0].set_title(f"RSSI: {ap['ssid']}  [{ap['bssid']}]")
            self.axes[i][0].set_xlabel("Time (s)")
            self.axes[i][0].set_ylabel("dBm")
            self.axes[i][0].grid(True)
            self.axes[i][0].legend(loc="upper right")

            # Distance axis
            (line_dist,) = self.axes[i][1].plot([], [], label="Distance (m)")
            self.axes[i][1].set_title(f"Distance: {ap['ssid']}")
            self.axes[i][1].set_xlabel("Time (s)")
            self.axes[i][1].set_ylabel("m")
            self.axes[i][1].grid(True)
            self.axes[i][1].legend(loc="upper right")

            self.lines.append((line_rssi, line_dist))

        # Keep a reference to the animation (prevents GC warning)
        self.ani = None
        self.start = None

    def update(self, _frame):
        ts = time.time() - self.start
        for i, ap in enumerate(self.aps):
            bssid_key = ap['bssid'].lower()
            sig = get_signal_for_bssid(ap['bssid'], iface=self.iface)
            if sig:
                pct, dbm, dist = sig
                self.data_log[bssid_key].append({"time": ts, "pct": pct, "dbm": dbm, "distance": dist})
                print(f"[{ts:6.1f}s] {ap['ssid']}: {pct:3d}% (~{dbm:6.1f} dBm) → {dist:6.2f} m")
            else:
                # still log time to keep timeline continuous
                self.data_log[bssid_key].append({"time": ts, "pct": None, "dbm": None, "distance": None})
                print(f"[{ts:6.1f}s] {ap['ssid']}: not visible")

            # Update plots from non-null rows only
            df = pd.DataFrame(self.data_log[bssid_key]).dropna()
            if not df.empty:
                # RSSI
                self.lines[i][0].set_data(df["time"].values, df["dbm"].values)
                self.axes[i][0].relim()
                self.axes[i][0].autoscale_view()
                # Distance
                self.lines[i][1].set_data(df["time"].values, df["distance"].values)
                self.axes[i][1].relim()
                self.axes[i][1].autoscale_view()

        # Return all Line2D artists
        return [ln for pair in self.lines for ln in pair]

    def run(self):
        self.start = time.time()
        self.ani = animation.FuncAnimation(
            self.fig,
            self.update,
            interval=int(self.interval * 1000),
            blit=False,
            cache_frame_data=False,
        )
        plt.tight_layout()
        plt.show()

        # Save all data to CSV when the window closes
        combined = []
        for ap in self.aps:
            key = ap["bssid"].lower()
            for row in self.data_log[key]:
                combined.append({"ssid": ap["ssid"], "bssid": ap["bssid"], **row})
        df = pd.DataFrame(combined)
        df.to_csv("multi_ap_rssi_log.csv", index=False)
        print("[+] Data saved to multi_ap_rssi_log.csv")

# ---------- Main ----------
if __name__ == "__main__":
    # Select APs from a fresh scan
    selector = APSelector(iface=DEFAULT_IFACE)
    selector.run()

    # Ask for sampling interval
    # (create a tiny transient Tk root for the dialog)
    _root = Tk(); _root.withdraw()
    interval = simpledialog.askfloat("Sampling interval", "Enter sampling interval (s):",
                                     minvalue=0.2, initialvalue=1.0, parent=_root)
    _root.destroy()
    if interval is None:
        print("[!] No interval entered; defaulting to 1.0s")
        interval = 1.0

    monitor = MultiAPMonitor(selector.selected_aps, iface=DEFAULT_IFACE, interval=interval)
    monitor.run()



