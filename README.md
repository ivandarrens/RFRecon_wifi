# Multi-AP WIFI RF Monitor and Trilateration Tool

This project is a Python based program that monitor WIFI signal strength and using trilateration tool to estimate multiple Access Point (AP) relative distance to a device running the program. The program logs the APs RSSI stregth and coverts it to relative distance and visualizes the data, using Matplotlib, into a heatmap.

---

## Features

- **Automatic WIFI Scanning** using Linux's 'iw' function.
- **GUI Selection** of AP using Tkinter.
- **Plot RSSI and Distance** for each APs.
- **Live CSV Logging** of MAC Address, Network name, RSSI value, and Distance.
- **Using Trilateration** to estimate device positions using known AP coordinates.
- **Set Custom Smapling Interval**
- **Auto-save AP Coordinate** in a JSON file.

---

## Requirements

### System
- Linux (Tested on Ubuntu/Debian system)
- WIFI NIC card.
- Python 3.8+

### Python Dependencies/Libraries
Install required packages:

```bash
pip install matplotlib pandas numpy
```
(Optional: Tkinter)
```bash
sudo apt install python3-tk
```

---

# License
This project is released under the MIT License

---

# Contributions 

Pull Requests and suggestions are Welcome!
If you wish to add improvements, feel free to submit a PR!

---
