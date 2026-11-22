# Data Dictionary

This file describes the physical quantities in 'data/raw/'.
All other released files (features, prior weights, driving preference features, labels) are self-explanatory from their file names and headers.

## De-identification
Device identities are replaced by numeric IDs (D01–D05). Timestamps are transformed into relative operating time, preserving sampling intervals; absolute dates are removed.

## data/raw/

Files: '{i}_device_raw_data.csv' (i = D01–D05), one file per device, one row per
sampling point. Missing values have been removed.

| Column | Unit | Description |
| --- | --- | --- |
| relative_operating_time (s) | s | Cumulative operating time of devices, since the BMS activation |
| SOC(%) | % | Pack state-of-charge, recorded by the BMS |
| I(A) | A | Pack current, positive = charge |
| package_v(V) | V | Pack terminal voltage |
| cell_{k}_v | V | Cell terminal voltage, k = 1-14 |
| point_{k}_T | °C | Temperature detected by individual sensor, k = 1-2 |

Sampling interval: 30 s when device is operating (charging or discharging), 300 s during rest.

