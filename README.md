# Solar Cycle Visualization

A Python tool for visualizing observed and predicted solar cycle data from NOAA's Space Weather Prediction Center.

## Overview

This project fetches real-time solar cycle data from NOAA and creates visualizations comparing observed historical sunspot numbers with predicted future values. Solar cycles are approximately 11-year periods of solar activity that are tracked using sunspot counts.

## Features

- Fetches live data from NOAA Space Weather Prediction Center APIs
- Combines observed and predicted solar cycle data
- Creates clear visualizations using Seaborn Objects
- Handles data preprocessing and datetime conversion automatically

## Requirements

```
pandas
seaborn
matplotlib
requests
```

## Installation

1. Install required packages:
```bash
pip install pandas seaborn matplotlib requests
```

## Usage

Simply run the main script:

```bash
python solar_cycle_viz.py
```

The script will:
1. Fetch observed solar cycle data from NOAA
2. Fetch predicted solar cycle data from NOAA
3. Combine and process the datasets
4. Display a line plot showing both observed and predicted smoothed sunspot numbers over time

## Data Sources

- **Observed Data**: [NOAA Observed Solar Cycle Indices](https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json)
- **Predicted Data**: [NOAA Predicted Solar Cycle](https://services.swpc.noaa.gov/json/solar-cycle/predicted-solar-cycle.json)

## Output

The visualization shows:
- **X-axis**: Date (time-tag)
- **Y-axis**: Smoothed sunspot number
- **Colors**: Distinguished lines for observed (historical) and predicted (future) values

## License

This project is open source and available under the MIT License.

## Acknowledgments

Data provided by NOAA's Space Weather Prediction Center.

