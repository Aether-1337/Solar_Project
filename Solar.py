import seaborn.objects as so
import matplotlib
import pandas as pd
import requests
import datetime

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)

def get_solar_cycle_df():
    urls = [
        'https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json',
        'https://services.swpc.noaa.gov/json/solar-cycle/predicted-solar-cycle.json'
    ]

    dataframes = []

    for url in urls:
        response = requests.get(url)
        if response.status_code == 200 and response.text.strip():
            try:
                data = response.json()
                df = pd.DataFrame(data)
                dataframes.append(df)
            except ValueError as e:
                print("JSON decode error:", e)
        else:
            print("Empty or failed response:", response.status_code)

    return dataframes


dataframes = get_solar_cycle_df()

if len(dataframes) == 2:
    observed_df = dataframes[0]
    predicted_df = dataframes[1]

    print('Observed_df columns:')
    print(observed_df.columns.tolist())
    print('\nPredicted_df columns:')
    print(predicted_df.columns.tolist())
    
   
    if 'predicted_ssn' in predicted_df.columns:
        predicted_df = predicted_df.rename(columns={'predicted_ssn': 'smoothed_ssn'})
    
    
    observed_df['source'] = 'Observed'
    predicted_df['source'] = 'Predicted'
    
    
    combined_df = pd.concat([observed_df, predicted_df], ignore_index=True)
    
    
    combined_df['time-tag'] = pd.to_datetime(combined_df['time-tag'])
    
    
    print('\nCombined df info:')
    print(combined_df.groupby('source').size())
    print('\nSample of combined data:')
    print(combined_df[['time-tag', 'smoothed_ssn', 'source']].tail(10))
    
    
    (
        so.Plot(combined_df, x='time-tag', y='smoothed_ssn', color='source')
        .add(so.Line())
        .label(x='Date', y='Smoothed Sunspot Number', title='Observed and Predicted Smoothed Sunspot Numbers')
        .show()
    )
else:
    print("One or both datasets failed to load.")