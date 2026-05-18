import numpy as np
import pandas as pd

def dataprocess(file):
    df = pd.read_csv(file, sep=';', decimal=',')

    for c in df.columns.tolist():
        if c in df.columns and not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(str).str.replace(',', '.').replace('nan', np.nan).astype(float)

    return df


def get_next_features(target_year, df_historical, feature_names):
    last_row = df_historical.iloc[-1]
    last_year = last_row['year']
    years_ahead = target_year - last_year

    next_values = []
    for col in feature_names:
        if col == 'year':
            next_values.append(target_year)
        else:
            avg_annual_growth = (df_historical[col].iloc[-1] - df_historical[col].iloc[0]) / len(df_historical)
            next_val = last_row[col] + (avg_annual_growth * years_ahead)
            next_values.append(next_val)
    return next_values