import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from dataprocess import dataprocess, get_next_features
import matplotlib
import matplotlib.pyplot as plt
from montecarlo import monte_carlo, mc_cataclysm
from regr import regr, regr_cataclysm
from result import graph_res

def predict(df, Ks, years, years_future, features, fig):

    mask_train = df['gdp'].notna() & df['salary_index'].notna()
    if mask_train.any():
        reg_proxy = LinearRegression()
        reg_proxy.fit(df.loc[mask_train, ['gdp']], df.loc[mask_train, 'salary_index'])
        mask_to_fill = df['salary_index'].isna() & df['gdp'].notna()
        if mask_to_fill.any():
            df.loc[mask_to_fill, 'salary_index'] = reg_proxy.predict(df.loc[mask_to_fill, ['gdp']])

    df_clean = df[df['year'] >= 2010].copy().ffill()
    df_clean['trade_balance'] = df_clean['export'] - df_clean['import']

    features = ['year', 'gdp', 'salary_index', 'trade_balance']
    components = ['births', 'deaths', 'immigrants', 'emigrants']

    df_res = regr(df_clean, years_future, features, components)
    df_res_c = regr_cataclysm(df_clean, years_future, features, components, Ks)

    mcres = monte_carlo(df_clean, years_future)
    mcres_c = mc_cataclysm(df_clean, years_future, Ks)

    graph_res(years, years_future, df, df_res, df_res_c, mcres, mcres_c, fig)

    export_data = pd.DataFrame({
        'Year': years_future,
        'Linear_Inertial': df_res['linear_pop'].values,
        'Poly_Inertial': df_res['poly_pop'].values,
        'Linear_Shock': df_res_c['linear_pop'].values,
        'Poly_Shock': df_res_c['poly_pop'].values,
        'MonteCarlo_Mean_Inertial': np.mean(mcres, axis=0),
        'MonteCarlo_Mean_Shock': np.mean(mcres_c, axis=0)
    })

    return export_data
