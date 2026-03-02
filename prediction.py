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

def predict(datafile, Ks):
    np.random.seed(42)

    df = dataprocess(datafile)

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

    years = df['year']
    years_future = np.arange(2023, 2033)

    df_res = regr(df_clean, years_future, features, components)
    df_res_c = regr_cataclysm(df_clean, years_future, features, components, Ks)

    mcres = monte_carlo(df_clean, years_future)
    mcres_c = mc_cataclysm(df_clean, years_future, Ks)

    graph_res(years, years_future, df, df_res, df_res_c, mcres, mcres_c)

predict("stats/main_dataset.csv", 0.583)