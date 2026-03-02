import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from dataprocess import get_next_features

def regr(df, years_future, features, components):
    models = {}
    models_lin = {}
    models_poly = {}

    for comp in components:
        # 1. ЧИСТО ЛІНІЙНА РЕГРЕСІЯ (Пряма залежність)
        # Вона каже: "Якщо ВВП виросте на 1%, народжуваність виросте на Х%"
        model_l = make_pipeline(StandardScaler(), LinearRegression())
        model_l.fit(df[features], df[comp])
        models_lin[comp] = model_l

        # 2. ПОЛІНОМІАЛЬНА РЕГРЕСІЯ (Крива залежність)
        # Вона каже: "Зв'язок складніший, він може прискорюватись або сповільнюватись"
        model_p = make_pipeline(PolynomialFeatures(degree=2), StandardScaler(), Ridge(alpha=1.0))
        model_p.fit(df[features], df[comp])
        models_poly[comp] = model_p

    pop_lin = df['population'].iloc[-1]
    pop_poly = df['population'].iloc[-1]

    results = []

    for yr in years_future:
        current_x = get_next_features(yr, df, features)

        l_preds = {c: models_lin[c].predict([current_x])[0] for c in components}
        pop_lin += (l_preds['births'] - l_preds['deaths']) + (l_preds['immigrants'] - l_preds['emigrants'])

        p_preds = {c: models_poly[c].predict([current_x])[0] for c in components}
        pop_poly += (p_preds['births'] - p_preds['deaths']) + (p_preds['immigrants'] - p_preds['emigrants'])

        results.append({
            'year': yr,
            'linear_pop': pop_lin,
            'poly_pop': pop_poly
        })

    df_res = pd.DataFrame(results)
    return df_res


def regr_cataclysm(df, years_future, features, components, Ks):
    models = {}

    models_lin = {}
    models_poly = {}

    for comp in components:
        model_l = make_pipeline(StandardScaler(), LinearRegression())
        model_l.fit(df[features], df[comp])
        models_lin[comp] = model_l

        model_p = make_pipeline(PolynomialFeatures(degree=2), StandardScaler(), Ridge(alpha=1.0))
        model_p.fit(df[features], df[comp])
        models_poly[comp] = model_p

    pop_lin = df['population'].iloc[-1]
    pop_poly = df['population'].iloc[-1]

    results = []

    for yr in years_future:
        current_x = get_next_features(yr, df, features)

        l_preds = {c: models_lin[c].predict([current_x])[0] for c in components}
        pop_lin += (l_preds['births'] - l_preds['deaths']) + (l_preds['immigrants'] - l_preds['emigrants'])

        p_preds = {c: models_poly[c].predict([current_x])[0] for c in components}
        pop_poly += (p_preds['births'] - p_preds['deaths']) + (p_preds['immigrants'] - p_preds['emigrants'])

        results.append({
            'year': yr,
            'linear_pop': pop_lin*Ks,
            'poly_pop': pop_poly*Ks
        })

    df_res = pd.DataFrame(results)
    return df_res
