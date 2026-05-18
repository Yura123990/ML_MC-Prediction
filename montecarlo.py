import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def monte_carlo(df, years_future):
    stats = {}
    for comp in ['births', 'deaths', 'immigrants', 'emigrants']:
        stats[comp] = {
            'mean': df[comp].mean(),
            'std': df[comp].std()
        }

    print("Історична статистика для Монте-Карло зібрана!")
    n_simulations = 50000
    initial_pop = df['population'].iloc[-1]

    mc_population_results = np.zeros((n_simulations, len(years_future)))

    for i in range(n_simulations):
        current_pop = initial_pop

        for j, yr in enumerate(years_future):
            rand_N = np.random.normal(stats['births']['mean'], stats['births']['std'])
            rand_M = np.random.normal(stats['deaths']['mean'], stats['deaths']['std'])
            rand_I = np.random.normal(stats['immigrants']['mean'], stats['immigrants']['std'])
            rand_E = np.random.normal(stats['emigrants']['mean'], stats['emigrants']['std'])

            current_pop = current_pop + (rand_N - rand_M) + (rand_I - rand_E)

            mc_population_results[i, j] = current_pop

    return mc_population_results


def mc_cataclysm(df, years_future, Ks):
    stats = {}
    for comp in ['births', 'deaths', 'immigrants', 'emigrants']:
        stats[comp] = {
            'mean': df[comp].mean(),
            'std': df[comp].std()
        }
    print(Ks)
    print(Ks * 40000000)

    print("Історична статистика для Монте-Карло зібрана!")

    n_simulations = 3000
    initial_pop = df['population'].iloc[-1]

    mc_population_results = np.zeros((n_simulations, len(years_future)))

    for i in range(n_simulations):
        current_pop = initial_pop

        for j, yr in enumerate(years_future):
            rand_N = np.random.normal(stats['births']['mean'], stats['births']['std'])
            rand_M = np.random.normal(stats['deaths']['mean'], stats['deaths']['std'])
            rand_I = np.random.normal(stats['immigrants']['mean'], stats['immigrants']['std'])
            rand_E = np.random.normal(stats['emigrants']['mean'], stats['emigrants']['std'])

            current_pop = current_pop + (rand_N - rand_M) + (rand_I - rand_E)

            mc_population_results[i, j] = current_pop * Ks

    return mc_population_results
