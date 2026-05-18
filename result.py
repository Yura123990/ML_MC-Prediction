import pandas as pd
import numpy as np


def graph_res(years, years_future, df, reg_res, reg_res_c, mc_res, mc_res_c, fig=None):
    if fig is None:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
    else:
        fig.clear()
        ax1, ax2 = fig.subplots(2, 1)

    last_year = years[-1]
    ysf = np.insert(years_future, 0, last_year)
    last_pop = df.loc[df['year'] == last_year, 'population'].values[0]

    lstat = pd.DataFrame({'year': [last_year], 'linear_pop': [last_pop], 'poly_pop': [last_pop]})
    reg_res = pd.concat([lstat, reg_res], ignore_index=True)
    reg_res_c = pd.concat([lstat, reg_res_c], ignore_index=True)

    # Графік 1 (Без Ks)
    lower_bound = np.percentile(mc_res, 5, axis=0)
    upper_bound = np.percentile(mc_res, 95, axis=0)
    median_mc = np.percentile(mc_res, 50, axis=0)
    median_mc_graph = np.insert(median_mc, 0, last_pop)

    ax1.fill_between(years_future, lower_bound, upper_bound, color='red', alpha=0.15, label='Ймовірнісний коридор')
    ax1.plot(ysf, median_mc_graph, 'r', label='Медіана Монте-Карло')
    ax1.plot(reg_res['year'], reg_res['linear_pop'], 'r--', label='Сценарій інерційного розвитку')
    ax1.plot(reg_res['year'], reg_res['poly_pop'], 'm--', label='Сценарій кумулятивного впливу')
    ax1.plot(df['year'], df['population'], 'ko-', label='Історичні дані (Факт)')

    # Графік 2 (З Ks)
    lower_bound_c = np.percentile(mc_res_c, 5, axis=0)
    upper_bound_c = np.percentile(mc_res_c, 95, axis=0)
    median_mc_c = np.percentile(mc_res_c, 50, axis=0)
    median_mc_graph_c = np.insert(median_mc_c, 0, last_pop)

    ax2.fill_between(years_future, lower_bound_c, upper_bound_c, color='blue', alpha=0.15,
                     label='Ймовірнісний коридор з Ks')
    ax2.plot(ysf, median_mc_graph_c, 'b', label='Медіана Монте-Карло з Ks')
    ax2.plot(reg_res_c['year'], reg_res_c['linear_pop'], 'y--', label='Сценарій інерційного розвитку з Ks')
    ax2.plot(reg_res_c['year'], reg_res_c['poly_pop'], 'g--', label='Сценарій кумулятивного впливу з Ks')
    ax2.plot(df['year'], df['population'], 'ko-', label='Історичні дані (Факт)')

    fig.suptitle('Результат прогнозування')
    ax1.set_ylabel('Населення');
    ax1.legend(fontsize='small');
    ax1.grid(True, alpha=0.2)
    ax2.set_ylabel('Населення');
    ax2.legend(fontsize='small');
    ax2.grid(True, alpha=0.2)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])