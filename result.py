import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def graph_res(years, years_future, df, reg_res, reg_res_c, mc_res, mc_res_c):
    ysf = np.insert(years_future, 0, 2022)

    lstat = pd.DataFrame({'year':2022, 'linear_pop':df.loc[df['year']==2022, 'population'], 'poly_pop':df.loc[df['year']==2022, 'population']})
    reg_res = pd.concat([lstat, reg_res], ignore_index=True)
    reg_res_c = pd.concat([lstat, reg_res_c], ignore_index=True)


    print(f"reg res ==== {reg_res}")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))

    lower_bound = np.percentile(mc_res, 5, axis=0)
    upper_bound = np.percentile(mc_res, 95, axis=0)
    median_mc = np.percentile(mc_res, 50, axis=0)

    median_mc_graph = np.insert(median_mc, 0, df.loc[df['year']==2022, 'population'])
    print(f"median - {median_mc_graph}")

    ax1.fill_between(years_future, lower_bound, upper_bound, color='red', alpha=0.15, label='Ймовірнісний коридор')
    ax1.plot(ysf, median_mc_graph, 'r', label='Медіана Монте-Карло')

    lower_bound = np.percentile(mc_res_c, 5, axis=0)
    upper_bound = np.percentile(mc_res_c, 95, axis=0)
    median_mc = np.percentile(mc_res_c, 50, axis=0)
    median_mc_graph = np.insert(median_mc, 0, df.loc[df['year'] == 2022, 'population'])

    ax2.fill_between(years_future, lower_bound, upper_bound, color='blue', alpha=0.15, label='Ймовірнісний коридор з Ks')
    ax2.plot(ysf, median_mc_graph, 'b', label='Медіана Монте-Карло з Ks')

    ax1.plot(reg_res['year'], reg_res['linear_pop'], 'r--', label='Лінійна регресія')
    ax1.plot(reg_res['year'], reg_res['poly_pop'], 'm--', label='Поліноміальна регресія')

    ax2.plot(reg_res_c['year'], reg_res_c['linear_pop'], 'y--', label='Лінійна регресія з Ks')
    ax2.plot(reg_res_c['year'], reg_res_c['poly_pop'], 'g--', label='Поліноміальна регресія з Ks')

    ax1.plot(df['year'], df['population'], 'ko-', label='Історичні дані (Факт)')
    ax2.plot(df['year'], df['population'], 'ko-', label='Історичні дані (Факт)')

    plt.suptitle('Результат передбачення')
    ax1.set_xlabel('Рік')
    ax1.set_ylabel('Населення')
    ax1.legend()
    ax1.grid(True, alpha=0.2)
    ax2.set_xlabel('Рік')
    ax2.set_ylabel('Населення')
    ax2.legend()
    ax2.grid(True, alpha=0.2)
    plt.show()