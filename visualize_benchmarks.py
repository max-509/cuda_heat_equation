import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

benchmarks_pd = pd.read_csv('benchmarks.csv', sep=';')

time_col = 'Elapsed Time'
iters_without_err_col = 'Iters without err counting'


def plot_benchmarks_by_target():
    versions = ['OpenACC', 'CUDA naive',
                'CUDA without redundant sync', 'CUDA once mem alloc']
    n_blocks_iters_unique = np.unique(benchmarks_pd[['Grid size', 'Number of iters']], axis=0)
    
    print(n_blocks_iters_unique)

    

    for n_blocks, n_iters in n_blocks_iters_unique:
        fig, ax = plt.subplots(figsize=(18, 9))
        print()
        benchmarks_by_blocks_iters = benchmarks_pd[(benchmarks_pd['Grid size'] == n_blocks) &
                                             (benchmarks_pd['Number of iters'] == n_iters)]

        bars = ax.bar(benchmarks_by_blocks_iters['Algo ver'],
               benchmarks_by_blocks_iters['Elapsed Time'])
        ax.set_xlabel('Algorithm version')
        ax.set_ylabel('Elapsed time, seconds')
        ax.set_xticks(list(range(len(benchmarks_by_blocks_iters['Algo ver']))))
        ax.set_xticklabels(benchmarks_by_blocks_iters['Algo ver'], rotation=10)
        ax.bar_label(bars)

        fig.suptitle(f'Number of blocks: {n_blocks}, Number of iters: {n_iters}')
        fig.savefig(f'benchmarks_{n_blocks}_blocks_{n_iters}_iters.png')


plot_benchmarks_by_target()
