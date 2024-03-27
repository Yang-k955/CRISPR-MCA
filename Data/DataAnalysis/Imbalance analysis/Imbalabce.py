import pandas as pd
import os
import numpy as np
from scipy.stats import variation, entropy
import matplotlib.pyplot as plt

def gini_coefficient(label_counts):
   
    sorted_counts = np.sort(label_counts)
    cum_counts = np.cumsum(sorted_counts)
    total_counts = cum_counts[-1]
    x_lorenz = np.arange(1, len(label_counts) + 1) / len(label_counts)  # 等份划分
    y_lorenz = cum_counts / total_counts
    gini = 1 - 2 * np.trapz(y_lorenz, x_lorenz)

    plt.figure()
    plt.plot(x_lorenz, y_lorenz, label='Lorenz Curve')
    plt.fill_between(x_lorenz, 0, y_lorenz, alpha=0.3)
    plt.plot([0, 1], [0, 1], label='Line of Equality', linestyle='--', color='red')

    plt.title(f'Gini Coefficient: {gini:.6f}')
    plt.xlabel('Fraction of Population')
    plt.ylabel('Cumulative Share of Labels')
    plt.legend()
    plt.show()

    return gini

folder_path = '../DataSets/Mismatch'
file_names = os.listdir(folder_path)
datasets_stats = []

for dataset in file_names:
    data = pd.read_csv(f'{folder_path}/{dataset}')
    try:
        unique_values = data['on_seq'].unique()
        label_counts = data['label'].value_counts()
    except:
        pass

    try:
        unique_values = data['sgRNA_seq'].unique()
        label_counts = data['label'].value_counts()
    except:
        pass

    try:
        unique_values = data['crRNA'].unique()
        label_counts = data['label'].value_counts()
    except:
        pass

    ir = label_counts.max() / label_counts.min()
    cvir = variation(label_counts) if len(label_counts) >= 1 else 0

    info_entropy = entropy(label_counts, base=2)

    datasets_stats.append({
        "Dataset": dataset,
        "sgrna Count": len(unique_values),
        "pos": label_counts[0],
        "nea": label_counts[1],
        "Imbalance Ratio": ir,
        "Coefficient of Variation of IR": cvir,
        "Information Entropy": info_entropy
    })

stats_df = pd.DataFrame(datasets_stats)
print(stats_df)
stats_df.to_csv('./dataset.csv')
