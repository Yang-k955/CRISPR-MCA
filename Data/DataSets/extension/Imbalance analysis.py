

import pandas as pd
import numpy as np
from scipy.stats import entropy, variation

data = {'dataset': ['Hek293t', 'K562', 'D3', 'D8', 'D5', 'D6', 'D7'],
        'old_pos': [536, 120, 54, 354, 56, 3767, 52],
        'new_pos': [4935, 1076, 330, 3381, 281, 41412, 327],
        'neg': [132378, 20199, 95775, 294180, 383407, 213966, 10077]}
df = pd.DataFrame(data)

df['old_label_counts'] = df.apply(lambda row: [row['old_pos'], row['neg']], axis=1)
df['new_label_counts'] = df.apply(lambda row: [row['new_pos'], row['neg']], axis=1)

df['old_ir'] = df.apply(lambda row: row['neg'] / row['old_pos'], axis=1)
df['new_ir'] = df.apply(lambda row: row['neg'] / row['new_pos'], axis=1)

df['cv_old_ir'] = df.apply(lambda row: variation(row['old_label_counts']) if len(row['old_label_counts']) >= 1 else 0, axis=1)
df['cv_new_ir'] = df.apply(lambda row: variation(row['new_label_counts']) if len(row['new_label_counts']) >= 1 else 0, axis=1)

def calc_entropy(pos, neg):
    total = pos + neg
    p_pos = pos / total
    p_neg = neg / total
    return entropy([neg , pos], base=2)

df['old_entropy'] = df.apply(lambda row: calc_entropy(row['old_pos'], row['neg']), axis=1)
df['new_entropy'] = df.apply(lambda row: calc_entropy(row['new_pos'], row['neg']), axis=1)
df = df.sort_values(by='old_ir', ascending=True)
df.to_csv("./imbanlance.csv")


# import csv
#
# data = [
#     ['dataset', 'pos->new_pos', 'IR', 'CVIR', 'IE'],
#     ['D6', '3767->41412', '56.8001->5.1668', '0.9654->0.6757', '0.1260->0.6395'],
#     ['K562', '120->1076', '168.3250->18.7723', '0.9882->0.8988', '0.0522->0.2888'],
#     ['D7', '52->327', '193.7885->30.8165', '0.9897->0.9371', '0.0464->0.2015'],
#     ['Hek293t', '536->4935', '246.9739->26.8243', '0.9919->0.9281', '0.0379->0.2234'],
#     ['D8', '354->3381', '831.0169->87.0098', '0.9976->0.9773', '0.0134->0.0897'],
#     ['D3', '54->330', '1773.6111->290.2273', '0.9989->0.9931', '0.0069->0.0331'],
#     ['D5', '56->281', '6846.5536->1364.4377', '0.9997->0.9985', '0.0021->0.0087']
# ]
#
# with open('imbanlance.csv', mode='w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(data)