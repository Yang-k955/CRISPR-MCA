
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('cleaned_Listgarten_Extension.csv')
data = data[data['type'] == 'extension']

encoding_map = {
    'AC': 0, 'AG': 1, 'AT': 2,
    'CA': 3, 'CG': 4, 'CT': 5,
    'GA': 6, 'GC': 7, 'GT': 8,
    'TA': 9, 'TC': 10, 'TG': 11
}

data['code'] = data.apply(lambda x: encoding_map[x['base'] + x['change']], axis=1)
mismatch_matrix = pd.DataFrame(0, index=range(12), columns=range(21))

for i in range(1, 22):
    temp = data[data['position'] == i]['code'].value_counts()
    for j in temp.index:
        mismatch_matrix.loc[j, i-1] = mismatch_matrix.loc[j, i-1]+temp[j]

keys_list = ["rA - dC", "rA - dG", "rA - dT", "rC - dA", "rC - dG", "rC - dT", "rG - dA", "rG - dC", "rG - dT", "rT - dA", "rT - dC", "rT - dG"]
keys_list = [key.replace('r', '').replace('d', '').replace('-', 'to') for key in keys_list]
mismatch_matrix = mismatch_matrix.set_index([keys_list])

fig, ax = plt.subplots(figsize=(16, 10))
sns.heatmap(mismatch_matrix, cmap='YlGnBu', square=True, annot=False, fmt='d', cbar=True, cbar_kws={"fraction": 0.027, "pad": 0.03})
# ax.set_title('Mismatch Matrix', fontsize=14)
ax.set_yticklabels(keys_list, rotation=0, fontsize=12)
ax.set_xlabel('Location of substitute', fontsize=14)
ax.set_ylabel('Type of substitute', fontsize=14)
ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
ax.set_yticklabels(keys_list, rotation=0, fontsize=12)
x_labels = [str(i + 1) for i in range(21)]
ax.set_xticks(range(21))
ax.set_xticklabels(x_labels, fontsize=12)

plt.savefig("./Result/change_hotmap.svg", dpi=600, format="svg", bbox_inches='tight')
# plt.savefig('./Result/change_hotmap.pdf' , bbox_inches='tight', dpi=600, format='pdf') 
plt.show()
