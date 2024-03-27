import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

seqlen = 24
folder_path = '../../DataSets/Mismatch/'

files = [file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
filenames = [os.path.splitext(file)[0] for file in files]
mismatch_counts_list = [{i: 0 for i in range(1, seqlen)} for _ in files]
colors = sns.color_palette('Set1', len(files)+1)

for idx, file in enumerate(files):
    print(f'Starting analysis of {file}...')

    datapath = os.path.join(folder_path, file)
    datalist = pd.read_csv(datapath)
    datalist = datalist[datalist['label'] == 1][0:200]
    for index, row in datalist.iterrows():
        rna_seq, dna_seq = row[0], row[1]
        for pos in range(len(rna_seq)):

            if rna_seq[pos] == 'N' or dna_seq[pos] == 'N':
                continue

            if rna_seq[pos] != dna_seq[pos]:
                mismatch_counts_list[idx][pos + 1] += 1

positions = list(mismatch_counts_list[0].keys())
plt.figure(figsize=(15, 8))

for idx, mismatch_counts in enumerate(mismatch_counts_list):
    counts = list(mismatch_counts.values())
    if idx in [5,6]:
        plt.plot(positions, counts, marker='o', label=filenames[idx], color=colors[idx+1])
    else:
        plt.plot(positions, counts, marker='o', label=filenames[idx], color=colors[idx])

plt.xlabel('Position', fontsize=14)  
plt.ylabel('Mismatch Count', fontsize=14)  
plt.xticks(range(1, seqlen), fontsize=12)
plt.yticks(fontsize=12) 
plt.grid(False)
plt.savefig(f'./Mismatch Counts_Indel.pdf', bbox_inches='tight', dpi=600, format='pdf')  
plt.legend(title='', loc='upper center', bbox_to_anchor=(0.5, 1), ncol=len(filenames), fontsize=11)  
plt.show()

