import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

indels = 0

folder_path = '../../DataSets/test/'
files = [(file, os.path.splitext(file)[0], os.path.splitext(file)[1]) for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
all_mismatch_matrices = []

for file, filename, extension in files:
    print(f'starting {filename}........................')
    datapath = os.path.join(folder_path, file)

    datalist = pd.read_csv(datapath)
    datalist = datalist[datalist['label'] == 1]

    datalist = np.array(datalist)
    data = datalist[:, 0:2]
    label = datalist[:, 2]

    encoding_map = {
        'AC': 0, 'AG': 1, 'AT': 2,
        'CA': 3, 'CG': 4, 'CT': 5,
        'GA': 6, 'GC': 7, 'GT': 8,
        'TA': 9, 'TC': 10, 'TG': 11,
    }

    mismatch_matrix = pd.DataFrame(0, index=range(12), columns=range(23))
    keys_list = ["rA - dC", "rA - dG", "rA - dT", "rC - dA", "rC - dG", "rC - dT", "rG - dA", "rG - dC", "rG - dT", "rT - dA", "rT - dC", "rT - dG"]

    for i in range(len(data)):
        pair = data[i]
        seq1 = pair[0]
        seq2 = pair[1]
        for j in range(len(seq2)):
            a = seq1[j].upper()
            b = seq2[j].upper()
            if a == 'N':
                a = b
            if b == 'N':
                b = a

            if a == '_':
                a = '-'
            if b == '_':
                b = '-'

            if a != b:
                piars = a + b
                mappiars = encoding_map[piars]
                mismatch_matrix.loc[mappiars, j] = mismatch_matrix.loc[mappiars, j] + 1
    mismatch_matrix.index = mismatch_matrix.index + 1  
    mismatch_matrix.columns = mismatch_matrix.columns + 1  
    all_mismatch_matrices.append(mismatch_matrix)

fig, axes = plt.subplots(2, 2, figsize=(15, 8))

for i, mismatch_matrix in enumerate(all_mismatch_matrices):
    row = i // 2 
    col = i % 2 
    ax = sns.heatmap(mismatch_matrix, cmap='YlOrRd', square=True, annot=False, fmt='d', cbar=True, cbar_kws={"fraction": 0.1}, ax=axes[row, col])
    ax.set_yticklabels(keys_list, rotation=1)
    ax.set_xlabel('Mismatch Position', fontsize=14)
    ax.set_ylabel('Mismatch Type', fontsize=14)
    ax.set_title(f'{files[i][1]}')

plt.savefig('all_mismatch_matrices.pdf' , bbox_inches='tight', dpi=600, format='pdf')  
plt.show()  

# ---------------------------------------------------------
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import os
#

# indels = 0
#
# folder_path = '../../DataSets/Mismatch/'
# files = [(file, os.path.splitext(file)[0], os.path.splitext(file)[1]) for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
# all_mismatch_matrices = []
#
# for file, filename, extension in files:
#     print(f'starting {filename}........................')
#     datapath = os.path.join(folder_path, file)
#
#     datalist = pd.read_csv(datapath)
#     datalist = datalist[datalist['label'] == 1]
#
#     datalist = np.array(datalist)
#     data = datalist[:, 0:2]
#     label = datalist[:, 2]
#
#     encoding_map = {
#         'AC': 0, 'AG': 1, 'AT': 2,
#         'CA': 3, 'CG': 4, 'CT': 5,
#         'GA': 6, 'GC': 7, 'GT': 8,
#         'TA': 9, 'TC': 10, 'TG': 11,
#         '-A': 0, '-C': 1, '-G': 2, '-T': 3,
#         'A-': 4, 'C-': 5, 'G-': 6, 'T-': 7,
#         '--': 8
#     }
#
#    
#     if indels:
#         mismatch_matrix = pd.DataFrame(0, index=range(8), columns=range(24))
#         keys_list = ["- A", "- C", "- G", "- T", "A -", "C -", "G -", "T -"]
#     else:
#         mismatch_matrix = pd.DataFrame(0, index=range(12), columns=range(23))
#         keys_list = ["AC", "AG", "AT", "CA", "CG", "CT", "GA", "GC", "GT", "TA", "TC", "TG"]
#
#  
#     for i in range(len(data)):
#         pair = data[i]
#         seq1 = pair[0]
#         seq2 = pair[1]
#         for j in range(len(seq2)):
#             a = seq1[j].upper()
#             b = seq2[j].upper()
#             if a == 'N':
#                 a = b
#             if b == 'N':
#                 b = a
#
#             if a == '_':
#                 a = '-'
#             if b == '_':
#                 b = '-'
#
#             if a != b:
#                 piars = a + b
#                 mappiars = encoding_map[piars]
#                 mismatch_matrix.loc[mappiars, j] = mismatch_matrix.loc[mappiars, j] + 1
#     mismatch_matrix.index = mismatch_matrix.index + 1  # Modify row index
#     mismatch_matrix.columns = mismatch_matrix.columns + 1  # Modify column index
#     all_mismatch_matrices.append((filename, mismatch_matrix))
#
# # Save each heatmap separately with larger font size and prominent titles
# for filename, mismatch_matrix in all_mismatch_matrices:
#     fig, ax = plt.subplots(figsize=(15, 10))
#     sns.heatmap(mismatch_matrix, cmap='YlOrRd', square=True, annot=False, fmt='d', cbar=True, cbar_kws={"fraction": 0.025}, ax=ax)
#     ax.set_yticklabels(keys_list, rotation=0, fontsize=12)  # Larger font size for y-axis labels
#     ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)  # Larger font size for x-axis labels
#     ax.set_xlabel('Mismatch Position', fontsize=14)  # Larger font size for x-axis label
#     ax.set_ylabel('Mismatch Type', fontsize=14)  # Larger font size for y-axis label
#     ax.set_title(f'{filename}', fontsize=16)  # Larger font size for title
#     plt.subplots_adjust(hspace=0.5, wspace=0.3)  
#
#     # Save the heatmap as a separate image
#     plt.savefig(f'./{filename}_mismatch_matrix.png', bbox_inches='tight')  # Use bbox_inches='tight' to prevent clipping
#     plt.close()  # Close the current figure

# Show or save the final plots (optional)
# plt.show()
