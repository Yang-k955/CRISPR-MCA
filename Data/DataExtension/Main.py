
from crisot import CRISOT
import pandas as pd
import os
def cal_score(sgr, tar):
    model = CRISOT()
    return model.single_score_(sgr, tar)


def cal_spec(df_in, On='On', Off='Off'):
    model = CRISOT()
    spec = model.spec(data_df=df_in, On=On, Off=Off, out_df=False)
    return spec

def cal_casoffinder_spec(sgr, tar, mm, dev='G0'):
    model = CRISOT()
    spec = model.CasoffinderSpec_(sgr, tar, mm=mm, dev=dev)
    return spec

def crisotex(grna, dna, mm):
    bases = ['A', 'T', 'C', 'G']
    results = []

    for position in range(20): 
        for base in bases:
            oribase = grna[position]
            if base != oribase:
                new_grna = grna[:position] + base + grna[position + 1:]
                efficiency = cal_score(new_grna, dna)
                results.append((f'position {position + 1}, base {oribase}->{base}', new_grna, dna, efficiency))

    sorted_results = sorted(results, key=lambda x: x[3], reverse=True)[:15]

    for i, (desc, sgr, tar, efficiency) in enumerate(sorted_results):
        specificity = cal_casoffinder_spec(sgr, tar, mm)
        sorted_results[i] = (desc, sgr, tar, efficiency, specificity, efficiency*0.5+specificity*0.5)

    return sorted(sorted_results, key=lambda x: x[5], reverse=True)

folder_path = '../DataSets/t/'
files = [(file, os.path.splitext(file)[0], os.path.splitext(file)[1]) for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))]
mm_dic = [1]
for mm in mm_dic:
    for file, filename, extension in files:
        print(f'staring {filename} with {mm} mismatches........................')
        if os.path.exists(f'../DataSets/extension/{filename}_Extension{mm}.csv'):
            print(f"{filename}_Extension{mm} allready exist")
            continue
        datapath = os.path.join(folder_path, file)
        try:
            datalist = pd.read_csv(datapath)
        except Exception as e:
            print(f"Error reading {datapath}: {e}")
            continue

        label_1_data = datalist[datalist['label'] == 1]
        new_rows = []

        for index, row in label_1_data.iterrows():
            grna = row['on_seq']
            dna = row['off_seq']
            print(f"{filename}---{grna}-----{dna}")
            new_rows.append({'on_seq': grna, 'off_seq': dna, 'score': cal_score(grna, dna), 'spec': cal_casoffinder_spec(grna, dna, mm), 'type': 'origin', 'desc': None})
            new_pairs = crisotex(grna, dna, mm)
            for pair in new_pairs:
                new_rows.append({'on_seq': pair[1], 'off_seq': pair[2], 'label':1 ,'score': pair[3], 'spec': pair[4], 'type': 'extension', 'desc':pair[0]})

        new_data = pd.DataFrame(new_rows, columns=['on_seq', 'off_seq', 'score', 'spec', 'type', 'desc'])
        new_data.to_csv(f'../DataSets/extension/{filename}_Extension{mm}.csv', index=False)
