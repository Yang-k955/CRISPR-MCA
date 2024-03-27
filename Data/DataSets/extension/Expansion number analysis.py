
import os
import pandas as pd

csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]

data_list = []

for csv_file in csv_files:
    data = pd.read_csv(csv_file)

    origin_count = 0
    extension_count = 0

    for _, row in data.iterrows():
        if row['type'] == 'origin':
            origin_count += 1
        elif row['type'] == 'extension':
            extension_count += 1
    data_list.append(
        {'file_name': csv_file.split('.')[0], 'origin_count': origin_count, 'extension_count': extension_count})

df = pd.DataFrame(data_list)

df.to_csv('output.csv', index=False)