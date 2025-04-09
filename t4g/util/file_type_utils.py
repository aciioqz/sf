import os
import pandas as pd


def convert_dat_to_csv(dir:str,  encoding='ISO-8859-1'):
    listdir = os.listdir(dir)
    for filename in listdir:
        if filename.endswith('dat'):
            dat_path = os.path.join(dir, filename)
            csv_pah = os.path.join(dir, filename.replace('.dat', '.csv'))
            df = pd.read_csv(dat_path, sep='::', engine='python', encoding=encoding)
            df.to_csv(csv_pah, index=False)

if __name__ == '__main__':
    convert_dat_to_csv()