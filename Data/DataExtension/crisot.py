import pandas as pd
import numpy as np
import os
import pickle
import time
import subprocess


bins = [0, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 1.01]
a_b = [0.0334699680573522, 0.686321918650349]
weights = np.array([0.00000e+00, 2.90000e-05, 5.00000e-05, 1.50000e-04, 1.40800e-03,
                    1.57140e-02, 3.27450e-02, 1.48352e-01, 5.02174e-01, 9.42251e-01,
                    1.00000e+00])
parampath = r"./crisot_score_param.csv"
paramread = pd.read_csv(parampath, header=0, index_col=0)

class CRISOT:
    def __init__(self, param=paramread, a_b=a_b, cutoff=None, opti_th=None, prob_weight=weights, bins=bins):
        self.feat_dict = {}
        for key in param.index:
            for i in range(20):
                self.feat_dict['Pos' + str(i + 1) + '_' + key] = param.loc[key, :].values[i]
        if a_b == None:
            a_b = [1,0]
        self.a_b = a_b
        if cutoff == None:
            cutoff = 0.
        self.cutoff = cutoff
        if opti_th == None:
            opti_th = 0.8
        self.opti_th = opti_th
        self.prob_weight = prob_weight
        self.bins = bins

        # self.proj_t = time.time()
        self.proj_t = 'test'

    def single_score_(self, on_seq, off_seq):
        y_pred = np.array([self.feat_dict['Pos' + str(j + 1) + '_' + on_seq[j] + off_seq[j]] for j in range(20)]).sum()
        y_pred = y_pred * self.a_b[0] + self.a_b[1]
        if y_pred > 1:
            y_pred = 1.
        elif y_pred < 0:
            y_pred = 0.
        return y_pred

    def score(self, data_path=None, data_df=None, On='On', Off='Off', Active=None):
        if data_df is not None:
            data_set = data_df
        else:
            data_set = pd.read_csv(data_path, sep=",", header=0, index_col=None)
        ont = data_set.loc[:, On].values
        offt = data_set.loc[:, Off].values
        y_pred = np.array([self.single_score_(ont[i], offt[i]) for i in range(ont.shape[0])])
        if Active == None:
            return y_pred
        else:
            y_ori = data_set.loc[:, Active].values
            return y_ori, y_pred

    def score_bin_(self, y_pred):
        y_pred = np.array(y_pred)
        if y_pred.shape[0] != 0:
            y_df = pd.DataFrame(y_pred.reshape(-1,1), columns=['CRISOT-Score'])
            y_count = y_df['CRISOT-Score'].value_counts(bins=self.bins, sort=False).values
        else:
            y_count = np.array([0] * (len(self.bins)-1))
        return y_count

    def single_aggre_(self, y_pred, out_cnt=True):

        cnt = self.score_bin_(y_pred)
        aggre = (cnt * self.prob_weight).sum()
        if out_cnt:
            return np.append(cnt[-4:], aggre)
        else:
            return aggre

    def single_spec_(self, y_pred):
        aggre = self.single_aggre_(y_pred, out_cnt=False)
        spec = 10 / (10 + aggre)
        return spec

    def spec(self, data_path: object = None, data_df: object = None, On: object = 'On', Off: object = 'Off', target: object = None, out_df: object = False) -> object:

        if data_df is not None:
            data_set = data_df
        else:
            data_set = pd.read_csv(data_path, sep=",", header=0, index_col=None)
        offt = data_set.loc[:, Off]

        if target is not None:
            assert len(target) == 23, 'target sequence must have 23 nt'
            assert data_set.loc[offt == target, :].shape[0] > 0, 'No sequence match the target sequence'
            data_set = pd.concat([data_set.loc[offt == target, :], data_set.loc[offt != target, :]])
        y_pred = self.score(data_df=data_set, On=On, Off=Off, Active=None)
        spec = self.single_spec_(y_pred[1:])
        if out_df:
            data_set['CRISOT-Score'] = y_pred
            return spec, data_set
        else:
            return spec

    def CasoffinderSpec_(self, sgrna, target, out_df=False, mm=6, dev='G0'):
        self.ref_genome = r'./genome/hg38.2bit'
        log_dir = f"./casfile{mm}"
        os.makedirs(log_dir, exist_ok=True)

        if not os.path.exists(f'./casfile{mm}/temp_{sgrna}_casoffinder.out'):
            print(f"Creating {sgrna} outfile")
            with open(f'temp_{self.proj_t}_casoffinder.in', 'w') as file:
                file.write(f'{self.ref_genome}\n')
                file.write(f'{"N"*21}GG\n')
                file.write(f'{sgrna[:20]}NNN {mm}\n')
            os.system(f"cas-offinder temp_{self.proj_t}_casoffinder.in {dev} ./casfile{mm}/temp_{sgrna}_casoffinder.out")
            os.remove(f'temp_{self.proj_t}_casoffinder.in')

        data_set = pd.read_csv(f'./casfile{mm}/temp_{sgrna}_casoffinder.out', sep="\t", header=None, index_col=None)
        print(f'./casfile{mm}/temp_{sgrna}_casoffinder.out')
   
        offt = data_set.loc[:, 3].values
        offt = np.array([str.upper(t) for t in offt])
        data_set.loc[:, 3] = offt
        data_set = data_set[-data_set[3].str.contains('N|R|W|M|V|Y|K|D|S|J')]
        data_set.drop_duplicates([1,2,3], inplace=True)
        if data_set[data_set[3].str.contains(target[:20])].shape[0] == 0:
            data_set = data_set.append(pd.Series([data_set.iloc[0, 0], np.nan, np.nan, target, np.nan, np.nan]), ignore_index=True)
        data_set = pd.concat(
            [data_set[data_set[3].str.contains(target[:20])], data_set[-data_set[3].str.contains(target[:20])]])
        if out_df:
            spec, out_dset = self.spec(data_df=data_set, On=0, Off=3, out_df=out_df)
        else:
            spec = self.spec(data_df=data_set, On=0, Off=3)

        if out_df:
            return spec, out_dset
        else:
            return spec