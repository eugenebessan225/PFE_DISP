#!/usr/bin/env python
# coding: utf-8


import os
import re

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import  GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix

from scipy.fft import fft
from scipy.stats import skew, kurtosis
import statistics as stc

import matplotlib.pyplot as plt
import seaborn as sns


# # Lecture des fichiers


# directory = "Experimentations"
# df_names = []
# for root, directories, files in os.walk(directory):
#     for file_name in files:
#         if file_name.endswith(".xlsx"):
#             file_path = os.path.join(root, file_name)
#             var_add = file_path.split('\\')
#             data_frames = []
#             excel_file = pd.ExcelFile(file_path)
#             sheets = excel_file.sheet_names[:2]
#             for i in range(len(sheets)-1, -1, -1):
#                 data = excel_file.parse(sheets[i])
#                 data = data.drop(data.index[0])
#                 data_frames.append(data)
#                 df = pd.concat(data_frames, ignore_index=True)
#                 for elt in var_add[1:]:
#                     if elt != file_name:
#                         df[elt] = 1  
#                     else: 
#                         match = re.findall(r'(\d+%)', file_name)
#                         try:
#                             df["vitesse_a"] = match[0]
#                             df["vitesse_r"] = match[1]
#                         except:
#                             pass
#             df_name = "data_"+"_".join(var_add[1:]).replace(" ", "_").replace("'", "_")
#             df_names.append(df_name)
#             globals()[df_name] = df




acier_usure_75_75 = "Experimentations/40_2022_01_12_134028_75%_75%_L_US.xlsx"
acier_simp_75_75 = "Experimentations/19_2022_01_10_155909_75%_75%_L.xlsx"

acier_usure_120_100 = "Experimentations/39_2022_01_12_132408_120%_100%_L_US.xlsx"
acier_simp_120_100 = "Experimentations/13_2022_01_10_135525_120%_100%_L.xlsx"



def load_df(link):
    data_frames = []
    excel_file = pd.ExcelFile(link)
    sheets = excel_file.sheet_names[:2]
    for i in range(len(sheets)-1, -1, -1):
        data = excel_file.parse(sheets[i])
        data = data.drop(data.index[0])
        data_frames.append(data)
        merged_data = pd.concat(data_frames, ignore_index=True)
        merged_data["Time"] = merged_data["Time"].astype(float)
        merged_data["acc_broche"] = merged_data["acc_broche"].astype(float)
        merged_data["acc_table"] = merged_data["acc_table"].astype(float)
    return merged_data



df_us_75 = load_df(acier_usure_75_75) #df d'expé sur de l'acier usé à 75%, 75% et avec lubrifiant

df_75 = load_df(acier_simp_75_75) #df d'expé sur de l'acier à 75%, 75% et avec lubrifiant

df_us_120 = load_df(acier_usure_120_100)

df_120 = load_df(acier_simp_120_100)

df_us_75.info()


def get4seq(df, lower_bound, upper_bound, train_end, test_start):
    df = df[df["Time"].between(lower_bound, upper_bound)]
    X_train = df[df["Time"].between(lower_bound, train_end)]
    X_test = df[df["Time"].between(test_start, upper_bound)]
    return df, X_train, X_test


def plot_data(df, X_train, X_test):
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 6))
    ax1.plot(df['Time'], df[['acc_broche', 'acc_table']])
    ax2.plot(X_train['Time'], X_train[['acc_broche', 'acc_table']])
    ax2.set_title('Train set')
    ax3.plot(X_test['Time'], X_test[['acc_broche', 'acc_table']])
    ax3.set_title('Test set')
    plt.show()


lower_bound = 687.0000
upper_bound = 822.5000
df_us_75, X1_train, X1_test = get4seq(df_us_75, lower_bound, upper_bound, 789.5000, 801.0000)


plot_data(df_us_75, X1_train, X1_test)


lower_bound = 691.0000
upper_bound = 827.5000
df_75, X2_train, X2_test = get4seq(df_75, lower_bound, upper_bound, 792.5000, 807.5000)



plot_data(df_75, X2_train, X2_test)


lower_bound = 449.0000
upper_bound = 534.5000
df_us_120, X3_train, X3_test = get4seq(df_us_120, lower_bound, upper_bound, 512.5000, 521.5000)


plot_data(df_us_120, X3_train, X3_test)


lower_bound = 438.0000
upper_bound = 524.5000
df_120, X4_train, X4_test = get4seq(df_120, lower_bound, upper_bound, 503.5000, 512.0000)


plot_data(df_120, X4_train, X4_test)


train = {1:[X1_train, X3_train], 0:[X2_train, X4_train]}
test = {1:[X1_test, X3_test], 0:[X2_test, X4_test]}


def stat(groupe):
    stats = groupe.agg(['std', 'median', 'max', 'min'])
    # Transposer le DataFrame et aplatir les colonnes
    df_flat = stats.stack()
    df_flat = df_flat.T.reset_index()
    df_flat["column"] = df_flat["level_0"]+"_"+df_flat["level_1"]
    df_flat.drop(["level_0", "level_1"], inplace=True, axis=1)
    df_flat["values"] = df_flat[0]
    df_flat.drop(df_flat.columns[0], axis=1, inplace=True)
    df_stats = df_flat.set_index("column")
    df_stats = df_stats.transpose()
    return df_stats

def car(df, w_size):
    
    """Fonction qui retourne un df de caractéristiques 
    telles que le ratio d'energie, 
    le crete à crete, l'écart type, le rms et le crest fatcor"""

    j=0
    df_new = pd.DataFrame()
    df_stats = pd.DataFrame()
    ## Calcul de l'énergie totale pour deduire le ratio d'energie
    acc_table = df['acc_table'].tolist()
    acc_broche = df['acc_broche'].tolist()
    fft_table = fft(acc_table)
    fft_broche = fft(acc_broche)
    tot_energy_table = np.sum(np.abs(fft_table)**2)
    tot_energy_broche = np.sum(np.abs(fft_broche)**2)
    for i in range(0, len(df), w_size):
        df_car = {}
        groupe = df.iloc[i:i+w_size]
        df_stat = stat(groupe) # Calcul des stats (median, std, min, max)
        ## Energy ratio
        # Calcul de la transformée de fourier
        fft_table = fft(groupe['acc_table'].tolist())
        fft_broche = fft(groupe['acc_broche'].to_list())
        # Calcul de l'energie totale par groupe
        energy_table = np.sum(np.abs(fft_table))
        energy_broche = np.sum(np.abs(fft_broche))
        # Calcul du ratio d'énergie
        energy_ratio_table = energy_table / tot_energy_table
        df_car['energy_table'] = round(energy_table, 4)
        df_car['energy_broche'] = round(energy_broche, 4)
        ## Peak to Peak
        peak_to_peak_table = round(max(groupe['acc_table']) - min(groupe['acc_table']), 4) 
        peak_to_peak_broche = round(max(groupe['acc_broche']) - min(groupe['acc_broche']), 4)
        df_car['peak_to_peak_table'] = peak_to_peak_table
        df_car['peak_to_peak_broche'] = peak_to_peak_broche
        ## standard deviation
        std_deviation_table = np.std(groupe['acc_table'].tolist())
        std_deviation_broche = np.std(groupe['acc_broche'].tolist())
        df_car['std_deviation_table'] = round(std_deviation_table, 4)
        df_car['std_deviation_broche'] = round(std_deviation_broche, 4)
        ## Root mean Square 
        rms_table = np.sqrt(np.mean(np.square(groupe['acc_table'].tolist())))
        rms_broche = np.sqrt(np.mean(np.square(groupe['acc_broche'].tolist())))
        df_car['rms_table'] = round(rms_table, 4)
        df_car['rms_broche'] = round(rms_broche, 4)
        ## Crest factor
        peak_value_table = np.max(groupe["acc_table"].tolist())
        peak_value_broche = np.max(groupe["acc_broche"].tolist())
        crest_factor_table = peak_value_table / rms_table
        crest_factor_broche = peak_value_broche / rms_broche
        df_car['crest_factor_table'] = round(crest_factor_table, 4)
        df_car['crest_factor_broche'] = round(crest_factor_broche, 4)
        ## skewness (asymétrie)
        skew_table = skew(groupe['acc_table'])
        skew_broche = skew(groupe['acc_broche'])
        df_car['skew_table'] = round(skew_table, 4)
        df_car['skew_broche'] = round(skew_broche, 4)
        ## Kurtosis (Aplatissement)
        kurtosis_table = kurtosis(groupe['acc_table'])
        kurtosis_broche = kurtosis(groupe['acc_broche'])
        df_car['kurtosis_table'] = round(kurtosis_table, 4)
        df_car['kurtosis_broche'] = round(kurtosis_broche, 4)
        df_car = pd.DataFrame(df_car, index=[j])
        j+=1
        #display(df_car)
        df_new = pd.concat([df_new, df_car], axis=0)
        df_stats = pd.concat([df_stats, df_stat], axis=0)
        df_stats = df_stats.reset_index(drop=True)
    X = pd.concat([df_new, df_stats], axis=1)
    return X


def create_train_test(train, w_size):
    X1  = pd.DataFrame()
    X0  = pd.DataFrame()
    for y, x in train.items():
        if y==1:
            for d in x:
                data = car(d, w_size)
                data = data.dropna()
                X1 = pd.concat([X1, data], axis=0)
        elif y==0:
            for d in x:
                data = car(d, w_size)
                data = data.dropna()
                X0 = pd.concat([X0, data], axis=0)
    Y1 = np.ones(X1.shape[0])
    Y0 = np.zeros(X0.shape[0])
    print(f'{Y1.shape} class 1 values')
    print(f'{Y0.shape} class 0 values')
    X = pd.concat([X1, X0])
    Y = np.concatenate([Y1, Y0])
    return X, Y


class Model():
    def __init__(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        name,
        model
    ):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.name = name
        self.model = model
        self.accuracy = 0 
        self.acc_stdev = 0
        self.precision = 0
        self.prec_stdev =0
        self.f1 = 0
        self.f_stdev = 0
        
    def run(self):
        acc = []
        prec = []
        f = []
        print(f"<==========================Training started for {self.name} ==========================>")
        for i in range(1, 10):
            print(f"***************** epoch {i} ********************")
            self.model.fit(self.X_train, self.y_train)
            y_pred = self.model.predict(self.X_test)
            accu = accuracy_score(self.y_test, y_pred)
            print(f'accuracy = {accu}')
            acc.append(accu)
            preci = precision_score(self.y_test, y_pred)
            prec.append(preci)
            print(f'precision = {preci}')
            f1 = f1_score(self.y_test, y_pred)
            print(f'f1 = {f1}')
            f.append(f1)
        self.accuracy = stc.mean(acc)
        self.acc_stdev = stc.stdev(acc)
        self.precision = stc.mean(prec)
        self.prec_stdev = stc.stdev(prec)
        self.f1 = stc.mean(f)
        self.f_stdev = stc.stdev(f)
        return {"accuracy":self.accuracy, "acc_std": self.acc_stdev, "precision": self.precision, "prec_std": self.prec_stdev, "f1" :self.f1, "f_std": self.f_stdev}


# ## Analyse des 4 dernières séquences avec les données brutes


#Modele de machine learning simple
knn_model = KNeighborsClassifier()
rdf_model = RandomForestClassifier()
lgr_model = LogisticRegression()
xgb_model = XGBClassifier()
mlp_model = MLPClassifier()

## Definition of grids
knn_grid = {
    'n_neighbors': [i for i in range(3,33,2)],
    'p':[1, 2],
    'weights' : ["uniform", "distance"]
}

rdf_grid = {
    'n_estimators': [i for i in range(50,550,50)],
    'max_features' : ["sqrt", "log2", None]
}

lgr_grid = {
    'penalty':['l1', 'l2', 'elasticnet'],
    'solver': ['liblinear', 'saga'],
    'C': [0.1, 1, 10]
}

xgb_grid = {
    'learning_rate' : [0.1, 0.01, 0.001],
    'max_depth' : [i for i in range(3,10,1)],
    'n_estimators': [i for i in range(50,550,50)]
}

mlp_grid = {
    'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100), (100,50,20), (100,100,50), (100,100,100), (200,100,50), (200,100,50, 20)],
    'activation': ['relu', 'tanh', 'identity', 'logistic'],
    'solver': ['adam', 'lbfgs', 'sgd']
}



gs_knn = GridSearchCV(knn_model, knn_grid, cv=5)
gs_rdf = GridSearchCV(rdf_model, rdf_grid, cv=5)
gs_lgr = GridSearchCV(lgr_model, lgr_grid, cv=5)
gs_xgb = GridSearchCV(xgb_model, xgb_grid, cv=5)
gs_mlp = GridSearchCV(mlp_model, mlp_grid, cv=5)


models_train = {
    "knn": gs_knn,
    "rdf": gs_rdf,
    "lgr": gs_logr,
    "xgb":gs_xgb,
    "mlp":gs_mlp
}

models = {
    
    "knn": KNeighborsClassifier,
    "rdf": RandomForestClassifier,
    "lgr": LogisticRegression,
    "xgb": XGBClassifier,
    "mlp": MLPClassifier
}



w_sizes = [2000]

resl = {}
for w_size in w_sizes: 
    accuracy = []
    acc_std = []
    precision = []
    prec_std = []
    f1 = []
    f_std = []
    turn = {}
    X_train, y_train = create_train_test(train, w_size)   
    X_test, y_test = create_train_test(test, w_size) 
    for n, m in models_train.items():
        m.fit(X_train, y_train)
        print(f" Best Params for {n} = {m.best_params_}")
        for i, j in models.items():
            if i == n:
                m = j(**m.best_params_)
        model = Model(X_train, X_test, y_train, y_test, n, m)
        scores = model.run()
        accuracy.append(scores["accuracy"])
        acc_std.append(scores["acc_std"])
        precision.append(scores["precision"])
        prec_std.append(scores["prec_std"])
        f1.append(scores["f1"])
        f_std.append(scores["f_std"])
    turn["acc"] = accuracy
    turn["acc_std"] = acc_std
    turn["prec"] = precision
    turn["prec_std"] = prec_std
    turn["f1"] = f1
    turn["f_std"] = f_std
    resl[w_size] = turn
    print(f'End for {w_size}')
    

names = ["KNN", "RDF", "LGR", "XGB", "MLP"]
bar_colors = ['#FF0000', '#0000FF', '#00FF00', '#FFA500', '#800080']
for k, v in resl.items():
    print(f"plot for {k}")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 6))
    ax1.bar(names, v["acc"], yerr=v["acc_std"], align="center", alpha=0.5, ecolor="black", color=bar_colors)
    ax1.set_title("accuracy")
    ax2.bar( names, v["prec"], yerr=v["prec_std"], align="center", alpha=0.5, ecolor="black", color=bar_colors)
    ax2.set_title("Precision")
    ax3.bar(names, v["f1"], yerr=v["f_std"], align="center", alpha=0.5, ecolor="black", color=bar_colors)
    ax3.set_title("F1")
    plt.show()

for i in range(len(names)):
    a, p, f, t = [], [], [], []
    for k, v in resl.items():
        a.append(v["acc"][i])
        p.append(v["prec"][i])
        f.append(v["f"][i])
        t.append(k)
    print(f'Metrics evolutions for {names[i]}')    
    t = [str(j) for j in t]
    fig, (aa, ap, af) = plt.subplots(3, 1, figsize=(10, 8))
    aa.plot(t, a)
    aa.set_title('Accuracy Evolution')
    ap.plot(t, p)
    ap.set_title('Precision Evolution')
    af.plot(t, f)
    af.set_title('F1 Evolution')
    plt.show()



LogisticRegression().get_params().keys()



