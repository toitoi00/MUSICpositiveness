from re import X
import matplotlib.pyplot as plt
import csv
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn.decomposition import PCA
import skfuzzy as fuzz

#ファジィクラスタリング

filename = '特徴量.csv'
filename_RMS = '特徴量_RMS.csv'
ncenters=2
parameter=4.0

dataset = pd.read_csv(filename,index_col=0)
dataset_RMS = pd.read_csv(filename_RMS,index_col=0)

pca = PCA(n_components=3)  # PCA を行ったり PCA の結果を格納したりするための変数を、pca として宣言
pca.fit_transform(dataset)  # PCA を実行

score = pd.DataFrame(pca.transform(dataset), index=dataset.index)

fig1, ax1 = plt.subplots()
ax1.set_title('fuzzyclustering')
center, u, uo, d, jm, p, fpc = fuzz.cmeans(score.T,ncenters,parameter,error=0.005, maxiter=1000, init=None)

pd.DataFrame(u).to_csv('fuzzy.csv')