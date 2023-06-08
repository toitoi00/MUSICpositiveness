# librosaをインポート
from msvcrt import kbhit
from os import P_OVERLAY
from re import X
from sre_constants import SRE_FLAG_IGNORECASE
from tokenize import Number
from unicodedata import numeric
import math
import librosa
import librosa.display
import librosa.core
import soundfile
from scipy.fftpack import fft
from scipy import signal
# numpyをインポート（配列を生成するため）
import numpy as np
# matplotlibをインポート（グラフ描画するため）
import matplotlib.pyplot as plt
#csv書き込み(確認用)
import pandas as pd
import csv
from pyfilterbank import FractionalOctaveFilterbank


def initaudio():
    # 音楽ファイルのパスを設定（例："/foldername/filename.mp3"）
    file_name = "filename"
    # loadメソッドでy=音声信号の値（audio time series）、sr=サンプリング周波数（sampling rate）を取得

    #ロード　mono:モノラルに変換するか否か offset:読み取り開始時間 duration:読み取る時間
    y, sr = librosa.load(file_name,sr=22050,mono=True)
    print(sr)
    return y, sr 

def mfcc(y,sr):
    #MFCC　n_mfcc:次元数 dct_type:離散コサイン型
    mfcc_tmp = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, dct_type=3)

    print('mfcc \n')
    mfcc=np.zeros(13)
    for i in range(13):
        mfcc[i]=np.average(mfcc_tmp[i,:])

    print(mfcc)
    pd.DataFrame(mfcc).to_csv('mfcc.csv')
    #スペクトログラムの表示(確認用)
    #librosa.display.specshow(mfcc_tmp, sr=sr,x_axis='time')
    #plt.colorbar()
    #plt.show()

def RMS(x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, S, S_1, S_2, S_3, S_4, S_5, S_6, S_7):
    rms=np.zeros(8)
    #7分割したサブバンドそれぞれのRMS計算
    rms[0] = np.average(librosa.feature.rms(y=x_1,S=librosa.amplitude_to_db(S_1, ref=1)))
    rms[1] = np.average(librosa.feature.rms(y=x_2,S=librosa.amplitude_to_db(S_2, ref=1)))
    rms[2] = np.average(librosa.feature.rms(y=x_3,S=librosa.amplitude_to_db(S_3, ref=1)))
    rms[3] = np.average(librosa.feature.rms(y=x_4,S=librosa.amplitude_to_db(S_4, ref=1)))
    rms[4] = np.average(librosa.feature.rms(y=x_5,S=librosa.amplitude_to_db(S_5, ref=1)))
    rms[5] = np.average(librosa.feature.rms(y=x_6,S=librosa.amplitude_to_db(S_6, ref=1)))
    rms[6] = np.average(librosa.feature.rms(y=x_7,S=librosa.amplitude_to_db(S_7, ref=1)))

    #スペクトル全体のRMS計算(4096点)
    rms[7] = np.average(librosa.feature.rms(y=x,S=librosa.amplitude_to_db(S, ref=1)))

    return rms

#ピーク検出
# 波形(x, y)からn個のピークを幅wで検出する関数(xは0から始まる仕様）
def findpeaks(dx, y, n, w, PoV, subnum):
    if PoV=='P':
        index_all = list(signal.argrelmax(y, order=w))                  # scipyのピーク検出
    elif PoV=='V':
        index_all = list(signal.argrelmin(y, order=w))

    index = []                                                      # ピーク指標の空リスト
    peaks = []                                                      # ピーク値の空リスト
    # n個分のピーク情報(指標、値）を格納
    for i in range(n):
        # n個のピークに満たない場合は途中でループを抜ける（エラー処理）
        if i >= len(index_all[0]):
            break

        #if PoV=='V':
         #   if y[index_all[0][i]]<=0.00001:
          #      y[index_all[0][i]]=9999
        index.append(index_all[0][i])
        peaks.append(y[index_all[0][i]])
 
    # 個数の足りない分を0で埋める（エラー処理）
    if len(index) != n:
        if PoV=='P':
            index = index + ([0] * (n - len(index)))
            peaks = peaks + ([0] * (n - len(peaks)))
        elif PoV=='V':
            index = index + ([9999] * (n - len(index)))
            peaks = peaks + ([9999] * (n - len(peaks)))

    index = (np.array(index)+subnum) * dx                                  # xの分解能dxをかけて指標を物理軸に変換
    peaks = np.array(peaks)
    return index, peaks
# スペクトログラムからピークを検出する関数
    # fft_array=スペクトログラム（転置前）
    # dt, df=スペクトログラムの時間分解能, 周波数分解能
    # num_peaks=1つの周波数軸で検出するピーク数
    # w=ノイズ対策用の幅（order）
    # max_peaks=最終的にスペクトログラムから抽出するピーク数（振幅の大きい順）
def findpeaks_2d(fft_array, dt, df, num_peaks, w, max_peaks,PoV, subnum):
    # ピーク情報を初期化する
    time_index = np.zeros((len(fft_array), num_peaks))
    freq_index = np.zeros((len(fft_array), num_peaks))
    freq_peaks = np.zeros((len(fft_array), num_peaks))
 
    # 各周波数軸毎にピークを検出する
    for i in range(len(fft_array)):
        if subnum == 1:
            index, peaks = findpeaks(df, fft_array[i], n=num_peaks, w=w, PoV=PoV, subnum=0)    # ピーク検出
        else:
            index, peaks = findpeaks(df, fft_array[i], n=num_peaks, w=w, PoV=PoV, subnum=1+16*(2**(subnum-2)))              
        freq_peaks[i] = peaks                                           # 検出したピーク値(振幅)を格納
        freq_index[i] = index                                           # 検出したピーク位置(周波数)を格納
        time_index[i] = np.full(num_peaks, i) * dt             # 検出したピーク位置(時間)を格納
 
    # 平坦化する
    freq_peaks = freq_peaks.ravel()
    freq_index = freq_index.ravel()
    time_index = time_index.ravel()
 
    # ピークの大きい順（降順）にソートする

    if PoV=='P':
        freq_peaks_sort = np.sort(freq_peaks)[::-1]
        freq_index_sort = freq_index[np.argsort(freq_peaks)[::-1]]
        time_index_sort = time_index[np.argsort(freq_peaks)[::-1]]
        print('subband %d Peak' % subnum)
    
    elif PoV=='V':
        freq_peaks_sort = np.sort(freq_peaks)
        freq_index_sort = freq_index[np.argsort(freq_peaks)]
        time_index_sort = time_index[np.argsort(freq_peaks)]
    
    #print(freq_index_sort[:max_peaks])
    print(np.average(freq_peaks_sort[:max_peaks]))
    #print(time_index_sort[:max_peaks])
    return freq_index_sort[:max_peaks], freq_peaks_sort[:max_peaks], time_index_sort[:max_peaks]

#作ったけど多分使わない
def FFT(y,sr):
    parameter = len(y)//(4096*7*2)
    frag=0
    print(len(y))
    #FFTの場合
    #ylength=len(y)   
    ylength = parameter*4096*7*2
    frq=np.linspace(0,sr,ylength)

    han=signal.hann(sr)
    plt.plot(han)
    plt.show()

    acf=1/(sum(han)/ylength)

    yf = fft(y,n=ylength)/(ylength/2)
    yfft = acf*np.abs(yf)
    print(type(yfft),frq.shape)
    #各サブバンド4096点でシフト
    #slicenp=int(len(frq)/(4096*7))
    slicefrq=frq[::parameter]
    sliceyfft=yfft[::parameter]
    #print(len(slicefrq))
    #if len(slicefrq)%2==1:
     #   slicefrq=np.append(slicefrq,0)
      #  sliceyfft=np.append(sliceyfft,0)
       # frag=1
    print(len(frq))
    #sfrq=np.split(slicefrq,2)
    #syfft=np.split(sliceyfft,2)
    sfrq=np.split(slicefrq,2)
    syfft = np.split(sliceyfft,2)
    shiftfrq=sfrq[0]
    shiftyfft=syfft[0]
    pd.DataFrame(slicefrq).to_csv('FFT.csv')
    plt.plot(shiftfrq,shiftyfft)
    plt.axis([0,sr/2,0,max(shiftyfft)])
    plt.xlabel("Freq")
    plt.ylabel("Amp")
    plt.show()
    plt.close()
    #サブバンド分割
    #if frag==1:
    #    shiftyfft=np.delete(shiftyfft,-1)
    yfft_split = np.split(shiftyfft, 7)
    #RMS計算
    rms = RMS(yfft_split,yfft)
    print('rms')
    print(rms)
    return yfft,frq

def Sslise(S):
    zerohz = np.zeros(len(S[0]))
    zerohz = S[0]
    S1 = np.delete(S,0,axis=0)
    bufband = np.split(S1,2)
    band_7 = bufband[1]
    bufbanda = np.split(bufband[0],2)
    band_6 = bufbanda[1]
    bufbandb = np.split(bufbanda[0],2)
    band_5 = bufbandb[1]
    bufbandc = np.split(bufbandb[0],2)
    band_4 = bufbandc[1]
    bufbandd = np.split(bufbandc[0],2)
    band_3 = bufbandd[1]
    bufbande = np.split(bufbandd[0],2)
    band_2 = bufbande[1]
    band = bufbande[0]
    band_1 = np.insert(band,0,zerohz,axis=0)

    return band_1, band_2, band_3, band_4, band_5, band_6, band_7

#スペクトルセントロイド
def centroid(y,sr):
    cent = librosa.feature.spectral_centroid(S=y)
    print(cent.size)
    return np.average(cent)
#スペクトルロールオフ
def rolloff(y,sr):
    rolloff = librosa.feature.spectral_rolloff(S=y,roll_percent=0.95)
    print('spectral rolloff\n{:f}'.format(np.average(rolloff)))
    #pd.DataFrame(rolloff).to_csv('rolloff.csv')
#STFT(短時間フーリエ変換)
def STFT(y,sr):
    #STFT(フレームにおける周波数ビンの大きさを返す)
    #周波数ビン:n_fft/2+1個 時刻ビン:(sr/(n_fft/4))*秒数 個
    x=librosa.stft(y,n_fft=2048)
    S = np.abs(x)
    #Sを強度と位相に分離(多分使わない)
    Strength, phase = librosa.magphase(S)
    #サブバンド分割
    x_1, x_2, x_3, x_4, x_5, x_6, x_7 = Sslise(x)
    S_1, S_2, S_3, S_4, S_5, S_6, S_7 = Sslise(S)

    #RMS計算
    rms=RMS(x, x_1, x_2, x_3, x_4, x_5, x_6, x_7, S, S_1, S_2, S_3, S_4, S_5, S_6, S_7)
    print('rms')
    print(rms)

    #データをcsvに書き出し(確認用)
    pd.DataFrame(S_1).to_csv('STFT.csv')

    #スペクトログラム出力(確認用)
    #plt.figure(figsize=(10,4))
    #y軸:対数スケール x軸:秒数
    #librosa.display.specshow(librosa.amplitude_to_db(Strength, ref=1),sr=sr,hop_length=512,y_axis='hz', x_axis='time')
    #plt.title('Power spectrogram')
    #plt.colorbar(format='%+2.0f dB')

    #plt.tight_layout()
    #plt.show()

    return S, S_1, S_2, S_3, S_4, S_5, S_6, S_7

#スペクトルフラックス
def Sflux(y,sr):
    Senvelope = librosa.onset.onset_strength(y=y,sr=sr)
    pd.DataFrame(Senvelope).to_csv('Flux.csv')
    return np.average(Senvelope)

#tonnetz（つかってない）
def tone(y,sr):
    y=librosa.effects.harmonic(y)
    tonnetz_tmp = librosa.feature.tonnetz(y=y,sr=sr)
    tonnetz=np.zeros(6)
    print(tonnetz_tmp[4,:])
    print(np.average(tonnetz_tmp[4,:]))
    for i in range(6):
        tonnetz[i]=np.average(tonnetz_tmp[i,:])
    print(tonnetz)

#クロマグラム作成
def chroma(y,sr):
    y_harm,y_perc=librosa.effects.hpss(y)
    tempo = 200
    samplesize=round(sr*60/(tempo*2))
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr,hop_length=samplesize ,n_chroma=12)
    harm(chroma)
    #fig=plt.figure(figsize=(6.4,4.8))
    #ax = fig.add_subplot()
    #librosa.display.specshow(chroma,x_axis='time',y_axis='chroma')
    #plt.colorbar()
    #plt.tight_layout()
    #plt.show()
    pd.DataFrame(chroma).to_csv('chroma.csv')

#和音テンプレートとのマッチング
def matching(chroma,templates):
    #majorminor label
    label=[1,1,1,1,1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
    scorelist=np.zeros(24)
    for k in range(24):
        scorelist[k]=np.dot(templates[k,:],chroma)
    
    i=np.argmax(scorelist)
    return(label[i])

#和音テンプレート
def harm(chroma):
    #majorminor template
    template_major = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    template_minor = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])    
    templates = np.array(  [np.roll(template_major, k) for k in range(0, 12)] \
         + [np.roll(template_minor, k) for k in range(0, 12)])

    #各音程と周波数
    #C4  = 261.626,CH4 = 277.183,D4  = 293.665,DH4 = 311.127,E4  = 329.628,F4  = 349.228,FH4 = 369.994,G4  = 391.995,GH4 = 415.305,A4  = 440,AH4 = 466.164,B4  = 493.883
    #C5  = 523.251,CH5 = 554.365,D5  = 587.330,DH5 = 622.254,E5  = 659.255,F5  = 698.456,FH5 = 739.989
    chordfreq=[261.626, 277.183, 293.665, 311.127, 329.628, 349.228, 369.994, 391.995, 415.305, 440, 466.164, 493.883, 523.251, 554.365, 587.330, 622.254, 659.255, 698.456, 739.989] 
    modal = np.zeros(chroma.shape[1])
    print(chroma.shape[1])
    for i in range(chroma.shape[1]):
        notenum=np.sort(chroma[:,i])[::-1]
        fifthnote=notenum[4]
        #スパース化
        for k in range(12):
            if chroma[k,i]<fifthnote:
                chroma[k,i]==0
        #マッチング
        chordpara=matching(chroma[:,i],templates)

        modality=chordpara*notenum[0]*notenum[1]*notenum[2]
        modal[i] = modality
    
    modalave = np.average(modal)
    pd.DataFrame(modal).to_csv('modality.csv')
    print("modality")
    print(modalave)



#読み込み
y,sr = initaudio()
time = np.arange(0,len(y)) / sr


S, S_1, S_2, S_3, S_4, S_5, S_6, S_7=STFT(y,sr)
#mfcc
mfcc(y,sr)
#tone
chroma(y,sr)
#FFTなど
#yfft,frq=FFT(y,sr)
#スペクトル重心
print('spectral centroid\n{:f}'.format(centroid(S,sr)))
#スペクトルロールオフ
rolloff(S,sr)
#スペクトルフラックス
print('spectral Flux\n{:f}'.format(Sflux(y,sr)))

df=(sr/2)/1025
dt=1/(sr/512)
#スペクトルピーク・バレー(作ったけど使わなかった)
#Pfrq_index_1, Pfrq_peaks_1, Ptime_index_1=findpeaks_2d(S_1.T,dt,df,num_peaks=501,w=50,max_peaks=500,PoV='P',subnum=1)
#Vfrq_index_1, Vfrq_peaks_1, Vtime_index_1=findpeaks_2d(S_1.T,dt,df,num_peaks=501,w=50,max_peaks=500,PoV='V',subnum=1)
#Pfrq_index_2, Pfrq_peaks_2, Ptime_index_2=findpeaks_2d(S_2.T,dt,df,num_peaks=501,w=50,max_peaks=500,PoV='P',subnum=2)
#Vfrq_index_2, Vfrq_peaks_2, Vtime_index_2=findpeaks_2d(S_2.T,dt,df,num_peaks=501,w=50,max_peaks=500,PoV='V',subnum=2)
#Pfrq_index_3, Pfrq_peaks_3, Ptime_index_3=findpeaks_2d(S_3.T,dt,df,num_peaks=501,w=50,max_peaks=500,PoV='P',subnum=3)
#Vfrq_index_3, Vfrq_peaks_3, Vtime_index_3=findpeaks_2d(S_3.T,dt,df,num_peaks=501,w=50,max_peaks=500,PoV='V',subnum=3)
#Pfrq_index_4, Pfrq_peaks_4, Ptime_index_4=findpeaks_2d(S_4.T,dt,df,num_peaks=1000,w=50,max_peaks=500,PoV='P',subnum=4)
#Vfrq_index_4, Vfrq_peaks_4, Vtime_index_4=findpeaks_2d(S_4.T,dt,df,num_peaks=1000,w=50,max_peaks=500,PoV='V',subnum=4)
#Pfrq_index_5, Pfrq_peaks_5, Ptime_index_5=findpeaks_2d(S_5.T,dt,df,num_peaks=1000,w=50,max_peaks=500,PoV='P',subnum=5)
#Vfrq_index_5, Vfrq_peaks_5, Vtime_index_5=findpeaks_2d(S_5.T,dt,df,num_peaks=1000,w=50,max_peaks=500,PoV='V',subnum=5)
#Pfrq_index_6, Pfrq_peaks_6, Ptime_index_6=findpeaks_2d(S_6.T,dt,df,num_peaks=1000,w=50,max_peaks=500,PoV='P',subnum=6)
#Vfrq_index_6, Vfrq_peaks_6, Vtime_index_6=findpeaks_2d(S_6.T,dt,df,num_peaks=1000,w=50,max_peaks=500,PoV='V',subnum=6)
#Pfrq_index_7, Pfrq_peaks_7, Ptime_index_7=findpeaks_2d(S_7.T,dt,df,num_peaks=1000,w=50,max_peaks=500,PoV='P',subnum=7)
#Vfrq_index_7, Vfrq_peaks_7, Vtime_index_7=findpeaks_2d(S_7.T,dt,df,num_peaks=1000,w=50,max_peaks=500,PoV='V',subnum=7)


# xにtime、yにyとしてプロット
#plt.plot(timeres,y_res)
#plt.plot(x)
# x軸とy軸にラベルを設定（x軸は時間(今はフレーム)、y軸は振幅）
#plt.xlabel("time(1/200[sec])")
#plt.ylabel("amplitude")
#plt.xlim(0,1000)
#plt.show()

#テンポ検出
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

#plt.tight_layout()
#plt.title("amplitude")
#plt.show()
print('Estimated tempo: {:.2f} beats per minute'.format(tempo))




