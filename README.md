# MUSICpositiveness
卒業研究で使用した楽曲からポジティブ度を算出するのに使用したコードです．
msfeatureでは楽曲のオーディオ信号を入力にRMS，スペクトル重心，スペクトル変動，スペクトルロールオフ，BPM，調性，MFCC等の特徴量を出力しています．
fuzzyは楽曲毎の特徴量をファジィクラスタリングによってクラスタリングするコード，saitekikaは実験結果をもとにポジティブ度算出に関わるパラメータを出力するコードです．
