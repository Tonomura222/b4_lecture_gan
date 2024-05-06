# B4輪講　GAN

- GANの学習コード
- 学習させたモデルを用いて画像を生成するコード
- 生成画像の評価を評価するコード
がある

## DCGAN.py

GANの学習を行うコード  
パスやパラメータをArgumentParserで指定

多分このコードを参考にしている  
https://colab.research.google.com/github/YutaroOgawa/pytorch_tutorials_jp/blob/main/notebook/2_Image_Video/2_4_dcgan_faces_tutorial_jp.ipynb

## save_images.py

学習したモデルを用いて画像を生成する  
モデルと保存先のパスを指定

## fid_trial.py

FIDを計算したいフォルダのパスを指定する  
celebaについては、全画像を読み込むのが無理なので  
10000枚を別のフォルダにコピーしてそれを使う
