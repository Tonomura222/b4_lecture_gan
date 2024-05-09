# B4輪講　GAN
使うコードは3つ  
- GANの学習コード
- 学習させたモデルを用いて画像を生成するコード
- 生成画像を評価するコード

## GANの学習
DCGAN.pyを実行する  
```
python3 DCGAN.py
```
パスやパラメータを指定する
モデルや損失のグラフ、画像等が保存される

## 画像の生成
save_images.pyを実行する
```
python3 save_images.py
```

## 評価

```
python3 fid_trial.py
```
celebaの場合全画像を読み込むのはメモリ容量が厳しいので別フォルダにコピーしている。

