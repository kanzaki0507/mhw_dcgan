# newREAD
<p style=”position: relative;”>
    <img src=”sample_img1.jpg” width=”320″ height=”210″ alt=”写真” /><br />
    <span style=”position: absolute; top: 30px; left: 75px; width: 145px;”>
        画像の上に重ねて表示する文章です。
    </span>
</p>  

## ファイル説明  
・img: 64x64の画像dataset  
・img32: 32x32の画像dataset  
・imagechange.py: 画像サイズをresizeするプログラム  
・mhw_IB_dcgan.py: DCGANを動かすプログラム  
## Step  
#### 1. terminalで"git clone"する  
#### 2. mhw_IB_dcgan.pyを実行  

## 詳細  
・基本的にはimgファイルにある64x64の画像1920枚が読み込まれる。  
画像サイズを変更したいなら、imagechange.pyの中身を弄り、お好きなサイズにする。ただし、mhw_IB_dcgan.pyのimg_sizeとGeneratorや　Discriminatorの各レイヤーのsizeには気をつける。  
※GANはInputとOutputに気をつける。画像サイスを変更し実行にerrorが出たら、summaryを確認しながらレイヤーを修正する。