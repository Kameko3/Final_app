##推論ページ

import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np

import joblib
import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18

#画像アップロード
img = st.file_uploader("画像アップロード", type='jpg')

if img is not None:
    st.image(img, use_column_width = True)

#＜＜＜機械学習パート＞＞＞
# # 前処理用の関数
def transform(img):
    _transform = transforms.Compose([
	        transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
	])
    return _transform(img)

#学習モデル
class Net(pl.LightningModule):	

    def __init__(self):
            super().__init__()

            self.feature = resnet18(pretrained=True)
            self.fc = nn.Linear(1000, 10)

    def forward(self, x):
            h = self.feature(x)
            h = self.fc(h)
            return h

##### 推論パート#####
# モデルのインスタンス化（ネットワークの準備）
device = torch.device('cpu')
net = Net().to(device).eval()

# パラメータの読み込み
net.load_state_dict(torch.load('mobility_weight.pt',map_location = device))

# 推論、予測値の計算
img = Image.open(img)
	
x = transform(img)
x = x.unsqueeze(0)
with torch.no_grad():
	y = net(x)

 #確率値に変換
y_prob = F.softmax(y, dim=1)
proba_label = int(y_prob.max()*100)
                
# 正解ラベルを抽出
y_arg = y.argmax()
# tensor => numpy 型に変換
y_arg = y_arg.detach().numpy()

# ラベルの設定
if y_arg == 0:
	y_label = '飛行機'
if y_arg == 1:
	y_label = '自動車'
if y_arg == 8:
	y_label = '船'
if y_arg == 9:
	y_label = 'トラック'

##表示はあとでタグ見直す
st.write('これは・・・')
st.write(y_label)
st.markdown('信頼度は：')
st.write(proba_label)