from random import choices
from secrets import choice
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np

import joblib
import torch

import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18

#セキュリティ
import hashlib
def make_hashes(password):
	return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
	if make_hashes(password) == hashed_text:
		return hashed_text
	return False

# DB管理
import sqlite3 
conn = sqlite3.connect('data.db')
c = conn.cursor()

# DB機能
def create_usertable():
	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')

def add_userdata(username,password):
	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
	conn.commit()

def login_user(username,password):
	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
	data = c.fetchall()
	return data

def view_all_users():
	c.execute('SELECT * FROM userstable')
	data = c.fetchall()
	return data

# 機械学習
# 前処理用の関数
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

#メイン画面
def main():
	"""動物ずかん"""

	st.title("動物ずかん")

	choice = st.sidebar.selectbox(
			"Menu",
			["Home","Login","SignUp"],
	)	

	#ホーム
	if choice == "Home":
		st.subheader("🐶🐱🐴🐦🦌🐸")
		st.subheader("犬・猫・馬・鳥・鹿・蛙の図鑑です。")

	#サインアップ
	elif choice == "SignUp":
		st.subheader("Create New Account")
		new_user = st.text_input("Username")
		new_password = st.text_input("Password",type='password')

		if st.button("Signup"):
			create_usertable()
			add_userdata(new_user,make_hashes(new_password))
			st.success("You have successfully created a valid Account")
			st.info("Go to Login Menu to login")

	#ログイン
	elif choice == "Login":
		st.subheader("Login Section")
		username = st.sidebar.text_input("User Name")
		password = st.sidebar.text_input("Password",type='password')

		if st.sidebar.button("Login"):
			create_usertable()
			hashed_pswd = make_hashes(password)
			result = login_user(username,check_hashes(password,hashed_pswd))	

			if result: #ログイン成功　
				st.success("Logged In as {}".format(username))
				#画像アップロード 
				img = st.file_uploader("画像アップロード", type='jpg')
				
				#画像表示
				if img is not None:
					st.image(img, use_column_width = True) 	#画像サイズを画面サイズに合わせて調整

					# 推論
					# モデルのインスタンス化（ネットワークの準備）
					device = torch.device('cpu')
					net = Net().to(device).eval()

					# パラメータの読み込み
					net.load_state_dict(torch.load('mobility_weight.pt',map_location = device))

					# 推論、予測値の計算
					pred = Image.open(img)
						
					x = transform(pred)
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
						y_label = '動物ではない'
					if y_arg == 1:
						y_label = '動物ではない'
					if y_arg == 2:
						y_label = '鳥'
					if y_arg == 3:
						y_label = '猫'
					if y_arg == 4:
						y_label = '鹿'		
					if y_arg == 5:
						y_label = '犬'
					if y_arg == 6:
						y_label = '蛙'		
					if y_arg == 7:
						y_label = '馬'		
					if y_arg == 8:
						y_label = '動物ではない'
					if y_arg == 9:
						y_label = '動物ではない'		

					##表示はあとでタグ見直す
					st.subheader('これは')
					st.write(y_label)
					st.subheader('です。')

					# """
					# #特徴メモ
					# """
					
					# # st.subheader('信頼度は')
					# # st.write(proba_label)
					# # st.subheader('%です。')

			else:
				st.warning("Incorrect Username/Password")	

if __name__ == '__main__':
	main()