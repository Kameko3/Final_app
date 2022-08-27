from random import choices
from secrets import choice
from turtle import left, right
import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np

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

		if st.sidebar.checkbox("Login"):
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
						memo = '他の写真を読み込んでみてください'		
					if y_arg == 1:
						y_label = '動物ではない'
						memo = '他の写真を読み込んでみてください'	
					if y_arg == 2:
						y_label = '鳥'
						memo = '飛行を得意とした動物。体表が羽毛で覆われた恒温動物で、歯はなく、前肢が翼になって、飛翔のための適応が顕著であり、二足歩行を行う。現存する鳥類は約1万種と言われている。'	
						memo2 = '出展:wikipedia'
					if y_arg == 3:
						y_label = '猫'
						memo = '非常に優れた平衡感覚に、柔軟性と瞬発力のきわめて高い体の構造、武器である鋭い鉤爪（かぎづめ）や牙を持ち、足音が非常に小さく、体臭が少ない。いわゆる猫(ペットとして飼われる猫)の起源は、ネズミを捕獲させる目的で飼われ始めたリビアヤマネコの家畜化である'	
						memo2 = '出展:wikipedia'
					if y_arg == 4:
						y_label = '鹿'
						memo = 'シカ科 (Cervidae) に属する哺乳類の総称である。約16属36種が世界中の森林などに生息している。オスは枝分かれしたツノを持ち、枝角（アントラー）と呼ばれる。多くのシカ科のメスはツノを持たないがトナカイはオスメス共にツノを持つ。オスは枝分かれしたツノを持ち、枝角（アントラー）と呼ばれる。多くのシカ科のメスはツノを持たないがトナカイはオスメス共にツノを持つ。'		
						memo2 = '出展:wikipedia'
					if y_arg == 5:
						y_label = '犬'
						memo = '食肉目・イヌ科・イヌ属に分類される哺乳類の一種。イエイヌは人間の手によって作り出された動物群である。ジャパンケネルクラブ(JKC)では、国際畜犬連盟(FCI)が公認する331犬種を公認し、そのうち176犬種を登録してスタンダードを定めている。 なお、非公認犬種を含めると約700 - 800の犬種がいるとされている。 最も古くに家畜化されたと考えられる動物であり、現代でも、イエネコと並んで代表的なペットまたはコンパニオンアニマルとして、広く飼育され、親しまれている。ただし比較されるネコと違って独特の口臭がある。'	
						memo2 = '出展:wikipedia'
					if y_arg == 6:
						y_label = '蛙'	
						memo = '両生綱無尾目（むびもく）に分類される構成種の総称。成体の頭は三角形で、目は上に飛び出している。6,579種(日本には5科42種のカエルが生息している)ほど知られており、そのほとんどが水辺で暮らしている。'	
						memo2 = '出展:wikipedia'
					if y_arg == 7:
						y_label = '馬'		
						memo = '哺乳綱奇蹄目ウマ科ウマ属に分類される家畜動物。体長は2.4〜3m程度。体重は300〜800kg程度。寿命は約25年、稀に40年を超えることもある。'	
						memo2 = '出展:wikipedia'
					if y_arg == 8:
						y_label = '動物ではない'
						memo = '他の写真を読み込んでみてください'	
					if y_arg == 9:
						y_label = '動物ではない'	
						memo = '他の写真を読み込んでみてください'		

					#結果表示
					left_column, right_column = st.columns(2)
					left_column.write('動物名：')
					right_column.subheader(y_label)

					left_column.write('特徴：')
					st.write(memo)
					st.write(memo2)
					
					# # st.subheader('信頼度は')
					# # st.write(proba_label)
					# # st.subheader('%です。')

			else:
				st.warning("Incorrect Username/Password")	

if __name__ == '__main__':
	main()