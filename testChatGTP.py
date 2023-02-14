#ランダムに取得したインターネット上の画像から人間の顔を検出した場合赤線で囲って表示する

#-*- coding: utf-8 -*-

from PIL import Image
import requests
import io
import numpy as np
import cv2
import random




def main():

	#while文のカウンター
	counter = 0

	#顔の検出を試みる回数の上限値
	test_maximum = 30


	while counter < test_maximum:

		counter += 1

		#画像をランダムに選ぶURL
		img_url = "https://picsum.photos/300/300?random={}.format(random.randomint(0, 10000))" 

		#画像をダウンロード
		img_url_response = requests.get(img_url)
		img = Image.open(io.BytesIO(img_url_response.content))
		
		#画像の形式をnumpy_arrayに変換
		numpy_array_img = np.array(img)

		#特微分類器の読み込み
		face_cascade = cv2.CascadeClassifier ('./haarcascade_frontalface_default.xml')

		#計算を高速にするため、グレースケールに変換した画像を用意する
		gray = cv2.cvtColor(numpy_array_img, cv2.COLOR_BGR2GRAY)

		#検出した顔のリスト
		face = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30))

		#顔を検出できなかった場合はループを繰り返す
		if len(face) == 0:
			continue
		else:
			break

	#カウンターが上限値に達していた場合、顔を検出できなかった旨を表示する
	if counter == test_maximum:
		print("face could't ditected")
	
	else:
		#カウンターを表示する
		print('counter = ', counter)
			
		#顔領域を赤色の矩形で囲む
		for (x, y, w, h) in face:
			cv2.rectangle(numpy_array_img, (x, y), (x + w, y+h), (0,0,300), 4)

		#結果を出力
		cv2.imwrite('result.jpg', numpy_array_img)
		
		result_img = cv2.imread('./result.jpg')
		
		cv2.imshow('face_detect', result_img)
		
		#表示
		cv2.waitKey(10)
		cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
