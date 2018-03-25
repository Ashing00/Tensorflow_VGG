import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import vgg_inference
import vgg_train
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import cv2
from matplotlib import pyplot as plt

img_num=[0]*20


def display_result(my_prediction,My_Yd):	
	img_res=[0]*20
	font = cv2.FONT_HERSHEY_SIMPLEX
	for i in range(20):	 
		img_res[i] = np.zeros((64,64,3), np.uint8)
		img_res[i][:,:]=[255,255,255]
		if (my_prediction[i]%10)==(My_Yd[i]%10):
			cv2.putText(img_res[i],str(my_prediction[i]),(15,52), font, 2,(0,255,0),3,cv2.LINE_AA)
		else:
			cv2.putText(img_res[i],str(my_prediction[i]),(15,52), font, 2,(255,0,0),3,cv2.LINE_AA)

	Input_Numer_name = ['Input 0', 'Input 1','Input 2', 'Input 3','Input 4',\
					'Input 5','Input 6', 'Input 7','Input8', 'Input9',\
					'Input 0', 'Input 1','Input 2', 'Input 3','Input 4',\
					'Input 5','Input 6', 'Input 7','Input8', 'Input9',
					]
					
	predict_Numer_name =['predict 0', 'predict 1','predict 2', 'predict 3','predict 4', \
					'predict 5','predict6 ', 'predict 7','predict 8', 'predict 9',\
					'predict 0', 'predict 1','predict 2', 'predict 3','predict 4', \
					'predict 5','predict6 ', 'predict 7','predict 8', 'predict 9',
					]
				
	for i in range(20):
		if i<10:
			plt.subplot(4,10,i+1),plt.imshow(img_num[i],cmap = 'gray')
			plt.title(Input_Numer_name[0]), plt.xticks([]), plt.yticks([])
			plt.subplot(4,10,i+11),plt.imshow(img_res[i],cmap = 'gray')
			plt.title(predict_Numer_name[i]), plt.xticks([]), plt.yticks([])
		else:
			plt.subplot(4,10,i+11),plt.imshow(img_num[i],cmap = 'gray')
			plt.title(Input_Numer_name[0]), plt.xticks([]), plt.yticks([])
			plt.subplot(4,10,i+21),plt.imshow(img_res[i],cmap = 'gray')
			plt.title(predict_Numer_name[i]), plt.xticks([]), plt.yticks([])
		
	plt.show()	
def encode_labels( y, k):
	"""Encode labels into one-hot representation
	"""
	onehot = np.zeros((y.shape[0],k ))
	for idx, val in enumerate(y):
		onehot[idx,val] = 1.0  ##idx=0~xxxxx，if val =3 ,表示欄位3要設成1.0
	return onehot

def evaluate(X_test,y_test_lable,My_Yd):
	with tf.Graph().as_default() as g:
	
		# 定義輸出為4維矩陣的placeholder
		x_ = tf.placeholder(tf.float32, [None, vgg_train.n_input])	
		x = tf.reshape(x_, shape=[-1, 32, 32, 3])
		y = tf.placeholder(tf.float32, [None, vgg_train.n_classes])
	
		# Construct model
		pred = vgg_inference.inference(x, 1)     #dropout=1

		# Evaluate model
		pred_max=tf.argmax(pred,1)
		y_max=tf.argmax(y,1)
		correct_pred = tf.equal(pred_max,y_max)
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	
		#test_batch_len =int( X_test.shape[0]/vgg_train.BATCH_SIZE)
		test_acc=[]
		
		test_xs = np.reshape(X_test, (
					X_test.shape[0],
					32,
					32,
					3))
		
		batchsize=vgg_train.BATCH_SIZE
	
		# 'Saver' op to save and restore all the variables
		saver = tf.train.Saver()
		with tf.Session() as sess:
			saver.restore(sess,"./vgg/vgg_cifar_model")

			My_test_pred=sess.run(pred_max, feed_dict={x: test_xs})
			print("期望值：",My_Yd)
			print("預測值：",My_test_pred)
			My_acc = sess.run(accuracy, feed_dict={x: test_xs, y: y_test_lable})
			print('Test accuracy: %.2f%%' % (My_acc * 100))
			display_result(My_test_pred,My_Yd)
			return
		
def main(argv=None):
	#### Loading the data
	#預計輸入20張圖片
	My_X =np.zeros((20,3072), dtype=int) 
	#期望數字
	My_Yd =np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=int) 

	#輸入20張32x32x3=3072 pixel，
	Input_Numer=[0]*20
	Input_Numer[0]="a_0.jpg"
	Input_Numer[1]="a_1.jpg"
	Input_Numer[2]="a_2.jpg"
	Input_Numer[3]="a_3.jpg"
	Input_Numer[4]="a_4.jpg"
	Input_Numer[5]="a_5.jpg"
	Input_Numer[6]="a_6.jpg"
	Input_Numer[7]="a_7.jpg"
	Input_Numer[8]="a_8.jpg"
	Input_Numer[9]="a_9.jpg"
	Input_Numer[10]="a_10.jpg"
	Input_Numer[11]="a_11.jpg"
	Input_Numer[12]="a_12.jpg"
	Input_Numer[13]="a_13.jpg"
	Input_Numer[14]="a_14.jpg"
	Input_Numer[15]="a_15.jpg"
	Input_Numer[16]="a_16.jpg"
	Input_Numer[17]="a_17.jpg"
	Input_Numer[18]="a_18.jpg"
	Input_Numer[19]="a_19.jpg"
	for i in range(20):	 #read 20 digits picture
		img = cv2.imread(Input_Numer[i])	 
		img =cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
		img = cv2.resize(img,(32, 32), interpolation = cv2.INTER_CUBIC)
		#print("img.shape=",img.shape)
		#print("img=",img)
		img_num[i]=img.copy()
		img=img.reshape(My_X.shape[1])
		My_X[i] =img.copy()

	My_test=My_X
	My_label_ohe = vgg_train.encode_labels(My_Yd,10)
	##============================
	
	evaluate(My_test,My_label_ohe,My_Yd)

if __name__ == '__main__':
	main()
