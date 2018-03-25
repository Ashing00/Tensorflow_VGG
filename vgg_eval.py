import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import vgg_inference
import vgg_train
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

data_dir = "data/"
extract_folder = 'cifar-10-batches-bin'

def encode_labels( y, k):
	"""Encode labels into one-hot representation
	"""
	onehot = np.zeros((y.shape[0],k ))
	for idx, val in enumerate(y):
		onehot[idx,val] = 1.0  ##idx=0~xxxxx，if val =3 ,表示欄位3要設成1.0
	return onehot
def load_test_data():			#test_batch
	"""Load Cifar10 test data from `path`"""
	test_path = os.path.join(data_dir, extract_folder, 'test_batch.bin') 
	with open(test_path, 'rb') as testpath:
		test_img = np.fromfile(testpath, dtype=np.uint8)
	return test_img	

def evaluate(X_test,y_test_lable):
	with tf.Graph().as_default() as g:
	
		# 定義輸出為4維矩陣的placeholder
		x_ = tf.placeholder(tf.float32, [None, vgg_train.n_input])	
		x = tf.reshape(x_, shape=[-1, 32, 32, 3])
		y = tf.placeholder(tf.float32, [None, vgg_train.n_classes])
	
		# Construct model
		pred = vgg_inference.inference(x, 1)     #dropout=1

		# Evaluate model
		correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
	
		test_batch_len =int( X_test.shape[0]/vgg_train.BATCH_SIZE)
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

			for i in range(test_batch_len):
				temp_acc= sess.run(accuracy, feed_dict={x: test_xs[batchsize*i:batchsize*i+batchsize], y: y_test_lable[batchsize*i:batchsize*i+batchsize]})
				test_acc.append(temp_acc)
				print ("Test  batch ",i,":Testing Accuracy:",temp_acc)	

			t_acc=tf.reduce_mean(tf.cast(test_acc, tf.float32))	
			print("Average Testing Accuracy=",sess.run(t_acc))
			return

def main(argv=None):

	##Load Cifar-10 test image  and label	
	X_test_image = load_test_data()	#load test_batch.bin
	#reshape to (10000,3073)
	#in one Row ,the 1st byte is the label,other 3072byte =1024 Red +1024 green +1024 blue ch data
	X_test_image=X_test_image.reshape(-1,3073)
	tempA=X_test_image.copy()
	X_test_image=np.delete(X_test_image, 0, 1) #delete 1st column data
	X_test_image=X_test_image.reshape(-1,3,32,32)  #(1000,3,32,32)
	X_test_image = X_test_image.transpose([0, 2, 3, 1])	#transfer to (10000,32,32,3)
	X_test_image=X_test_image.reshape(-1,3072)  #(50000,3,32,32)

	#split to 3073 col,the first column is the label.
	tempA=np.hsplit(tempA,3073)	
	X_test_label=np.asarray(tempA[0])
	X_test_label=X_test_label.reshape([10000,])

	
	#mms=MinMaxScaler()
	#X_test_image=mms.fit_transform(X_test_image)
	
	X_test_label = encode_labels(X_test_label,10)
	
	
	print("X_test_image.shape=",X_test_image.shape)	
	print("X_test_label.shape=",X_test_label.shape)
	print(X_test_label[0:50])	
	

	
	evaluate(X_test_image,X_test_label)

if __name__ == '__main__':
	main()
