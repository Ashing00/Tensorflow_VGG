import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os
import struct
import numpy as np
from matplotlib import pyplot as plt

import cv2,csv
import vgg_inference

data_dir = "data/"
extract_folder = 'cifar-10-batches-bin'
def encode_labels( y, k):
	"""Encode labels into one-hot representation
	"""
	onehot = np.zeros((y.shape[0],k ))
	for idx, val in enumerate(y):
		onehot[idx,val] = 1.0  ##idx=0~xxxxx，if val =3 ,表示欄位3要設成1.0
	return onehot


def load_train_data(n):			#n=1,2..5,data_batch_1.bin ~data_batch_5.bin
	"""Load Cifar10 data from `path`"""
	images_path = os.path.join(data_dir, extract_folder, 'data_batch_{}.bin'.format(n)) 
	with open(images_path, 'rb') as imgpath:
		images = np.fromfile(imgpath, dtype=np.uint8)
	return images
	
def load_test_data():			#test_batch
	"""Load Cifar10 test data from `path`"""
	test_path = os.path.join(data_dir, extract_folder, 'test_batch.bin') 
	with open(test_path, 'rb') as testpath:
		test_img = np.fromfile(testpath, dtype=np.uint8)
	return test_img	

# Parameters
MODEL_SAVE_PATH = "./vgg/"
MODEL_NAME = "vgg_cifar_model"
learning_rate = 0.001
BATCH_SIZE = 120
display_step = 10
TRAINING_STEPS=3000
# Network Parameters
n_input = 3072 # cifar data input (img shape: 32x32x3)
n_classes = 10 # cifar10 total classes (0-9 )
dropout = 0.60# Dropout, probability to keep units

def train(X_train,y_train_lable):
	shuffle=True
	batch_idx=0
	batch_len =int( X_train.shape[0]/BATCH_SIZE)
	train_loss=[]
	train_acc=[]
	train_idx=np.random.permutation(batch_len)#打散btach_len=500 group

	# tf Graph input
	x_ = tf.placeholder(tf.float32, [None, n_input])	
	y = tf.placeholder(tf.float32, [None, n_classes])
	keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)
	x = tf.reshape(x_, shape=[-1, 32, 32, 3])

	# Construct model
	pred =vgg_inference.inference(x,  keep_prob)

	# Define loss and optimizer
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
	#optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
	#GradientDescentOptimizer
	# Evaluate model
	correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

	# 初始化TensorFlow持久化類。
	saver = tf.train.Saver()
	# Initializing the variables
	init = tf.global_variables_initializer()

	# Launch the graph
	with tf.Session() as sess:
		sess.run(init)
		step = 1
		print ("Start  training!")
		# Keep training until reach max iterations:
		while step	< TRAINING_STEPS:
			#batch_xs, batch_ys = mnist.train.next_batch(BATCH_SIZE)
			if shuffle==True:
				batch_shuffle_idx=train_idx[batch_idx]
				batch_xs=X_train[batch_shuffle_idx*BATCH_SIZE:batch_shuffle_idx*BATCH_SIZE+BATCH_SIZE]
				batch_ys=y_train_lable[batch_shuffle_idx*BATCH_SIZE:batch_shuffle_idx*BATCH_SIZE+BATCH_SIZE]	
			else:
				batch_xs=X_train[batch_idx*BATCH_SIZE:batch_idx*BATCH_SIZE+BATCH_SIZE]
				batch_ys=y_train_lable[batch_idx*BATCH_SIZE:batch_idx*BATCH_SIZE+BATCH_SIZE]
		
			if batch_idx<batch_len:
				batch_idx+=1
				if batch_idx==batch_len:
					batch_idx=0
			else:
				batch_idx=0
			reshaped_xs = np.reshape(batch_xs, (
					BATCH_SIZE,
					32,
					32,
					3))
			
			# Run optimization op (backprop)
			sess.run(optimizer, feed_dict={x: reshaped_xs, y: batch_ys,
										keep_prob: dropout})
			# Calculate batch loss and accuracy
			loss, acc = sess.run([cost, accuracy], feed_dict={x: reshaped_xs,
																y: batch_ys,
																keep_prob: 1.})
			train_loss.append(loss)
			train_acc.append(acc)
			if step % display_step == 0:
				print("Step: " + str(step) + ", Minibatch Loss= " + \
					"{:.6f}".format(loss) + ", Training Accuracy= " + \
					"{:.5f}".format(acc))
			step += 1
		print("Optimization Finished!")
		print("Save model...")
		#saver.save(sess, "./vgg/vgg_model")
		saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
		
		plt.subplot(1,2,1)
		plt.plot(train_loss)
		plt.xlabel('Iter')
		plt.ylabel('loss')
		plt.title('lr=%f, ti=%d, bs=%d' % (learning_rate, TRAINING_STEPS, BATCH_SIZE))
		#plt.tight_layout()

		plt.subplot(1,2,2)
		plt.plot(train_acc)
		plt.xlabel('Iter')
		plt.ylabel('accuracy')
		plt.title('lr=%f, ti=%d, bs=%d' % (learning_rate, TRAINING_STEPS, BATCH_SIZE))
		#plt.tight_layout()
		plt.savefig('vgg_cifar10_acc.jpg', dpi=200)
		plt.show()				
		
										
def main(argv=None):

	
	##Load Cifar-10 train image and label	
	X_train_image1 = load_train_data(1)	#load data_batch_1.bin
	X_train_image2 = load_train_data(2)	#load data_batch_2.bin
	X_train_image3 = load_train_data(3)	#load data_batch_3.bin
	X_train_image4 = load_train_data(4)	#load data_batch_4.bin
	X_train_image5 = load_train_data(5)	#load data_batch_5.bin
	print(X_train_image1.shape)
	
	X_train_image=np.concatenate((X_train_image1,X_train_image2,X_train_image3,X_train_image4,X_train_image5),axis=0)
	print(X_train_image.shape)
	
	#reshape to (50000,3073)
	#in one Row ,the 1st byte is the label,other 3072byte =1024 Red +1024 green +1024 blue ch data
	X_train_image=X_train_image.reshape(-1,3073)
	tempA=X_train_image.copy()
	X_train_image=np.delete(X_train_image, 0, 1) #delete 1st column data
	X_train_image=X_train_image.reshape(-1,3,32,32)  #(50000,3,32,32)
	X_train_image = X_train_image.transpose([0, 2, 3, 1])	#transfer to (10000,32,32,3)
	X_train_image=X_train_image.reshape(-1,3072)  #(50000,3,32,32)

	#split to 3073 col,the first column is the label.
	tempA=np.hsplit(tempA,3073)	
	X_train_label=np.asarray(tempA[0])
	X_train_label=X_train_label.reshape([50000,])

	print(X_train_image.shape)	
	print(X_train_label.shape)	
	#print(X_train_label[0:50])	
	

	X_train_label = encode_labels(X_train_label,10)
	print("y_train_lable.shape=",X_train_label.shape)
	#print(X_train_label[0:50])	
	##============================
	
	train(X_train_image,X_train_label)

if __name__ == '__main__':
	main()
