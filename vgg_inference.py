
import tensorflow as tf

def batch_norm(inputs, is_training,is_conv_out=True,decay = 0.999):

	scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
	beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
	pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
	pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

	if is_training:
		if is_conv_out:
			batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
		else:
			batch_mean, batch_var = tf.nn.moments(inputs,[0])

		train_mean = tf.assign(pop_mean, pop_mean * decay + batch_mean * (1 - decay))
		train_var = tf.assign(pop_var, pop_var * decay + batch_var * (1 - decay))

		with tf.control_dependencies([train_mean, train_var]):
			return tf.nn.batch_normalization(inputs,
				batch_mean, batch_var, beta, scale, 0.001)
	else:
		return tf.nn.batch_normalization(inputs,
			pop_mean, pop_var, beta, scale, 0.001)
			
def print_activations(t):
  print(t.op.name, ' ', t.get_shape().as_list())


def inference(images,dropout):
	parameters = []
	# conv1
	with tf.variable_scope('layer1-conv1'):
		kernel = tf.Variable(tf.truncated_normal([3, 3, 3, 64], dtype=tf.float32, stddev=1e-1), name='weights')
		conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
		biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
		bias = tf.nn.bias_add(conv, biases)
		bias = batch_norm(bias,True)
		conv1 = tf.nn.relu(bias)
		print_activations(conv1)
		parameters += [kernel, biases]
	

		kernel_b = tf.Variable(tf.truncated_normal([3, 3, 64, 64], dtype=tf.float32, stddev=1e-1), name='weights')
		conv_b = tf.nn.conv2d(conv1, kernel_b, [1, 1, 1, 1], padding='SAME')
		biases_b = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32), trainable=True, name='biases')
		bias_b = tf.nn.bias_add(conv_b, biases_b)
		bias_b = batch_norm(bias_b,True)
		conv1_b = tf.nn.relu(bias_b)
		print_activations(conv1_b)
		parameters += [kernel_b, biases_b]
	
		# pool1
		pool1 = tf.nn.max_pool(conv1_b,
						 ksize=[1, 3, 3, 1],
						 strides=[1, 1, 1, 1],
						 padding='SAME',
						 name='pool1')
		print_activations(pool1)
		
	# conv2
	with tf.variable_scope('layer1-conv2'):
		kernel = tf.Variable(tf.truncated_normal([3, 3, 64, 128], dtype=tf.float32, stddev=1e-1), name='weights')
		conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
		biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name='biases')
		bias = tf.nn.bias_add(conv, biases)
		bias = batch_norm(bias,True)
		conv2 = tf.nn.relu(bias)
		print_activations(conv2)
		parameters += [kernel, biases]
	
	
		kernel_b = tf.Variable(tf.truncated_normal([3, 3, 128, 128], dtype=tf.float32, stddev=1e-1), name='weights')
		conv_b = tf.nn.conv2d(conv2, kernel_b, [1, 1, 1, 1], padding='SAME')
		biases_b = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32), trainable=True, name='biases')
		bias_b = tf.nn.bias_add(conv_b, biases_b)
		bias_b = batch_norm(bias_b,True)
		conv2_b = tf.nn.relu(bias_b)
		print_activations(conv2_b)
		parameters += [kernel_b, biases_b]
	

		# pool2
		pool2 = tf.nn.max_pool(conv2_b,
						 ksize=[1, 3, 3, 1],
						 strides=[1, 2, 2, 1],
						 padding='SAME',
						 name='pool2')
		print_activations(pool2)	
		
		# conv3
	with tf.variable_scope('layer3-conv3'):
		kernel = tf.Variable(tf.truncated_normal([3, 3, 128, 256], dtype=tf.float32, stddev=1e-1), name='weights')
		conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
		biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
		bias = tf.nn.bias_add(conv, biases)
		bias = batch_norm(bias,True)
		conv3 = tf.nn.relu(bias)
		print_activations(conv3)
		parameters += [kernel, biases]	

		kernel_b = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1), name='weights')
		conv_b = tf.nn.conv2d(conv3, kernel_b, [1, 1, 1, 1], padding='SAME')
		biases_b = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
		bias_b = tf.nn.bias_add(conv_b, biases_b)
		bias_b = batch_norm(bias_b,True)
		conv3_b = tf.nn.relu(bias_b)
		print_activations(conv3_b)
		parameters += [kernel_b, biases_b]	
		
		
		kernel_c = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1), name='weights')
		conv_c = tf.nn.conv2d(conv3_b, kernel_c, [1, 1, 1, 1], padding='SAME')
		biases_c = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
		bias_c = tf.nn.bias_add(conv_c, biases_c)
		bias_c = batch_norm(bias_c,True)
		conv3_c = tf.nn.relu(bias_c)
		print_activations(conv3_c)
		parameters += [kernel_c, biases_c]
		
		
		# pool3
		pool3 = tf.nn.max_pool(conv3_c,
						 ksize=[1, 3, 3, 1],
						 strides=[1, 2, 2, 1],
						 padding='SAME',
						 name='pool3')
		print_activations(pool3)	
		
		
		
	# conv4
	with tf.variable_scope('layer4-conv4'):
		kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 512], dtype=tf.float32, stddev=1e-1), name='weights')
		conv = tf.nn.conv2d(pool3, kernel, [1, 1, 1, 1], padding='SAME')
		biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
		bias = tf.nn.bias_add(conv, biases)
		bias = batch_norm(bias,True)
		conv4 = tf.nn.relu(bias)
		print_activations(conv4)
		parameters += [kernel, biases]	

		kernel_b = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), name='weights')
		conv_b = tf.nn.conv2d(conv4, kernel_b, [1, 1, 1, 1], padding='SAME')
		biases_b = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
		bias_b = tf.nn.bias_add(conv_b, biases_b)
		bias_b = batch_norm(bias_b,True)
		conv4_b = tf.nn.relu(bias_b)
		print_activations(conv4_b)
		parameters += [kernel_b, biases_b]	
		
		
		kernel_c = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), name='weights')
		conv_c = tf.nn.conv2d(conv4_b, kernel_c, [1, 1, 1, 1], padding='SAME')
		biases_c = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
		bias_c = tf.nn.bias_add(conv_c, biases_c)
		bias_c = batch_norm(bias_c,True)
		conv4_c = tf.nn.relu(bias_c)
		print_activations(conv4_c)
		parameters += [kernel_c, biases_c]
		
		
		# pool4
		pool4 = tf.nn.max_pool(conv4_c,
						 ksize=[1, 3, 3, 1],
						 strides=[1, 2, 2, 1],
						 padding='SAME',
						 name='pool4')
		print_activations(pool4)	
		
	# conv5
	with tf.variable_scope('layer5-conv5'):
		kernel = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), name='weights')
		conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
		biases = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
		bias = tf.nn.bias_add(conv, biases)
		bias = batch_norm(bias,True)
		conv5 = tf.nn.relu(bias)
		print_activations(conv5)
		parameters += [kernel, biases]	

		kernel_b = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), name='weights')
		conv_b = tf.nn.conv2d(conv5, kernel_b, [1, 1, 1, 1], padding='SAME')
		biases_b = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
		bias_b = tf.nn.bias_add(conv_b, biases_b)
		bias_b = batch_norm(bias_b,True)
		conv5_b = tf.nn.relu(bias_b)
		print_activations(conv5_b)
		parameters += [kernel_b, biases_b]	
		
		
		kernel_c = tf.Variable(tf.truncated_normal([3, 3, 512, 512], dtype=tf.float32, stddev=1e-1), name='weights')
		conv_c = tf.nn.conv2d(conv5_b, kernel_c, [1, 1, 1, 1], padding='SAME')
		biases_c = tf.Variable(tf.constant(0.0, shape=[512], dtype=tf.float32), trainable=True, name='biases')
		bias_c = tf.nn.bias_add(conv_c, biases_c)
		bias_c = batch_norm(bias_c,True)
		conv5_c = tf.nn.relu(bias_c)
		print_activations(conv5_c)
		parameters += [kernel_c, biases_c]
		
		
		# pool5
		pool5 = tf.nn.max_pool(conv5_c,
						 ksize=[1, 3, 3, 1],
						 strides=[1, 2, 2, 1],
						 padding='SAME',
						 name='pool5')
		print_activations(pool5)	
		
		
	with tf.variable_scope('layer6-fc1'):
		
		fc1_weights = tf.get_variable("weight", [2048, 4096], initializer=tf.truncated_normal_initializer(stddev=0.1))
		fc=tf.reshape(pool5,[-1,fc1_weights.get_shape().as_list()[0]])
		fc1_biases= tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32), trainable=True, name='biases')							 
		fc1=tf.add(tf.matmul(fc,fc1_weights),fc1_biases)
		fc1 = batch_norm(fc1,True,False)
		fc1=tf.nn.relu(fc1)
		
		#fc1 = tf.nn.relu(tf.matmul(fc, fc1_weights) + fc1_biases)
		print_activations(fc1)
		fc1=tf.nn.dropout(fc1,dropout)
	
	with tf.variable_scope('layer7-fc2'):
		fc2_weights = tf.get_variable("weight", [4096, 4096], initializer=tf.truncated_normal_initializer(stddev=0.1))
		fc2_biases= tf.Variable(tf.constant(0.0, shape=[4096], dtype=tf.float32), trainable=True, name='biases')							 
		fc2=tf.add(tf.matmul(fc1,fc2_weights),fc2_biases)
		fc2 = batch_norm(fc2,True,False)
		fc2=tf.nn.relu(fc2)
		print_activations(fc2)	
		#dropout
		fc2=tf.nn.dropout(fc2,dropout)
		
	with tf.variable_scope('layer8-out'):	
		#輸出層
		out_weights = tf.get_variable("weight", [4096, 10], initializer=tf.truncated_normal_initializer(stddev=0.1))
		out_biases= tf.Variable(tf.constant(0.0, shape=[10], dtype=tf.float32), trainable=True, name='biases')
		out=tf.add(tf.matmul(fc2,out_weights),out_biases)
		
	return out
		
