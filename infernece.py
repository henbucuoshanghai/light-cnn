import tensorflow as tf
import numpy as np
FC_SIZE = 512
#Light CNN-9
def inference(input_tensor,regularizer):
	with tf.variable_scope('layer1-conv1'):
		conv1_weights = tf.get_variable(
			"weight", [9,9,3,96 ],
			initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv1_biases = tf.get_variable("bias", [96], initializer=tf.constant_initializer(0.0))
		conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='VALID')
		relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
		res=tf.reshape(relu1,[-1,120,120,2,48])
		relu1=tf.reduce_max(res,axis=[3])
	with tf.name_scope("layer2-pool1"):
		pool1 = tf.nn.max_pool(relu1, ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME")

	with tf.variable_scope("layer3-conv2"):
		conv2_weights = tf.get_variable(
			"weight", [5,5,48,192],
			initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv2_biases = tf.get_variable("bias", [192], initializer=tf.constant_initializer(0.0))
		conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='VALID')
		relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
		res=tf.reshape(relu2,[-1,56,56,2,96])
                relu2=tf.reduce_max(res,axis=[3])

	with tf.name_scope("layer3-pool1"):
                pool2 = tf.nn.max_pool(relu2, ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME")

	with tf.variable_scope("layer4-conv2"):
                conv2_weights = tf.get_variable(
                        "weight", [5,5,96,256],
                        initializer=tf.truncated_normal_initializer(stddev=0.1))
                conv2_biases = tf.get_variable("bias", [256], initializer=tf.constant_initializer(0.0))
                conv2 = tf.nn.conv2d(pool2, conv2_weights, strides=[1, 1, 1, 1], padding='VALID')
                relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
                res=tf.reshape(relu2,[-1,24,24,2,128])
                relu3=tf.reduce_max(res,axis=[3])

        with tf.name_scope("layer4-pool1"):
                pool3 = tf.nn.max_pool(relu3, ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME")

	with tf.variable_scope("layer5-conv2"):
                conv2_weights = tf.get_variable(
                        "weight", [4,4,128,384],
                        initializer=tf.truncated_normal_initializer(stddev=0.1))
                conv2_biases = tf.get_variable("bias", [384], initializer=tf.constant_initializer(0.0))
                conv2 = tf.nn.conv2d(pool3, conv2_weights, strides=[1, 1, 1, 1], padding='VALID')
                relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
                res=tf.reshape(relu2,[-1,9,9,2,192])
                relu4=tf.reduce_max(res,axis=[3])

        with tf.name_scope("layer5-pool1"):
                pool4 = tf.nn.max_pool(relu4, ksize = [1,2,2,1],strides=[1,2,2,1],padding="SAME")
	with tf.name_scope("layer6-pool2"):
		pool_shape = pool4.get_shape()
		nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
		reshaped = tf.reshape(pool4, [-1, nodes])
	with tf.variable_scope('layer7-fc1'):
		fc1_weights = tf.get_variable("weight", [nodes,512],
									  initializer=tf.truncated_normal_initializer(stddev=0.1))
		if regularizer != None: tf.add_to_collection('losses', regularizer(fc1_weights))
		fc1_biases = tf.get_variable("bias", [512], initializer=tf.constant_initializer(0.1))

		fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
		res = tf.reduce_max(tf.reshape(fc1,[-1,2,256]),reduction_indices=[1])


	with tf.variable_scope('layer8-fc2'):
		fc2_weights = tf.get_variable("weight", [256,5],
									  initializer=tf.truncated_normal_initializer(stddev=0.1))
		if regularizer != None: tf.add_to_collection('losses', regularizer(fc2_weights))
		fc2_biases = tf.get_variable("bias", [5], initializer=tf.constant_initializer(0.1))
		logit = tf.matmul(res, fc2_weights) + fc2_biases

	return logit
