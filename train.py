#coding:utf-8
import random
import time
import tensorflow as tf
import infernece
from tensorflow.examples.tutorials.mnist import input_data
import data
import os
import struct
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import cv2,csv

			
def encode_labels( y, k):
	"""Encode labels into one-hot representation
	"""
	onehot = np.zeros((y.shape[0],k ))
	for idx, val in enumerate(y):
		onehot[idx,val] = 1.0  ##idx=0~xxxxx，if val =3 ,表示欄位3要設成1.0
	return onehot

BATCH_SIZE = 500 
LEARNING_RATE_BASE = 0.001
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 1000
MOVING_AVERAGE_DECAY = 0.99 
MODEL_SAVE_PATH = "./lenet5/"
MODEL_NAME = "lenet5_model"
INPUT_NODE = 784
OUTPUT_NODE = 10
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10
display_step =50
learning_rate_flag=True
batch=data.batch

def train(train_data):
	shuffle=True
	batch_idx=0
	
	test_acc=[]
	train_acc=[]
	x_ = tf.placeholder(tf.float32, [None,128,128,3],name='x-input')	
	y_ = tf.placeholder(tf.float32, [None,5], name='y-input')

	regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
	logits= infernece.inference(x_,regularizer)
	print logits
	global_step = tf.Variable(0, trainable=False)

        pred_max=tf.argmax(logits,1)
        y_max=tf.argmax(y_,1)
        correct_pred = tf.equal(pred_max,y_max)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


	if learning_rate_flag==True:
		learning_rate = tf.train.exponential_decay(
			LEARNING_RATE_BASE,
			global_step,
			50, LEARNING_RATE_DECAY,
			staircase=True)
	else:	
		learning_rate = 0.001 #Ashing test
	cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=tf.argmax(y_, 1))
	cross_entropy_mean = tf.reduce_mean(cross_entropy)
	loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

	# 初始化TensorFlow持久化類。
	saver = tf.train.Saver()
	with tf.Session() as sess:
		tf.global_variables_initializer().run()
		index=0
		step = 1
		print ("Start  training!")
		while step	< TRAINING_STEPS:
			train_img=[]
        		num=len(train_data)//batch
        		if index<=num:
                		train_img=train_data[index*batch:index*batch+batch]
                		index+=1
                		if index>num:
                        		index=0
                       			random.shuffle(train_data)
			dnn_imgs,label=data.dnn_input(train_img)
		        y_train_lable = encode_labels(np.array(label),5)
			_,step,acc_train,loss_value=sess.run([train_step,global_step,accuracy,loss], feed_dict={x_:dnn_imgs, y_:y_train_lable})
			if step % display_step == 0:
				print   'After',step,('steps acc on train batch is %g'%(acc_train)),'loss is',loss_value
				saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME),global_step=global_step)
			step += 1
		print ("Optimization Finished!")
		#train_acc_avg=tf.reduce_mean(tf.cast(train_acc, tf.float32))	
		print("Save model...")
		#saver.save(sess, "./lenet5/lenet5_model")
		saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))

		
def main(argv=None):
	data_pwd='/home/ubuntu/flower_photos'
	train_data,teat_data=data.get_data(data_pwd)
        random.shuffle(train_data)
	train(train_data)

if __name__ == '__main__':
        start = time.time()
	main()
        end = time.time()
        print  end-start
        print  'I have trained %d mins and %d seconds'%((end-start)/60,(end-start)%60)
#coding:utf-8
