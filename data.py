import os
import cv2
import random
kind_class=['sunflowers', 'daisy', 'roses', 'tulips', 'dandelion']
dict_class=dict(zip(kind_class,range(len(kind_class))))

batch=15
data_pwd='/home/ubuntu/flower_photos'
train_data=[]
teat_data=[]
def get_data(data_pwd):
	global train_data,teat_data
	i=0
        for kind in os.listdir(data_pwd):
		all_flow=[]
		one_flower=os.path.join(data_pwd,kind)
		for img in os.listdir(one_flower):
			all_flow.append((os.path.join(one_flower,img),i))	
		lll=int(0.9*len(all_flow))
		train_data=train_data+all_flow[:lll]
		teat_data=teat_data+all_flow[lll:]
		i+=1
	return train_data,teat_data

def get_input(data_list):
	train_img=[]
	global index
	index=0
	num=len(data_list)//batch
	if index<=num:
		train_img=data_list[index*batch:index*batch+batch]
		index+=1
		if index>num:
			index=0
			random.shuffle(data_list)
	return train_img		
def dnn_input(data):
	dnn_imgs=[]
	label=[]
	for i in data:
		img=cv2.imread(i[0])
		img=cv2.resize(img,(128,128))
		img=img*1.0/255
		dnn_imgs.append(img)
		label.append(i[1])

	return  dnn_imgs,label

if __name__=='__main__':
	train_data,teat_data=get_data(data_pwd)
	random.shuffle(train_data)	
	train_img=get_input(train_data)
	dnn_imgs,label=dnn_input(train_img)
	print label
	print dnn_imgs[0]
	print dict_class
