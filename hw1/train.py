# coding: utf-8
import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from numpy import linalg as LA

def parse_data(csv_file,feature_row,feature_s_row,num_of_feature):

	#===========parsing train data===========
	array_list = []
	#read in train.csv into array
	train_data = np.genfromtxt(csv_file,delimiter=',',encoding="big5")
	train_data = np.delete(train_data,[0,1,2],1) #delete first three column of array
	train_data = np.delete(train_data,0,0) #delete first row of array

	#split train data into days
	
	total_days_num = train_data.shape[0]/18 #the total day of collected data
	day_list = np.vsplit(train_data,total_days_num) #split the train data by day
	month_num = 12
	days_in_month = int(total_days_num/month_num)
	
	for i in range(month_num):
		index = i*days_in_month
		array = day_list[index]
		for j in range(days_in_month-1):
			array = np.hstack((array,day_list[index+j+1]))
		array_list.append(array)
		
	#concat array and see the relation between each feature with PM2.5
	train_array = []
	label_array = []
	#feature selection
		#select PM2.5 for past 9 hours
	feature_list = []	
	#process NR 
	for array in array_list:
		for i in range(480):
			if np.isnan(array[10][i]):
				array[10][i] = 0;

	#rearrange the train data into array which [training_num,feature]
	for array in array_list:
		for i in range(471):
			for j in range(len(feature_row)):
				train_array = np.concatenate((train_array,array[feature_row[j],i:i+9]),axis=0)
			for j in range(len(feature_s_row)):
				train_array = np.concatenate((train_array,array[feature_row[j],i:i+9]**2),axis=0)
		label_array = np.concatenate((label_array,array[9,9:480]),axis=0)

	train_array = np.transpose(train_array)
	train_array = np.reshape(train_array,(-1,num_of_feature))
	
	label_array = np.reshape(label_array,(label_array.shape[0],1))
	train_array = np.concatenate((train_array,label_array),axis=1)
	
	return train_array

def remove(array_list,remove_index):
	list = []
	for i in range(len(remove_index)):
		if i not in remove_index:
			list.append(array_list[i])
	return list
	
def shuffle(array):
	return np.random.permutation(array)

class LinearRegression_GD_ADA(object):

	def __init__(self,lr_b=30,lr_w=5,n_iter=10000,batch=100,regularization=0.005):
		self.lr_b = lr_b
		self.lr_w = lr_w
		self.n_iter = n_iter
		self.batch = batch
		self.regularization = regularization
	
	def normalize(self,train_set):
		mean = np.mean(train_set,axis=0)
		std = np.std(train_set,axis=0)
		
		return (train_set-mean)/std,mean,std
		
	def RMSE(self,loss):
		return np.sqrt(np.mean(loss**2))
		
	def fit(self,train_set,test_set):
	
		#==========Training model==============
		
		num_of_data,num_of_feature = train_set.shape
		
		sample_no = train_set.shape[0]
		ones_array = np.ones((num_of_data,1))
		
		#split the data into feature and ans
		x = train_set[:,:(num_of_feature-1)]
		y = train_set[:,-1]
		print("X shape is:",x.shape)
		print("Y shape is:",y.shape)
		
		#normalize
		#l,mean,std = self.normalize(x)
		#print("NP.MEAN shape is:",mean)
		#print("NP.STD shape is:",std.shape)
		
		#concat ones into x
		x = np.concatenate((ones_array,x),axis=1)
		print("X shape is:" , x.shape)
		y = np.reshape(y,(y.shape[0],1))
		
		#initialize weight and bias which first row is bias
		w = np.zeros((x.shape[1],1))
		
		#define loss function
		
		
		RMSE_ite = []
		#===start training===
		x_T = np.transpose(x)
		s_grad = 0;
		for i in range(self.n_iter):
			
			for j in range(int(num_of_data/self.batch)):
				#predict value based on current parameter
				y_predict = np.dot(x[j*self.batch:(j+1)*self.batch,:],w)
				#compute loss
				loss = y[(j*self.batch):(j+1)*self.batch,:]-y_predict
				#compute gradient1
				gradient = (-2)*np.dot(x_T[:,j*self.batch:(j+1)*self.batch],loss)+self.regularization*w
				#gradient /= self.batch
				#using ada
				s_grad = s_grad+gradient**2
					#print("Gradient is " ,gradient)
				ada = np.sqrt(s_grad)
				#updating the parameter
				w[:1,:] = w[:1,:] - self.lr_b*gradient[:1,:]/ada[:1,:]
				w[1:,:] = w[1:,:] - self.lr_w*gradient[1:,:]/ada[1:,:]
			RMSE_ite.append(self.RMSE(loss))
		
		
		#===========Testing Model================
		
		num_of_data,num_of_feature = test_set.shape
		
		ones_array = np.ones((num_of_data,1))
		x = test_set[:,:(num_of_feature-1)]

		x = np.concatenate((ones_array,x),axis=1)
		y = test_set[:,-1]
		
		
		#predicting
		y_predict = np.dot(x,w)
		error = y - y_predict
		

		np.savetxt("y.csv",data[:,-1],delimiter=",")
		np.savetxt("y_predict.csv",np.dot(x,w),delimiter=",")
		
		s_error = error**2
		ms_error = np.mean(s_error)
		RMSE = np.sqrt(ms_error)
		
		return w,self.RMSE(loss)
		
		
	def analytic_solution(self,data):
		num_of_data,num_of_feature = data.shape
		
		sample_no = data.shape[0]
		ones_array = np.ones((num_of_data,1))
		
		#split the data into feature and ans
		x = data[:,:(num_of_feature-1)]
		y = data[:,-1]
		
		#concat ones into x
		x = np.concatenate((ones_array,x),axis=1)
		print("X shape is:" , x.shape)
		y = np.reshape(y,(y.shape[0],1))
		
		w = np.dot(LA.pinv(x),y)
		
		return w	
def cross_validation(model_in,data,partition):
	loss = []
	total_data = data.shape[0]
	best_w = []
	amount_of_data = int(total_data/partition)
	for i in range(partition):
		train = data[amount_of_data:,:]
		test = data[:amount_of_data,:]
		data = np.roll(data,amount_of_data,axis=0)
		w,RMSE = model_in.fit(train,test)
		if loss:
			if RMSE < min(loss):
				best_w = w
		loss.append(RMSE)
		
	return best_w,loss#np.mean(loss)
	
def feature_transformation(data):
	sample_no,feature_num = data.shape
	feature_type = int((feature_num)/9)
	for i in range(feature_type):
		mean = np.mean(data[:,i*9:i*9+9],axis=1)
		var = np.std(data[:,i*9:i*9+9],axis=1)
		data = np.concatenate((data,np.reshape(mean,(mean.shape[0],1)),np.reshape(var,(var.shape[0],1))),axis=1)
	return data

def select_feature_based_on_corr(data,corr):
	num_of_feature = corr.shape[0]
	delete_list = []
	for i in range(num_of_feature):
		if corr[i] > -0.2 and corr[i]<0.2:
			#print("DELETE",i," th column")
			#print(" CORR[",i,"] is:", corr[i])
			delete_list.append(i)
	data = np.delete(data,delete_list,axis=1)
	return data
	
def clean_data(data):
	remove_index = []
	for i in range(data.shape[0]):
		if(data[i,36] <=0 or data[i,36]>=130):
			remove_index.append(i)
			continue
		for j in range(data.shape[1]-10):
			if(data[i,j] <=0 or data[i,j]>=130):
				remove_index.append(i)
				break
	data = np.delete(data,remove_index,axis=0)
	return data



train_file = sys.argv[1]

feature_row = [7,9,12]
feature_s_row = [9]
num_of_feature = (len(feature_row)+len(feature_s_row))*9
data = parse_data(train_file,feature_row,feature_s_row,num_of_feature)
corr = np.corrcoef(data,rowvar=False)

#========clean data===============
data = clean_data(data)
#label = np.reshape(data[:,-1],(data[:,-1].shape[0],1))
#data = feature_transformation(data[:,:36])
#data = np.concatenate((data,label),axis=1)
print("Data Shape is:",data.shape)

#==========create model===========
model = LinearRegression_GD_ADA(lr_b=0.001,lr_w=0.001,n_iter=100000,batch=200,regularization=0)
w,loss = model.fit(data,data)
#w,loss = cross_validation(model,data,8)#np.concatenate((data[:600,:],data[1200:3600,:]),axis=0),10)

#print("MAX is:",np.max(data))
#print("weight is",w)
#print("RMSE is:",loss)
#print("AVERAGE is:",np.mean(loss))


"""
#====================================================TESTING Area====================================================
model = LinearRegression_GD_ADA(lr_b=10,lr_w=10,n_iter=1000,batch=200)
w,loss_0 = model.fit(data,data)

model = LinearRegression_GD_ADA(lr_b=1,lr_w=1,n_iter=1000,batch=200)
w,loss_1 = model.fit(data,data)

model = LinearRegression_GD_ADA(lr_b=0.1,lr_w=0.1,n_iter=1000,batch=200)
w,loss_2 = model.fit(data,data)

model = LinearRegression_GD_ADA(lr_b=0.01,lr_w=0.01,n_iter=1000,batch=200)
w,loss_3 = model.fit(data,data)

model = LinearRegression_GD_ADA(lr_b=0.001,lr_w=0.001,n_iter=1000,batch=200)
w,loss_4 = model.fit(data,data)

#=======================================================================================================================
#=========plot graph=========
ite = range(len(loss))
ite = [j+1 for j in ite]
line1, = plt.plot(ite,loss_0,'b')
line2, = plt.plot(ite,loss_1,'g')
line3, = plt.plot(ite,loss_2,'r')
line4, = plt.plot(ite,loss_3,'k')
line5, = plt.plot(ite,loss_4,'m')

plt.legend([line1,line2,line3,line4,line5],['LR=10','LR=1','LR=0.1','LR=0.01','LR=0.001'],fontsize=20)
plt.xlabel('iteration count')
plt.ylabel('RMSE')
#plt.axis([0,1000,0,500])
plt.show()
"""
"""

#===========================Testing Area 2===================================================
model = LinearRegression_GD_ADA(lr_b=0.01,lr_w=0.01,n_iter=10000,batch=200,regularization=100000000)
w_0,loss_0 = model.fit(data,data)

model = LinearRegression_GD_ADA(lr_b=0.01,lr_w=0.01,n_iter=10000,batch=200,regularization=10000000)
w_1,loss_1 = model.fit(data,data)

model = LinearRegression_GD_ADA(lr_b=0.01,lr_w=0.01,n_iter=10000,batch=200,regularization=1000000)
w_2,loss_2 = model.fit(data,data)

model = LinearRegression_GD_ADA(lr_b=0.01,lr_w=0.01,n_iter=10000,batch=200,regularization=100000)
w_3,loss_3 = model.fit(data,data)

model = LinearRegression_GD_ADA(lr_b=0.01,lr_w=0.01,n_iter=10000,batch=200,regularization=10000)
w_4,loss_4 = model.fit(data,data)

model = LinearRegression_GD_ADA(lr_b=0.01,lr_w=0.01,n_iter=10000,batch=200,regularization=1000)
w_5,loss_5 = model.fit(data,data)

#plot graph
x = [loss_5,loss_4,loss_3,loss_2,loss_1,loss_0];
x_kaggle = [8.1286,8.432,10.433,14.93,23.696,26.825];
print("LOSS is:",x)
plt.figure(1)
line1, = plt.plot(['100000000','10000000','1000000','100000','10000','1000'],x,'b')
line2, = plt.plot(['100000000','10000000','1000000','100000','10000','1000'],x_kaggle,'g')
plt.legend([line1,line2],["Training RMSE","Kaggle RMSE"])
plt.xscale('log')
plt.xlabel("lambda")
plt.ylabel("RMSE")
plt.show()


W = [LA.norm(w_0,2),LA.norm(w_1,2),LA.norm(w_2,2),LA.norm(w_3,2),LA.norm(w_4,2),LA.norm(w_5,2)]
print("WEIGHT is:",W)
plt.figure(2)
plt.title("Weight Norm")
plt.plot(['100000000','10000000','1000000','100000','10000','1000'],x,'b')
plt.xscale('log')
plt.xlabel("lambda")
plt.ylabel("Norm")
plt.show()
"""

np.savetxt("weight.csv",w,delimiter=",")