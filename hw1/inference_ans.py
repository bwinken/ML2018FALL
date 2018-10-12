# coding: utf-8
import pandas as pd
import numpy as np
import sys

def parse_data(csv_file,feature_row,feature_s_row,num_of_feature):

	#===========parsing train data===========

	#read in train.csv into array
	train_data = np.genfromtxt(csv_file,delimiter=',',encoding="gb18030")
	train_data = np.delete(train_data,[0,1],1) #delete first three column of array

	#split train data into days

	days_num = train_data.shape[0]/18 #the total day of collected data
	array_list = np.vsplit(train_data,days_num) #split the train data by day

	#concat array and see the relation between each feature with PM2.5
	train_array = []
	label_array = []
	#feature selection
		#select PM2.5 for past 9 hours
	feature_list = []	
	mean_list = []
	std_list = []
	
	#normalize
	#for i in range(len(feature_row)):
	#	feature_list = []
	#	for array in array_list:
	#		feature_list.append(array[feature_row[i],:])
	#	mean_list.append(np.mean(feature_list))
	#	std_list.append(np.std(feature_list))
	
	#for array in array_list:
	#	for i in range(len(feature_row)):
	#		array[feature_row[i],:] = (array[feature_row[i],:]-mean_list[i])/std_list[i]
	
	#feature_row = [8,9,12]
	#number_of_feature = 36

	
	for array in array_list:
		for i in range(9):
			if np.isnan(array[10][i]):
				array[10][i] = 0;
				
	#rearrange the train data into array which [training_num,feature]
	for array in array_list:
		for i in range(1):
			for j in range(len(feature_row)):
				train_array = np.concatenate((train_array,array[feature_row[j],i:i+9]),axis=0)
			for j in range(len(feature_s_row)):
				train_array = np.concatenate((train_array,array[feature_row[j],i:i+9]**2),axis=0)
	#feature_array = np.delete(feature_array,0,1)

	train_array = np.transpose(train_array)
	train_array = np.reshape(train_array,(-1,num_of_feature))

	return train_array

def feature_transformation(data):
	sample_no,feature_num = data.shape
	feature_type = int((feature_num)/9)
	for i in range(feature_type):
		mean = np.mean(data[:,i*9:i*9+9],axis=1)
		var = np.std(data[:,i*9:i*9+9],axis=1)
		data = np.concatenate((data,np.reshape(mean,(mean.shape[0],1)),np.reshape(var,(var.shape[0],1))),axis=1)
	return data

#============read input command============
test_file = sys.argv[1]
out_file = sys.argv[2]
weight_file = "weight.csv"
#===========define the feature we took and parse the test.csv===========
feature_row = [7,9,12]
feature_s_row = [9]
num_of_feature = (len(feature_row)+len(feature_s_row))*9
data = parse_data(test_file,feature_row,feature_s_row,num_of_feature)

#===============feature transformation + expand data=================
#data = feature_transformation(data)

#==========concat ones array==============
num_of_data,num_of_feature = data.shape
ones_array = np.ones((num_of_data,1))
data = np.concatenate((ones_array,data),axis=1)

#==============read in weight==============
weight = np.genfromtxt(weight_file,delimiter=',')
weight = np.reshape(weight,(weight.shape[0],1))

#=============Predict==============
y_predict = np.dot(data,weight)
#y_predict = np.reshape(y_predict,(y_predict.shape[0],1))
#===========write in ans.csv============
ans = np.array([["id","value"]])
head = np.array(["id","value"])
id_list = []
for i in range(y_predict.shape[0]):
	id_list.append("id_"+str(i))

id_array = np.asarray([id_list])
id_array = np.transpose(id_array)

id_array =np.concatenate((id_array,y_predict),axis=1)

ans = np.concatenate((ans,id_array),axis=0)
ANS = pd.DataFrame(ans)
ANS.to_csv(out_file,index = False,header = False)


