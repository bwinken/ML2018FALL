import numpy as np
from numpy.linalg import inv
import pandas as pd
import sys

def parse_data(train_csv,test_csv,label_csv):
	train = pd.read_csv(train_csv)
	test  = pd.read_csv(test_csv)
	label = pd.read_csv(label_csv)

	
	train =pd.concat([train[['PAY_0']],label],axis = 1)
	test = test[['PAY_0']].values

	class1= train[train.Y==1]
	class2= train[train.Y==0]
	
	class1 = class1[['PAY_0']].values
	class2 = class2[['PAY_0']].values
	return class1,class2,test

def gene(class1,class2):

	u1 = np.mean(class1,axis=0)
	u2 = np.mean(class2,axis=0)

	N1 = class1.shape[0]
	N2 = class2.shape[0]
	TOTAL = N1+N2

	
	cov1 = np.cov(np.transpose(class1))
	cov2 = np.cov(np.transpose(class2))

	cov = (N1/TOTAL)*cov1 + (N2/TOTAL)*cov2
	
	cov = cov.reshape(1,1)
	cov1 = cov1.reshape(1,1)
	cov2 = cov2.reshape(1,1)
	
	w = (u1-u2)*inv(cov)
	b = -0.5*u1*inv(cov1)*u1+0.5*u2*inv(cov2)*u2 + np.log(N1/N2)
	
	return w,b
  

train_csv = sys.argv[1]
label_csv = sys.argv[2]
test_csv  = sys.argv[3]
output	  = sys.argv[4]

class1,class2,test = parse_data(train_csv,test_csv,label_csv)
w,b =gene(class1,class2)

y_p = np.dot(test,w)+b

predict = []
for i in range(len(y_p)):
	if y_p[i] > 0:
		predict.append(1)
	else:
		predict.append(0)


with open(output, "w") as f:
	f.write("id,Value\n")
	for i in range(len(predict)-1):
		f.write("id_" + str(i) + "," + str(predict[i]) + "\n")
	f.write("id_" + str(len(predict)-1) + "," + str(predict[len(predict)-1]))
