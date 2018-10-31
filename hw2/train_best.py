import numpy as np
import pandas as pd
import sys

def parse_data(train_csv,test_csv,label_csv):
	train = pd.read_csv(train_csv)
	test  = pd.read_csv(test_csv)
	label = pd.read_csv(label_csv)
	
	#add
	#train['ADD'] = train['LIMIT_BAL']**2
	#test['ADD'] = test['LIMIT_BAL']**2
	#train['ADD1'] = train['LIMIT_BAL']*train['PAY_0']
	#test['ADD1'] = test['LIMIT_BAL']*test['PAY_0']
	
	
	FEATURE = ['LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE','PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6','BILL_AMT1','BILL_AMT2',
	'BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
	
	VALUE_DATA =['LIMIT_BAL','AGE','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1',
	'PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
	
	NEW_VALUE = ['AGE','LIMIT_BAL','BILL_AMT1','BILL_AMT2','BILL_AMT3']
	
	SEX 	= train[['SEX']].values
	EDU 	= train[['EDUCATION']].values
	MAR 	= train[['MARRIAGE']].values
	#AGE 	= train[['AGE']].values
	PAY_0 	= train[['PAY_0']].values
	PAY_2 	= train[['PAY_2']].values
	PAY_3 	= train[['PAY_3']].values
	PAY_4 	= train[['PAY_4']].values
	PAY_5 	= train[['PAY_5']].values
	PAY_6 	= train[['PAY_6']].values
	VALUE 	= train[VALUE_DATA].values
	#normalize VALUE
	VALUE = (VALUE - VALUE.mean())/VALUE.std()
	#label
	label = label[['Y']].values**2
	#One hot encoding
	SEX_OH 		= one_hot(SEX-np.min(SEX),np.max(SEX)-np.min(SEX)+1)
	EDU_OH 		= one_hot(EDU-np.min(EDU),np.max(EDU)-np.min(EDU)+1)
	MAR_OH 		= one_hot(MAR-np.min(MAR),np.max(MAR)-np.min(MAR)+1)
	PAY0_OH 	= one_hot(PAY_0-np.min(PAY_0),11)
	PAY2_OH 	= one_hot(PAY_2-np.min(PAY_2),11)
	PAY3_OH 	= one_hot(PAY_3-np.min(PAY_3),11)
	PAY4_OH 	= one_hot(PAY_4-np.min(PAY_4),11)
	PAY5_OH 	= one_hot(PAY_5-np.min(PAY_5),11)
	PAY6_OH 	= one_hot(PAY_6-np.min(PAY_6),11)
	
	#======================TEST===================
	SEX_T 	= test[['SEX']].values
	EDU_T 	= test[['EDUCATION']].values
	MAR_T 	= test[['MARRIAGE']].values
	#AGE 	= test[['AGE']].values
	PAY_0_T 	= test[['PAY_0']].values
	PAY_2_T 	= test[['PAY_2']].values
	PAY_3_T 	= test[['PAY_3']].values
	PAY_4_T 	= test[['PAY_4']].values
	PAY_5_T 	= test[['PAY_5']].values
	PAY_6_T 	= test[['PAY_6']].values
	VALUE_T 	= test[VALUE_DATA].values
	#normalize VALUE
	VALUE_T = (VALUE_T - VALUE.mean())/VALUE.std()
    
	#One hot encoding
	SEX_OH_T 		= one_hot(SEX_T-np.min(SEX),2)
	EDU_OH_T 		= one_hot(EDU_T-np.min(EDU),7)
	MAR_OH_T 		= one_hot(MAR_T-np.min(MAR),4)
	PAY0_OH_T	 	= one_hot(PAY_0_T-np.min(PAY_0),11)
	PAY2_OH_T	 	= one_hot(PAY_2_T-np.min(PAY_2),11)
	PAY3_OH_T	 	= one_hot(PAY_3_T-np.min(PAY_3),11)
	PAY4_OH_T	 	= one_hot(PAY_4_T-np.min(PAY_4),11)
	PAY5_OH_T	 	= one_hot(PAY_5_T-np.min(PAY_5),11)
	PAY6_OH_T	 	= one_hot(PAY_6_T-np.min(PAY_6),11)

	#train = np.hstack((PAY0_OH,PAY2_OH,PAY3_OH,PAY4_OH,PAY5_OH,PAY6_OH,SEX_OH,EDU_OH,MAR_OH,VALUE))
	#test = np.hstack((PAY0_OH_T,PAY2_OH_T,PAY3_OH_T,PAY4_OH_T,PAY5_OH_T,PAY6_OH_T,SEX_OH_T,EDU_OH_T,MAR_OH_T,VALUE_T))


	train = np.hstack((PAY0_OH,PAY2_OH,PAY3_OH,PAY4_OH,PAY5_OH,PAY6_OH))
	test = np.hstack((PAY0_OH_T,PAY2_OH_T,PAY3_OH_T,PAY4_OH_T,PAY5_OH_T,PAY6_OH_T))
	return train,test,label

def sigmoid(x):
	return 1/(1.0+np.exp(-x))

def cross_entropy(y,y_p):
	return -(y*np.log(y_p)+(1-y)*np.log(1-y_p))

def shuffle(train,label):
	randomize = np.arange(len(train))
	np.random.shuffle(randomize)
	return (train[randomize], label[randomize])
	
def one_hot(data, nb_classes):
    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]
  
class Logistic_Regression(object):
	
	def __init__(self,lr=0.01,batch_size=200,epoch=200):
		self.lr 		= lr
		self.batch_size = batch_size
		self.epoch		= epoch
		
	def train(self,train,label):
		L_ite= []
		data_no,feature_no = train.shape
		step_no = int(data_no/self.batch_size)
		s_grad_b = 1
		s_grad_w = np.ones((feature_no,1))
	
		b 	= 0
		w 	= np.zeros((feature_no,1))
		
		for i in range(self.epoch):
			error = 0
			for j in range(step_no):
				x = train[j*self.batch_size:(j+1)*self.batch_size,:]
				y = label[j*self.batch_size:(j+1)*self.batch_size,:]
				
				x_T = np.transpose(x)
				
				y_p = sigmoid(np.dot(x,w)+b)
				L	= y - y_p
				
				gra_b = -2*np.sum(L)/self.batch_size 
				gra_w = -1*np.dot(x_T,L)/self.batch_size
				
				s_grad_b = s_grad_b + gra_b**2
				s_grad_w = s_grad_w + gra_w**2
				
				#ada
				ada_b = np.sqrt(s_grad_b)
				ada_w = np.sqrt(s_grad_w)
				#update
				b = b - self.lr*gra_b/ada_b
				w = w - self.lr*gra_w/ada_w
				#print(b)
				#print(w)
				L_ite.append(np.sum(L))
				error += L**2

			# print(error)
			#print(np.sqrt(np.sum(error)/j/self.batch_size))
			
		return b,w,L_ite
		
	def evaluate(self,test,label,w,b,th):
		data_no,feature_no = test.shape
		cor_cnt = 0
		wro_cnt = 0
		y_p = np.dot(test,w)+b
		for i in range(data_no):
			if y_p[i] >= th and label[i] ==1:
				cor_cnt += 1
			elif y_p[i] < th and label[i] ==0:
				cor_cnt += 1

		return cor_cnt/data_no
	#def test(self,test,label):
	
	
train_csv = sys.argv[1]
label_csv = sys.argv[2]
test_csv  = sys.argv[3]
output	  = sys.argv[4]

train,test,label = parse_data(train_csv,test_csv,label_csv)
model = Logistic_Regression(lr=1,batch_size=50,epoch=1000)
b,w,L = model.train(train,label)
#train,label = shuffle(train,label)

ACC = []
TH = []
for i in range(20):
	th = -1+i*0.1
	acc = model.evaluate(train[:5000,:],label[:5000,:],w,b,th)
	TH.append(th)
	ACC.append(acc)

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
