from keras.models import load_model
import sys
import pandas as pd
import numpy as np

def parse_data(filename):
	
	train = pd.read_csv(filename)
	
	train = train[['feature']].values

	print("TRAIN shape:",train.shape)
	
	new_train = np.zeros((train.shape[0],48*48))
	for i in range(train.shape[0]):
		new_train[i,:] = train[i,0].split()
		
	new_train = new_train.reshape(new_train.shape[0],48,48,1).astype('float32')

	return new_train
	
	
test_csv = sys.argv[1]
ans_csv = sys.argv[2]
test = parse_data(test_csv)

model = load_model("model.h5")

print("Loaded model from disk")

y_pre = model.predict_classes(test)

print("Y_PRE shape",y_pre.shape)

with open(ans_csv, "w") as f:
	f.write("id,label\n")
	for i in range(len(y_pre)-1):
		f.write(str(i) + "," + str(y_pre[i]) + "\n")
	f.write(str(len(y_pre)-1) + "," + str(y_pre[len(y_pre)-1]))

