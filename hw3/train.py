import numpy as np
import pandas as pd
import sys
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,BatchNormalization,LeakyReLU
from keras.layers.convolutional import Conv2D,MaxPooling2D,AveragePooling2D,ZeroPadding2D
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
#from sklearn.metrics import confusion_matrix
#from keras.models import load_model
#import itertools
def parse_data(filename):
	
	train = pd.read_csv(filename)
	
	label = train[['label']].values
	train = train[['feature']].values
	
	print("LABEL shape:",label.shape)
	print("TRAIN shape:",train.shape)
	
	new_train = np.zeros((train.shape[0],48*48))
	for i in range(train.shape[0]):
		new_train[i,:] = train[i,0].split()
	
	new_train = new_train.reshape(new_train.shape[0],48,48,1).astype('float')

	label = to_categorical(label)
	return new_train,label
	

  

def dnn(train_x,train_y,test_x,test_y,batch_size,epochs):

	model = Sequential()
	model.add(Flatten(input_shape=(48,48,1)))
	model.add(Dense(512,activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1024,activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(2048,activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1024,activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(7,activation='softmax'))
	model.summary()
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	
	data_gen = ImageDataGenerator(
			rotation_range=10.0,
			width_shift_range=0.1,
			height_shift_range=0.1,
			#preprocessing_function=AHE
			)    

			
	data_gen.fit(train_x)
	score = model.evaluate(train_x,train_y)
	print ('\nTrain Acc:', score[1])
	score = model.evaluate(test_x,test_y)
	print ('\nVal Acc:', score[1]) 

	save = ModelCheckpoint('./dnn.h5', 
		monitor='val_acc', 
		verbose=1,
		save_best_only = True, 
		save_weights_only=False,
		mode='auto', 
		period=1) # save improved model only
	learning_rate_function = ReduceLROnPlateau(
		monitor='val_acc', 
		patience=4, 
		verbose=1,
		factor=0.5, 
		min_lr=0.000005)
	# training model
	model_result = model.fit_generator(
		data_gen.flow(train_x, train_y, batch_size), 
		validation_data=(test_x, test_y),
		steps_per_epoch=train_x.shape[0]//batch_size, 
		epochs=epochs,
		callbacks=[learning_rate_function, save])
		
	score = model.evaluate(train_x,train_y)
	print ('\nTrain Acc:', score[1])
	score = model.evaluate(test_x,test_y)
	print ('\nVal Acc:', score[1]) 
	plt.plot(model_result.history['acc'])
	plt.plot(model_result.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	# summarize history for loss
	plt.plot(model_result.history['loss'])
	plt.plot(model_result.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	print("Saved model to disk")
def cnn(train_x,train_y,test_x,test_y,batch_size,epochs):

	model = Sequential()
	model.add(Conv2D(64,kernel_size=(3,3),strides=(1,1),input_shape=(48,48,1),padding='same'))
	model.add(LeakyReLU(alpha=1./10))
	model.add(BatchNormalization())
	model.add(Conv2D(64,kernel_size=(3,3),strides=(1,1),padding='same'))
	model.add(LeakyReLU(alpha=1./10))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
	model.add(Dropout(0.2))
	
	#model.add(AveragePooling2D(pool_size=(2,2),strides=(2,2)))
	model.add(Conv2D(128,(3,3),padding='same')) 
	model.add(LeakyReLU(alpha=1./10))
	model.add(BatchNormalization())
	model.add(Conv2D(128,(3,3),padding='same')) 
	model.add(LeakyReLU(alpha=1./10))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
	model.add(Dropout(0.3))
	
	model.add(Conv2D(192,(3,3),padding='same')) 
	model.add(LeakyReLU(alpha=1./10))
	model.add(BatchNormalization())
	model.add(Conv2D(192,(3,3),padding='same')) 
	model.add(LeakyReLU(alpha=1./10))
	model.add(BatchNormalization())
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.3))
	
	model.add(Conv2D(256,(3,3),padding='same')) 
	model.add(LeakyReLU(alpha=1./10))
	model.add(BatchNormalization())
	model.add(Conv2D(256,(3,3),padding='same')) 
	model.add(LeakyReLU(alpha=1./10))
	model.add(BatchNormalization())
	model.add(MaxPooling2D())
	model.add(Dropout(0.3))
	
	model.add(Conv2D(512,(3,3),padding='same')) 
	model.add(LeakyReLU(alpha=1./10))
	model.add(BatchNormalization())
	model.add(Conv2D(512,(3,3),padding='same')) 
	model.add(LeakyReLU(alpha=1./10))
	model.add(BatchNormalization())
	model.add(MaxPooling2D())
	model.add(Dropout(0.3))
	
	model.add(Flatten())
	model.add(Dense(1024,activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(7,activation='softmax'))
	model.summary()
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	
	data_gen = ImageDataGenerator(
			rotation_range=10.0,
			width_shift_range=0.1,
			height_shift_range=0.1,
			#preprocessing_function=AHE
			)    

			
	data_gen.fit(train_x)
	
	#model.fit_generator(data_gen.flow(train_x,train_y,batch_size), validation_data=(test_x, test_y), steps_per_epoch=train_x.shape[0]//batch_size, epochs=epochs)
	#model.fit(train_x,train_y,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(test_x,test_y))
	
	save = ModelCheckpoint('./model.h5', 
		monitor='val_acc', 
		verbose=1,
		save_best_only = True, 
		save_weights_only=False,
		mode='auto', 
		period=1) # save improved model only
	learning_rate_function = ReduceLROnPlateau(
		monitor='val_acc', 
		patience=4, 
		verbose=1,
		factor=0.5, 
		min_lr=0.000005)
	# training model
	model_result = model.fit_generator(
		data_gen.flow(train_x, train_y, batch_size), 
		validation_data=(test_x, test_y),
		steps_per_epoch=train_x.shape[0]//batch_size, 
		epochs=epochs,
		callbacks=[learning_rate_function, save])
		
	score = model.evaluate(train_x,train_y)
	print ('\nTrain Acc:', score[1])
	score = model.evaluate(test_x,test_y)
	print ('\nVal Acc:', score[1]) 
	plt.plot(model_result.history['acc'])
	plt.plot(model_result.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	# summarize history for loss
	plt.plot(model_result.history['loss'])
	plt.plot(model_result.history['val_loss'])
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	plt.show()
	print("Saved model to disk")


def train_test_split(train,label,test_size):
	index = int(train.shape[0]*(1-test_size))
	
	train_x = train[:index,:]
	train_y = label[:index,:]
	
	test_x = train[index::,:]
	test_y = label[index::,:]
	
	return train_x,test_x,train_y,test_y

#def plot_conf(test_x,test_y):
#	model = load_model("model.h5")
#
#	print("Loaded model from disk")
#
#	y_pre = model.predict_classes(test_x)
#	conf_mat = confusion_matrix(test_y,y_pre)
#	plt.figure()
#	plot_confusion_matrix(conf_mat,classes=["Angry","Disgust","Fear","Happy","Sad","Suprise","Neural"])
#	plt.show()
#	
#def plot_confusion_matrix(cm, classes,
#                          normalize=False,
#                          title='Confusion matrix',
#                          cmap=plt.cm.Blues):
#    """
#    This function prints and plots the confusion matrix.
#    Normalization can be applied by setting `normalize=True`.
#    """
#    if normalize:
#        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#        print("Normalized confusion matrix")
#    else:
#        print('Confusion matrix, without normalization')
#
#    print(cm)
#
#    plt.imshow(cm, interpolation='nearest', cmap=cmap)
#    plt.title(title)
#    plt.colorbar()
#    tick_marks = np.arange(len(classes))
#    plt.xticks(tick_marks, classes, rotation=45)
#    plt.yticks(tick_marks, classes)
#
#    fmt = '.2f' if normalize else 'd'
#    thresh = cm.max() / 2.
#    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#        plt.text(j, i, format(cm[i, j], fmt),
#                 horizontalalignment="center",
#                 color="white" if cm[i, j] > thresh else "black")
#
#    plt.ylabel('True label')
#    plt.xlabel('Predicted label')
#    plt.tight_layout()	

	
def main():
	train_csv = sys.argv[1]
	train,label = parse_data(train_csv)
	train_x, test_x, train_y, test_y = train_test_split(train, label, test_size=0.2)
	batch_size = 128
	#plot_conf(test_x,test_y)
	epochs = 100
	num_classes = 7
	cnn(train_x,train_y,test_x,test_y,batch_size,epochs)

if __name__ == '__main__':
	os.environ['CUDA_VISIBLE_DEVICES'] = "0"
	config = tf.ConfigProto()
	config.gpu_options.allow_growth=True
	sess = tf.Session(config=config)
	main()