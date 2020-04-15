import tensorflow as tf
import tflearn
import numpy as np
import math
import random
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time



def get_one_hot(targets, nb_classes):
   res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
   return res.reshape(list(targets.shape)+[nb_classes])
 
def calc_accuracy(mat):
    c = 0
    s = np.zeros(6)
    for m in mat:
        acc = m[c]/np.sum(m,dtype=np.float32)
        s[c] = acc
        c = c+1
    print ("Mean: " + str(np.mean(s)) + " Variance: " + str(np.std(s)))

field_order = ['blanket','grass','rubber','carpet','mdf','tile']
#field_order = ['T1','T2','T3','T4','T5','T6']
sensor_order = ['angX','angY','accX','accY','accZ','gyroX','gyroY','gyroZ','torque3','torque4']


# Total: = 1000
num_train = 700
num_valid = 150
num_test = 150
num_total = num_train + num_valid + num_test
 
num_impact = 28
num_sensors = len(sensor_order)
num_fields = len(field_order)
 
 
all_data   = np.zeros(num_fields*num_sensors*num_impact*num_total).reshape([num_fields, num_sensors, num_impact*num_total])
all_output = np.zeros(num_fields*num_fields*num_total).reshape([num_fields, num_total, num_fields])


# Fill the input data
for sensor in sensor_order:
  #all_data[:,sensor_order.index(sensor),:] = np.loadtxt("real/" + sensor + ".csv", delimiter=",")
  #all_data[:,sensor_order.index(sensor),:] = np.hstack((np.loadtxt("real_new/" + sensor + ".csv", delimiter=","), np.loadtxt("real/" + sensor + ".csv", delimiter=",")))
  all_data[:,sensor_order.index(sensor),:] = np.loadtxt("sim/" + sensor + ".csv", delimiter=",")

all_data = all_data.reshape([num_fields, num_sensors, num_total, num_impact])

# Shuffle data for each sensor/field
# Fill one-hot expected outputs
for sensor in sensor_order:
  for field in field_order:
    np.random.shuffle(all_data[field_order.index(field),sensor_order.index(sensor),:,:])
    all_output[field_order.index(field),:,:] = get_one_hot(np.array([field_order.index(field)]),num_fields)


# Separate into training/validation/testing data input and output
x_train = all_data[:,:,:num_train,:]
x_valid = all_data[:,:,num_train:(num_train + num_valid),:]
x_test  = all_data[:,:,(num_train + num_valid):,:]


y_train = all_output[:,:num_train,:]
y_valid = all_output[:,num_train:(num_train + num_valid),:]
y_test  = all_output[:,(num_train + num_valid):,:]

# Standardization of data for each sensor type
x_train = x_train.transpose((1,0,2,3))
x_valid = x_valid.transpose((1,0,2,3))
x_test = x_test.transpose((1,0,2,3))

x_train_fit = x_train.reshape(num_sensors,num_fields*num_impact*num_train, -1)
x_valid_fit = x_valid.reshape(num_sensors,num_fields*num_impact*num_valid, -1)
x_test_fit  = x_test.reshape(num_sensors,num_fields*num_impact*num_test, -1)


scalers = {}
for i in range(num_sensors):
  scalers[i] = StandardScaler()
  x_train_fit[i, :, :] = scalers[i].fit_transform(x_train_fit[i ,:, :]) 
  x_valid_fit[i, :, :] = scalers[i].transform(x_valid_fit[i ,:, :]) 
  x_test_fit[i, :, :] = scalers[i].transform(x_test_fit[i ,:, :]) 
x_train = x_train_fit.reshape([num_sensors, num_fields, num_train, num_impact])
x_valid = x_valid_fit.reshape([num_sensors, num_fields, num_valid, num_impact])
x_test = x_test_fit.reshape([num_sensors, num_fields, num_test, num_impact])


x_train = x_train.transpose((1,2,0,3))
x_valid = x_valid.transpose((1,2,0,3))
x_test = x_test.transpose((1,2,0,3))

x_train = x_train.reshape([-1, num_sensors, num_impact, 1])
x_valid  = x_valid.reshape([-1, num_sensors, num_impact, 1])
x_test  = x_test.reshape([-1, num_sensors, num_impact, 1])
y_train = y_train.reshape([num_train*num_fields, num_fields])
y_valid = y_valid.reshape([num_valid*num_fields, num_fields])
y_test = y_test.reshape([num_test*num_fields, num_fields])

print (np.shape(x_train))
print (np.shape(x_valid))
print (np.shape(x_test))
print (np.shape(y_train))
print (np.shape(y_valid))
print (np.shape(y_test))


#Input Layer
network = tflearn.input_data(shape=[None, num_sensors, num_impact, 1], name='input')

#First Convolutional Layer
network = tflearn.conv_2d(network, 32, 5, activation='relu',regularizer="L2")
network = tflearn.dropout(network, 0.9)
network = tflearn.local_response_normalization(network)

#Second Convolutional Layer
network = tflearn.conv_2d(network, 64, 3, activation='relu', regularizer="L2")
network = tflearn.dropout(network, 0.9)
network = tflearn.local_response_normalization(network)

#First Fully Connected layer
network = tflearn.fully_connected(network, 64, activation='tanh', regularizer="L2")
network = tflearn.dropout(network, 0.9)

#Second Fully Connected layer
network = tflearn.fully_connected(network, 32, activation='tanh', regularizer="L2")
network = tflearn.dropout(network, 0.9)

# Output layer
network = tflearn.fully_connected(network,num_fields, activation='softmax')


#Training parameters
network = tflearn.regression(network, optimizer='adam', learning_rate=0.001,loss='categorical_crossentropy', name='target')

# Training
model = tflearn.DNN(network, tensorboard_verbose=0)
model.fit({'input': x_train}, {'target': y_train}, n_epoch=1000,
          validation_set=({'input': x_valid}, {'target': y_valid}),
snapshot_step=100,batch_size=32, show_metric=True,shuffle=True, run_id='convnet_terrain')




comp = np.zeros(shape = (np.shape(x_test)[0],2),dtype=np.int)
y_true = np.zeros(np.shape(x_test)[0])
y_pred = np.zeros(np.shape(x_test)[0])
print (np.shape(comp))

total_time = 0

for i in range(0,np.shape(x_test)[0]):
    xi = x_test[i,:,:,:]
    yi = y_test[i,:]
    xi = xi.reshape([-1,num_sensors,num_impact,1])
    
    t = time.process_time()
    pred  = model.predict(xi)
    total_time = total_time + (time.process_time() - t)
    
    comp[i,0] = str(np.argmax(yi))
    comp[i,1] = str(np.argmax(pred))
    y_true[i] = comp[i,0]
    y_pred[i] = comp[i,1]


mat = confusion_matrix(y_true,y_pred)
print (mat)
calc_accuracy(mat)
print(classification_report(y_true, y_pred, target_names=field_order, digits = 10))

print ("Mean time for predition: " + str(total_time/num_test))
       
model.summary()       
