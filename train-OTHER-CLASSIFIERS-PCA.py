import time
import numpy as np
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

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
sensor_order = ['angX','angY','accX','accY','accZ','gyroX','gyroY','gyroZ','torque3','torque4']

# Total: 800 + 200 = 1000
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
  #all_data[:,sensor_order.index(sensor),:] = np.loadtxt("sim/" + sensor + ".csv", delimiter=",")

  all_data[:,sensor_order.index(sensor),:] = np.loadtxt("real/" + sensor + ".csv", delimiter=",")

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


x_train = x_train.reshape([-1, num_sensors*num_impact])
x_valid  = x_valid.reshape([-1, num_sensors*num_impact])
x_test  = x_test.reshape([-1, num_sensors * num_impact])
y_train = y_train.reshape([num_train*num_fields, num_fields])
y_valid = y_valid.reshape([num_valid*num_fields, num_fields])
y_test = y_test.reshape([num_test*num_fields, num_fields])



y_train = [np.where(r==1)[0][0] for r in y_train]
y_valid = [np.where(r==1)[0][0] for r in y_valid]
y_test = [np.where(r==1)[0][0] for r in y_test]

#creating PCA vectors

pca = PCA (0.95)
pca.fit(x_train)

x_train_PCA = pca.transform(x_train)
x_valid_PCA = pca.transform(x_valid)
x_test_PCA = pca.transform(x_test)



# TESTING ALL CLASSIFIERS


print("NEAREST CENTROID without PCA")

func = NearestCentroid()
func.fit(x_train, y_train)

t = time.process_time()
pred_y = func.predict(x_valid)
time_elapsed = time.process_time() - t

mat = confusion_matrix(y_valid,pred_y)
print (mat)
calc_accuracy(mat)
print(classification_report(y_valid,pred_y,target_names=field_order, digits = 5))
print ("Classification time : " + str(time_elapsed/num_valid)+ '\n')

print("NEAREST CENTROID with PCA")

func = NearestCentroid()
func.fit(x_train_PCA, y_train)

t = time.process_time()
pred_y = func.predict(x_valid_PCA)
time_elapsed = time.process_time() - t

mat = confusion_matrix(y_valid,pred_y)
print (mat)
calc_accuracy(mat)
print(classification_report(y_valid,pred_y,target_names=field_order, digits = 5))
print ("Classification time : " + str(time_elapsed/num_valid)+ '\n')

print("KNN without PCA")

func = KNeighborsClassifier(n_neighbors=5)
func.fit(x_train, y_train)

t = time.process_time()
pred_y = func.predict(x_valid)
time_elapsed = time.process_time() - t

mat = confusion_matrix(y_valid,pred_y)
print (mat)
calc_accuracy(mat)
print(classification_report(y_valid,pred_y,target_names=field_order, digits = 5))
print ("Classification time : " + str(time_elapsed/num_valid)+ '\n')


print("KNN with PCA")

func = KNeighborsClassifier(n_neighbors=5)
func.fit(x_train_PCA, y_train)

t = time.process_time()
pred_y = func.predict(x_valid_PCA)
time_elapsed = time.process_time() - t

mat = confusion_matrix(y_valid,pred_y)
print (mat)
calc_accuracy(mat)
print(classification_report(y_valid,pred_y,target_names=field_order, digits = 5))
print ("Classification time : " + str(time_elapsed/num_valid)+ '\n')


print("DECISION TREE without PCA")

func = DecisionTreeClassifier()
func.fit(x_train, y_train)

t = time.process_time()
pred_y = func.predict(x_valid)
time_elapsed = time.process_time() - t

mat = confusion_matrix(y_valid,pred_y)
print (mat)
calc_accuracy(mat)
print(classification_report(y_valid,pred_y,target_names=field_order, digits = 5))
print ("Classification time : " + str(time_elapsed/num_valid)+ '\n')



print("DECISION TREE with PCA")

func = DecisionTreeClassifier()
func.fit(x_train_PCA, y_train)

t = time.process_time()
pred_y = func.predict(x_valid_PCA)
time_elapsed = time.process_time() - t

mat = confusion_matrix(y_valid,pred_y)
print (mat)
calc_accuracy(mat)
print(classification_report(y_valid,pred_y,target_names=field_order, digits = 5))
print ("Classification time : " + str(time_elapsed/num_valid)+ '\n')



print("ADABOOST without PCA")

func = AdaBoostClassifier(n_estimators=100)
func.fit(x_train, y_train)

t = time.process_time()
pred_y = func.predict(x_valid)
time_elapsed = time.process_time() - t

mat = confusion_matrix(y_valid,pred_y)
print (mat)
calc_accuracy(mat)
print(classification_report(y_valid,pred_y,target_names=field_order, digits = 5))
print ("Classification time : " + str(time_elapsed/num_valid)+ '\n')


print("ADABOOST with PCA")

func = AdaBoostClassifier(n_estimators=100)
func.fit(x_train_PCA, y_train)

t = time.process_time()
pred_y = func.predict(x_valid_PCA)
time_elapsed = time.process_time() - t

mat = confusion_matrix(y_valid,pred_y)
print (mat)
calc_accuracy(mat)
print(classification_report(y_valid,pred_y,target_names=field_order, digits = 5))
print ("Classification time : " + str(time_elapsed/num_valid)+ '\n')

print("SGD without PCA")

func = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
func.fit(x_train, y_train)

t = time.process_time()
pred_y = func.predict(x_valid)
time_elapsed = time.process_time() - t

mat = confusion_matrix(y_valid,pred_y)
print (mat)
calc_accuracy(mat)
print(classification_report(y_valid,pred_y,target_names=field_order, digits = 5))
print ("Classification time : " + str(time_elapsed/num_valid)+ '\n')


print("SGD with PCA")

func = SGDClassifier(loss="hinge", penalty="l2", max_iter=5)
func.fit(x_train_PCA, y_train)

t = time.process_time()
pred_y = func.predict(x_valid_PCA)
time_elapsed = time.process_time() - t

mat = confusion_matrix(y_valid,pred_y)
print (mat)
calc_accuracy(mat)
print(classification_report(y_valid,pred_y,target_names=field_order, digits = 5))
print ("Classification time : " + str(time_elapsed/num_valid)+ '\n')

print("Random Forrest without PCA")

func = RandomForestClassifier(n_estimators=100)
func.fit(x_train, y_train)

t = time.process_time()
pred_y = func.predict(x_valid)
time_elapsed = time.process_time() - t

mat = confusion_matrix(y_valid,pred_y)
print (mat)
calc_accuracy(mat)
print(classification_report(y_valid,pred_y,target_names=field_order, digits = 5))
print ("Classification time : " + str(time_elapsed/num_valid)+ '\n')


print("Random Forrest with PCA")

func = RandomForestClassifier(n_estimators=100)
func.fit(x_train_PCA, y_train)

t = time.process_time()
pred_y = func.predict(x_valid_PCA)
time_elapsed = time.process_time() - t

mat = confusion_matrix(y_valid,pred_y)
print (mat)
calc_accuracy(mat)
print(classification_report(y_valid,pred_y,target_names=field_order, digits = 5))
print ("Classification time : " + str(time_elapsed/num_valid)+ '\n')

print("SVM LINEAR without PCA")

func = svm.SVC(kernel = "linear", gamma = 'auto')
func.fit(x_train, y_train)

t = time.process_time()
pred_y = func.predict(x_valid)
time_elapsed = time.process_time() - t

mat = confusion_matrix(y_valid,pred_y)
print (mat)
calc_accuracy(mat)
print(classification_report(y_valid,pred_y,target_names=field_order, digits = 5))
print ("Classification time : " + str(time_elapsed/num_valid)+ '\n')


print("SVM LINEAR with PCA")

func = svm.SVC(kernel = "linear", gamma = 'auto')
func.fit(x_train_PCA, y_train)

t = time.process_time()
pred_y = func.predict(x_valid_PCA)
time_elapsed = time.process_time() - t

mat = confusion_matrix(y_valid,pred_y)
print (mat)
calc_accuracy(mat)
print(classification_report(y_valid,pred_y,target_names=field_order, digits = 5))
print ("Classification time : " + str(time_elapsed/num_valid)+ '\n')

print("SVM POLY without PCA")

func = svm.SVC(kernel = "poly", gamma = 'auto')
func.fit(x_train, y_train)

t = time.process_time()
pred_y = func.predict(x_valid)
time_elapsed = time.process_time() - t

mat = confusion_matrix(y_valid,pred_y)
print (mat)
calc_accuracy(mat)
print(classification_report(y_valid,pred_y,target_names=field_order, digits = 5))
print ("Classification time : " + str(time_elapsed/num_valid)+ '\n')


print("SVM POLY with PCA")

func = svm.SVC(kernel = "poly", gamma = 'auto')
func.fit(x_train_PCA, y_train)

t = time.process_time()
pred_y = func.predict(x_valid_PCA)
time_elapsed = time.process_time() - t

mat = confusion_matrix(y_valid,pred_y)
print (mat)
calc_accuracy(mat)
print(classification_report(y_valid,pred_y,target_names=field_order, digits = 5))
print ("Classification time : " + str(time_elapsed/num_valid)+ '\n')

print("SVM RBF without PCA")

func = svm.SVC(kernel = "rbf", gamma = 'auto')
func.fit(x_train, y_train)

t = time.process_time()
pred_y = func.predict(x_valid)
time_elapsed = time.process_time() - t

mat = confusion_matrix(y_valid,pred_y)
print (mat)
calc_accuracy(mat)
print(classification_report(y_valid,pred_y,target_names=field_order, digits = 5))
print ("Classification time : " + str(time_elapsed/num_valid)+ '\n')


print("SVM RBF with PCA")

func = svm.SVC(kernel = "rbf", gamma = 'auto')
func.fit(x_train_PCA, y_train)

t = time.process_time()
pred_y = func.predict(x_valid_PCA)
time_elapsed = time.process_time() - t

mat = confusion_matrix(y_valid,pred_y)
print (mat)
calc_accuracy(mat)
print(classification_report(y_valid,pred_y,target_names=field_order, digits = 5))
print ("Classification time : " + str(time_elapsed/num_valid)+ '\n')
