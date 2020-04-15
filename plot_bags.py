import numpy as np
import matplotlib.pyplot as plt
import os

folder = "real"

angX = np.loadtxt(folder + "/angX.csv", delimiter=",")
angY = np.loadtxt(folder + "/angY.csv", delimiter=",")
accX = np.loadtxt(folder + "/accX.csv", delimiter=",")
accY = np.loadtxt(folder + "/accY.csv", delimiter=",")
accZ = np.loadtxt(folder + "/accZ.csv", delimiter=",")
gyroX = np.loadtxt(folder + "/gyroX.csv", delimiter=",")
gyroY = np.loadtxt(folder + "/gyroY.csv", delimiter=",")
gyroZ = np.loadtxt(folder + "/gyroZ.csv", delimiter=",")
torque3 = np.loadtxt(folder + "/torque3.csv", delimiter=",")
torque4 = np.loadtxt(folder + "/torque4.csv", delimiter=",")

field_order = ['blanket','grass','rubber','carpet','mdf','tile']

def plot_csv(field):
  fig1, ax1 = plt.subplots(2, 1)
  ax1[0].plot(torque3[field,:])
  ax1[0].set_xlabel('Step')
  ax1[0].set_ylabel('Torque3[N]')
  ax1[1].plot(torque4[field,:])
  ax1[1].set_xlabel('Step')
  ax1[1].set_ylabel('Torque4[N]')
  ax1[0].set_title(field_order[field])

  fig2, ax2 = plt.subplots(8, 1)
  ax2[0].plot(gyroX[field,:])
  ax2[0].set_xlabel('Step')
  ax2[0].set_ylabel('GyroX')
  ax2[1].plot(gyroY[field,:])
  ax2[1].set_xlabel('Step')
  ax2[1].set_ylabel('GyroY')
  ax2[2].plot(gyroZ[field,:])
  ax2[2].set_xlabel('Step')
  ax2[2].set_ylabel('GyroZ')
  ax2[3].plot(accX[field,:])
  ax2[3].set_xlabel('Step')
  ax2[3].set_ylabel('AccX')
  ax2[4].plot(accY[field,:])
  ax2[4].set_xlabel('Step')
  ax2[4].set_ylabel('AccY')
  ax2[5].plot(accZ[field,:])
  ax2[5].set_xlabel('Step')
  ax2[5].set_ylabel('AccZ')
  ax2[6].plot(angX[field,:])
  ax2[6].set_xlabel('Step')
  ax2[6].set_ylabel('AngX')
  ax2[7].plot(angY[field,:])
  ax2[7].set_xlabel('Step')
  ax2[7].set_ylabel('AngY')
  ax2[0].set_title(field_order[field])
  plt.show()



for field in range(len(field_order)):
  plot_csv(field)
  
  