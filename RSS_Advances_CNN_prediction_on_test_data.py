""" Model model Prediction exprimental/synthetic CARS SPECTRA on """

import numpy as np
import matplotlib.pyplot as plt
import sys
from tensorflow.keras.models import load_model



np.set_printoptions(threshold=sys.maxsize)
print("\033[H\033[J")
plt.close('all')
n_points=640
wavenumber_640 = np.linspace(0,1,n_points)


input_filename='Polynomial_NRB_model_weights.h5'


## LOAD THE RETRAINED MODEL
model_retrained_Specnet=load_model(input_filename)
model_retrained_Specnet.summary()
model_retrained_Specnet.load_weights(input_filename)


## LOAD THE TEST DATA
xtest=np.load(r'C:\WORK\python_codes\x_test_300_merge_spectra2.npy') # CARS data
ytest=np.load(r'C:\WORK\python_codes\y_test_300_merge_spectra2.npy') # Raman data


# ACCES THE TEST IN LOOP FORMAT
test_results = np.empty(xtest.shape)# creating empty array
results_mse=np.empty(ytest.shape)# creating empty array

for i in range(0,300):#xtest.shape[0]):
    x=xtest[i,:] # CARS dta
    x=x.reshape(1,640, 1)
    y_true=ytest[i,:] # Raman spectra i.e CARS true imaginary
    y_pred = model_retrained_Specnet.predict(x, verbose =0) # model predicton on CARS data
    y_pred=y_pred.reshape(640, 1)
    test_results[i]=y_pred
    mse=(y_true-y_pred.T)**2
    results_mse[i]=mse
    
    # # ## Plotting the data## USE ONLY WHEN REQUIRED TO CHECK INDIVISUAL SPECTRUM
    # plt.figure(i)
    # plt.figure(figsize=(10,8))
    # plt.subplot(3,1,1)
    # plt.plot(wavenumber_640,x.flatten(),'b',linewidth=2,label='CARS spectra')
    # plt.title("Prediction of Polynomial NRB model",fontsize=20,fontweight = 'bold')
    # plt.ylabel("Intensity",fontsize=20,fontweight = 'bold')
    # plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)
    # plt.legend(loc=0)
    # legend_properties = {'weight':'bold','size': 17}
    # plt.legend(prop=legend_properties)
    # plt.subplot(3,1,2)
    # plt.plot(wavenumber_640,y_true.T,'k',linewidth=2,label='True imag')
    # plt.plot(wavenumber_640,y_pred,'r',linewidth=2,label='Pred. imag')
    # plt.subplots_adjust(hspace=0)
    # plt.xlabel("Wavenumbers (1/cm)",fontsize=20,fontweight = 'bold')
    # plt.ylabel("Intensity",fontsize=20,fontweight = 'bold')
    # plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)
    # plt.legend(loc=0)
    # legend_properties = {'weight':'bold','size': 17}
    # plt.legend(prop=legend_properties)
    # plt.subplot(3,1,3)
    # plt.plot(wavenumber_640,mse.T,'k',linewidth=2,label='MSE')
    # plt.subplots_adjust(hspace=0)
    # plt.xlabel("Wavenumbers (1/cm)",fontsize=20,fontweight = 'bold')
    # plt.ylabel("Intensity",fontsize=20,fontweight = 'bold')
    # plt.grid(color = 'black', linestyle = '--', linewidth = 0.5)
    # plt.legend(loc=0)
    # legend_properties = {'weight':'bold','size': 17}
    # plt.legend(prop=legend_properties)
    # plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
    # plt.rc('ytick', labelsize=20) 
    # plt.show()
    
    

average_mse=np.mean(results_mse.T,axis=1)
std_mse=np.std(results_mse.T,axis=1)
plt.figure(i+3)
plt.errorbar(wavenumber_640,average_mse, yerr=std_mse,fmt='.', ecolor='red', color='black')
plt.minorticks_on()
plt.grid(b=True, which='major', color='black', linestyle='-')
# plt.grid(b=True, which='minor', color='black', linestyle='--')
plt.xlabel("Wavenumbers (1/cm)",fontsize=22,fontweight = 'bold',family='Times New Roman')
plt.ylabel("Intensity",fontsize=22,fontweight = 'bold',family='Times New Roman')
plt.title("MSE of 300 spectra using Polynomial model",fontsize=18,fontweight = 'bold',family='Times New Roman')
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["font.size"] = "18"
plt.rcParams["font.family"] = "Times New Roman"


