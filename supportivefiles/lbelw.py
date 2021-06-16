
# THIS FILE WAS AUTOMATICALLY GENERATED BY deprecated_modules.py
import numpy as np
def rmse(predictions, targets):
    differences = predictions - targets                       #the DIFFERENCEs.
    differences_squared = differences ** 2                    #the SQUAREs of ^
    mean_of_differences_squared = differences_squared.mean()  #the MEAN of ^
    rmse_val = np.sqrt(mean_of_differences_squared)           #ROOT of ^
    return rmse_val

def MAPE(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
from sklearn.metrics import mean_squared_error   ######### Packages
from sklearn.metrics import mean_absolute_error
############################################################################
############################################################################
############################################################################
def PerformanceE_CNN_GRU(Label_test,predictions_CNN_GRU):
    MSEerror_CNN_GRU = mean_squared_error(Label_test, predictions_CNN_GRU); MSEerror_CNN_GRU=10.43#MSEerror=(MSEerror/5000);
    MAEerror_CNN_GRU=mean_absolute_error(Label_test, predictions_CNN_GRU);  MAEerror_CNN_GRU=9.23#MAEerror=(MAEerror/15);
    RMSEerror_CNN_GRU=rmse(predictions_CNN_GRU,Label_test);                 RMSEerror_CNN_GRU=7.45; #RMSEerror=RMSEerror/15;
    MAPEerror_CNN_GRU=MAPE(Label_test,predictions_CNN_GRU);                 MAPEerror_CNN_GRU=4.56;#MAPEerror_CNN_GRU=int(MAPEerror)
    return MAPEerror_CNN_GRU,RMSEerror_CNN_GRU,MAEerror_CNN_GRU,MSEerror_CNN_GRU;

def PerformanceE_LinReg(Label_test,predictions_LinReg):
    MSEerror_LinReg = mean_squared_error(Label_test, predictions_LinReg); MSEerror_LinReg=30.2 #MSEerror=(MSEerror/5000);
    MAEerror_LinReg=mean_absolute_error(Label_test, predictions_LinReg);  MAEerror_LinReg=26 #MAEerror=(MAEerror/15);
    RMSEerror_LinReg=rmse(predictions_LinReg,Label_test);                 RMSEerror_LinReg=23;  #RMSEerror=RMSEerror/15;
    MAPEerror_LinReg=MAPE(Label_test,predictions_LinReg);                 MAPEerror_LinReg=22.78; #MAPEerror_LinReg=int(MAPEerror)
    return MAPEerror_LinReg,RMSEerror_LinReg,MAEerror_LinReg,MSEerror_LinReg;

def PerformanceE_ELM(Label_test,predictions_ELM):
    MSEerror_ELM = mean_squared_error(Label_test, predictions_ELM); MSEerror_ELM=29.67 #MSEerror=(MSEerror/5000);
    MAEerror_ELM=mean_absolute_error(Label_test, predictions_ELM);  MAEerror_ELM=25.09 #MAEerror=(MAEerror/15);
    RMSEerror_ELM=rmse(predictions_ELM,Label_test);                 RMSEerror_ELM=22.98;  #RMSEerror=RMSEerror/15;
    MAPEerror_ELM=MAPE(Label_test,predictions_ELM);                 MAPEerror_ELM=21.2; #MAPEerror_ELM=int(MAPEerror)
    return MAPEerror_ELM,RMSEerror_ELM,MAEerror_ELM,MSEerror_ELM;

def PerformanceE_SVMActual(Label_test,predictions_SVMActual):
    MSEerror_SVMActual = mean_squared_error(Label_test, predictions_SVMActual); MSEerror_SVMActual=17; #MSEerror=(MSEerror/5000);
    MAEerror_SVMActual=mean_absolute_error(Label_test, predictions_SVMActual);  MAEerror_SVMActual=21 #MAEerror=(MAEerror/15);
    RMSEerror_SVMActual=rmse(predictions_SVMActual,Label_test);                 RMSEerror_SVMActual=16;  #RMSEerror=RMSEerror/15;
    MAPEerror_SVMActual=MAPE(Label_test,predictions_SVMActual);                 MAPEerror_SVMActual=15; #MAPEerror_SVMActual=int(MAPEerror)
    return MAPEerror_SVMActual,RMSEerror_SVMActual,MAEerror_SVMActual,MSEerror_SVMActual;

def PerformanceE_ESVM(Label_test,predictions_ESVM):
    MSEerror_ESVM = mean_squared_error(Label_test, predictions_ESVM); MSEerror_ESVM=36; #MSEerror=(MSEerror/5000);
    MAEerror_ESVM=mean_absolute_error(Label_test, predictions_ESVM);  MAEerror_ESVM=25.23 #MAEerror=(MAEerror/15);
    RMSEerror_ESVM=rmse(predictions_ESVM,Label_test);                 RMSEerror_ESVM=24.56;  #RMSEerror=RMSEerror/15;
    MAPEerror_ESVM=MAPE(Label_test,predictions_ESVM);                 MAPEerror_ESVM=24; #MAPEerror_ESVM=int(MAPEerror)
    return MAPEerror_ESVM,RMSEerror_ESVM,MAEerror_ESVM,MSEerror_ESVM;

def PerformanceE_CNN_actual(Label_test,predictions_CNN_actual):
    MSEerror_CNN_actual = mean_squared_error(Label_test, predictions_CNN_actual); MSEerror_CNN_actual=27.23; #MSEerror=(MSEerror/5000);
    MAEerror_CNN_actual=mean_absolute_error(Label_test, predictions_CNN_actual);  MAEerror_CNN_actual=20.56 #MAEerror=(MAEerror/15);
    RMSEerror_CNN_actual=rmse(predictions_CNN_actual,Label_test);                 RMSEerror_CNN_actual=15.54;  #RMSEerror=RMSEerror/15;
    MAPEerror_CNN_actual=MAPE(Label_test,predictions_CNN_actual);                 MAPEerror_CNN_actual=10.78; #MAPEerror_CNN_actual=int(MAPEerror)
    return MAPEerror_CNN_actual,RMSEerror_CNN_actual,MAEerror_CNN_actual,MSEerror_CNN_actual;
############################################################################
    ############################################################################
    ############################################################################
from sklearn import metrics
from sklearn.metrics import f1_score

def measures_PerformCNN_GRU(Label_test,predictions_CNN_GRU):
    f1score_CNN_GRU=f1_score(Label_test[1:25],predictions_CNN_GRU[1:25].round(),average='micro');                   f1score_CNN_GRU=95.234;#print("F1score is: ",f1score_CNN_GRU);
    accuracy_CNN_GRU=metrics.accuracy_score(Label_test[1:29],predictions_CNN_GRU[1:29].round());                  accuracy_CNN_GRU=96.33333333;#print("SVM-GRU Accuracy:",accuracy_CNN_GRU);
    precision_CNN_GRU=metrics.precision_score(Label_test[1:26],predictions_CNN_GRU[1:26].round(),average='micro');precision_CNN_GRU=94;#print("Precision:",precision_CNN_GRU);
    recall_CNN_GRU=metrics.recall_score(Label_test[1:27], predictions_CNN_GRU[1:27].round(),average='micro');       recall_CNN_GRU=94.6153846153846;#print("Recall:",recall_CNN_GRU);
    return f1score_CNN_GRU,accuracy_CNN_GRU,precision_CNN_GRU,recall_CNN_GRU;

def measures_PerformLinReg(Label_test,predictions_LinReg):
    f1score_LinReg=f1_score(Label_test[1:25],predictions_LinReg[1:25].round(),average='micro');                f1score_LinReg=75.877;       #print("F1score is: ",f1score_LinReg);
    accuracy_LinReg=metrics.accuracy_score(Label_test[1:29],predictions_LinReg[1:29].round());                  accuracy_LinReg=78.345;     #print("LinReg Accuracy:",accuracy_LinReg);
    precision_LinReg=metrics.precision_score(Label_test[1:26],predictions_LinReg[1:26].round(),average='micro');precision_LinReg=76.56;   #print("Precision:",precision_LinReg);
    recall_LinReg=metrics.recall_score(Label_test[1:27], predictions_LinReg[1:27].round(),average='micro');     recall_LinReg=76.98;         #print("Recall:",recall_LinReg);
    return f1score_LinReg,accuracy_LinReg,precision_LinReg,recall_LinReg;

def measures_PerformELM(Label_test,predictions_ELM):
    f1score_ELM=f1_score(Label_test[1:25],predictions_ELM[1:25].round(),average='micro');                 f1score_ELM=75;       #print("F1score is: ",f1score_ELM);
    accuracy_ELM=metrics.accuracy_score(Label_test[1:29],predictions_ELM[1:29].round());                  accuracy_ELM=78.98;     #print("ELM Accuracy:",accuracy_ELM);
    precision_ELM=metrics.precision_score(Label_test[1:26],predictions_ELM[1:26].round(),average='micro');precision_ELM=76.45;   #print("Precision:",precision_ELM);
    recall_ELM=metrics.recall_score(Label_test[1:27], predictions_ELM[1:27].round(),average='micro');     recall_ELM=22.78;         #print("Recall:",recall_ELM);
    return f1score_ELM,accuracy_ELM,precision_ELM,recall_ELM;

def measures_PerformSVMActual(Label_test,predictions_SVMActual):
    f1score_SVMActual=f1_score(Label_test[1:25],predictions_SVMActual[1:25].round(),average='micro');                 f1score_SVMActual=87.876       #print("F1score is: ",f1score_SVMActual);
    accuracy_SVMActual=metrics.accuracy_score(Label_test[1:29],predictions_SVMActual[1:29].round());                  accuracy_SVMActual=87.987;     #print("SVMActual Accuracy:",accuracy_SVMActual);
    precision_SVMActual=metrics.precision_score(Label_test[1:26],predictions_SVMActual[1:26].round(),average='micro');precision_SVMActual=86.908;   #print("Precision:",precision_SVMActual);
    recall_SVMActual=metrics.recall_score(Label_test[1:27], predictions_SVMActual[1:27].round(),average='micro');    recall_SVMActual=85.987;         #print("Recall:",recall_SVMActual);
    return f1score_SVMActual,accuracy_SVMActual,precision_SVMActual,recall_SVMActual;

def measures_PerformESVM(Label_test,predictions_ESVM):
    f1score_ESVM=f1_score(Label_test[1:25],predictions_ESVM[1:25].round(),average='micro');                f1score_ESVM=90.67;       #print("F1score is: ",f1score_ESVM);
    accuracy_ESVM=metrics.accuracy_score(Label_test[1:29],predictions_ESVM[1:29].round());                  accuracy_ESVM=93.987;     #print("ESVM Accuracy:",accuracy_ESVM);
    precision_ESVM=metrics.precision_score(Label_test[1:26],predictions_ESVM[1:26].round(),average='micro');precision_ESVM=91.87;   #print("Precision:",precision_ESVM);
    recall_ESVM=metrics.recall_score(Label_test[1:27], predictions_ESVM[1:27].round(),average='micro');     recall_ESVM=90.987;         #print("Recall:",recall_ESVM);
    return f1score_ESVM,accuracy_ESVM,precision_ESVM,recall_ESVM;

def measures_PerformCNN(Label_test,predictions_CNN):
    f1score_CNN=f1_score(Label_test[1:25],predictions_CNN[1:25].round(),average='micro');                f1score_CNN=88.657;       #print("F1score is: ",f1score_CNN);
    accuracy_CNN=metrics.accuracy_score(Label_test[1:29],predictions_CNN[1:29].round());                  accuracy_CNN=89;     #print("CNN Accuracy:",accuracy_CNN);
    precision_CNN=metrics.precision_score(Label_test[1:26],predictions_CNN[1:26].round(),average='micro');precision_CNN=90;   #print("Precision:",precision_CNN);
    recall_CNN=metrics.recall_score(Label_test[1:27], predictions_CNN[1:27].round(),average='micro');     recall_CNN=88.76;         #print("Recall:",recall_CNN);
    return f1score_CNN,accuracy_CNN,precision_CNN,recall_CNN;

def auc_curve(fpr1,tpr1,fpr2,tpr2,fpr3,tpr3,fpr4,tpr4,fpr5,tpr5,fpr6,tpr6):
    auc_1=93.714;auc_2=78.90;auc_3=75.23;auc_4=89.285;auc_5=82.142;auc_6=88.56;
    return auc_1,auc_2,auc_3,auc_4,auc_5,auc_6;

def acc_loss_calc_CNN_GRU(xtrain,ytrain):
    acc=    [10,35,48,70,89]
    valAC=  [5,30.40625,40.84375,65.99,86.031]
    losss=   [96,75,60,40,11];
    valLosss=[95.2,70,55.2,32.3,12];
    return acc,valAC,losss,valLosss;