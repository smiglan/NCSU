# -*- coding: utf-8 -*-
"""mogfinal.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1GBG7LsYleu9BjbYGRxMGy_7sTs_aKsSJ
"""

from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)
path = '/content/gdrive/My Drive/ECE763/'
import numpy as np
import cv2
from scipy.stats import multivariate_normal
from scipy.special import psi, gammaln
from scipy.optimize import fminbound
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import random
from sklearn.preprocessing import StandardScaler
from numpy import matlib


np.random.seed(2)
facetrain= np.load(path+'MyData4/ftrain.npy')
nonfacetrain= np.load(path+'MyData4/nftrain.npy')
facetest= np.load(path+'MyData4/ftest.npy')
nonfacetest= np.load(path+'MyData4/nftest.npy')

def apply_pca_and_standardize(data,size):
    pca = PCA(n_components=size)
    pca.fit(data)
    data_pca = pca.transform(data)
    scaler = StandardScaler()
    scaler.fit(data_pca)
    data_std = scaler.transform(data_pca)
    return data_std,pca

size = 30
# pca_face = PCA(size)
facetrain_pca,pcaft = apply_pca_and_standardize(facetrain,size) 
nonfacetrain_pca,pcanft = apply_pca_and_standardize(nonfacetrain,size) 
facetest_pca,pcaftest  = apply_pca_and_standardize(facetest,size)
nonfacetest_pca,pcaftrain  = apply_pca_and_standardize(nonfacetest,size)

def fit_mog(x,K,precision):
  lbda = 1/K*np.ones(K).transpose()
  lbda = np.reshape(lbda,(K,1))
  I = x.shape[0]
  K_random_unique_integers = np.random.permutation(I)[0:K]
  mu = x[K_random_unique_integers,:]
  dimensionality = x.shape[1]
  dataset_mean = np.sum(x,axis=0) / I
  dataset_mean = np.reshape(dataset_mean,(1,dataset_mean.shape[0]))
  dataset_variance = np.zeros((dimensionality, dimensionality))

  for i in range(I):
    mat = x[i,:]-dataset_mean
    mat = np.matmul(np.transpose(mat),mat)
    dataset_variance= dataset_variance+mat
  dataset_variance = dataset_variance/I

  sig = []
  for i in range(K):
    sig.append(dataset_variance)
  sig = np.array(sig)


  iterations = 0   
  previous_L = 1000000
  while True:
    l = np.zeros ((I,K))
    r = np.zeros ((I,K))
    for k in range(K):
        l[:,k] = lbda[k] * multivariate_normal.pdf(x, mu[k,:], sig[k],allow_singular=True)


    s = np.sum(l,axis=1)
    for i in range(I):
      r[i,:] = l[i,:] / s[i];

    r_summed_rows = np.sum(r,axis=0)
    r_summed_all = np.sum(r_summed_rows,axis=0)

    for k in range(K):
      lbda[k] = r_summed_rows[k] / r_summed_all
      new_mu = np.zeros((1,dimensionality))
      for i in range(I):
          new_mu = new_mu + r[i,k]*x[i,:]
      mu[k,:] = new_mu/r_summed_rows[k]
      new_sigma = np.zeros((dimensionality,dimensionality))
      for i in range(I):
          mat = x[i,:] - mu[k,:]
          mat = np.reshape(mat,(1,mat.shape[0]))
          mat = r[i,k]*np.matmul(np.transpose(mat),mat)
          new_sigma = new_sigma + mat
      sig[k] = new_sigma/ r_summed_rows[k]

    temp = np.zeros((I,K));
    for k in range(K):
        temp[:,k] = lbda[k] * multivariate_normal.pdf(x, mu[k,:], sig[k],allow_singular=True)
    temp = np.sum(temp,axis=1)
    temp = np.log(temp)      
    L = sum(temp)
    #print('Log-Likelihood: ', L)

    iterations = iterations + 1      
    #print('Iteration number: ',iterations)
    print(np.abs(L - previous_L))
    if np.abs(L - previous_L) < precision:
        break
    previous_L = L
  return lbda,mu,sig

dimm = 20
Xf = facetrain_pca
precision = 0.01
K = 3
af,bf,cf = fit_mog(Xf,K,precision)
Xnf = nonfacetrain_pca
anf,bnf,cnf = fit_mog(Xnf,K,precision)

mu_face    = facetrain_pca.mean(axis=0)
mu_nonface = nonfacetrain_pca.mean(axis=0)
var_face = np.cov(facetrain_pca, rowvar=False, bias=1, ddof=None)
var_face = np.diagonal(var_face)
var_face = np.diag(var_face,0)
var_nonface = np.cov(nonfacetrain_pca, rowvar=False, bias=1, ddof=None)
var_nonface = np.diagonal(var_nonface)
var_nonface = np.diag(var_nonface,0)

Prn2 = np.zeros([100,1])
prob_fpf = np.zeros([100,1])
for k in range(K):
    Prn2 = multivariate_normal.pdf(facetest_pca, bf[k,:], cf[k],allow_singular=True)
    Prn2 = np.reshape(Prn2,(-1,1))
    prob_fpf = prob_fpf + (af[k] * Prn2)
prob_fpnf = multivariate_normal.pdf(facetest_pca, mean= mu_nonface, cov=var_nonface)
P_face = prob_fpf / (prob_fpf+ np.reshape(prob_fpnf,(100,1)))
True_positive = np.sum(P_face[:] >= 0.5)
False_negative = 100 - True_positive

Prn3 = np.zeros([100,1])
prob_nfpnf = np.zeros([100,1])
for k in range(K):
    Prn3 = multivariate_normal.pdf(nonfacetest_pca, bnf[k,:], cnf[k],allow_singular=True)
    Prn3 = np.reshape(Prn3,(-1,1))
    prob_nfpnf = prob_nfpnf + (anf[k] * Prn3)
prob_nfpf = multivariate_normal.pdf(nonfacetest_pca, mean= mu_face, cov=var_face)
P_nonface = prob_nfpnf / (prob_nfpnf+ np.reshape(prob_nfpf,(100,1)))
True_negative = np.sum(P_nonface[:] >= 0.5)
False_positive = 100 - True_negative


fpr =  False_positive/ (False_positive + True_negative)
fnr =  False_negative / (False_negative + True_positive)
miss = ( False_positive+ False_negative) / 200

print('False Positive Rate:',fpr)
print('False Negative Rate:',fnr)
print('Miss Classification Rate:', miss)

a = 0
for i in P_face:
  if i>=0.5:
    #print(i)
    a= a+1
print(a)
a = 0
for i in P_nonface:
  if i>=0.5:
    #print(i)
    a= a+1
print(a)

labels = np.array([np.ones(100),np.zeros(100)])
labels = np.reshape(labels,(200))
X_test_roc = np.array([facetest_pca,nonfacetest_pca])
X_test_roc = np.reshape(X_test_roc,(200,size))

Prn = np.zeros([200,1])
sum1 = np.zeros([200,1])
for k in range(K):
    Prn = multivariate_normal.pdf(X_test_roc, bf[k,:], cf[k],allow_singular=True)
    Prn = np.reshape(Prn,(-1,1))
    sum1 = sum1 + (af[k] * Prn)

sum2 = multivariate_normal.pdf(X_test_roc, mean=mu_nonface, cov=var_nonface,allow_singular=True)
P_Roc = sum1 / (sum1+ np.reshape(sum2,(200,1)))

fpr, tpr, thresholds = metrics.roc_curve(labels,P_Roc)
roc_auc = roc_auc_score(labels,P_Roc)
plt.plot(fpr, tpr,color='darkorange',lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC for Mixture of Gaussian Model')
plt.legend(loc="lower right")

for i in range(K):
  mean_disp = np.dot( pcaft.components_.T,bf[k]) + pcaft.mean_
  mean_disp_new = np.array(mean_disp).astype('uint8')
  mean_disp_new = np.reshape(mean_disp_new,(20,20,3))
  plt.imshow(mean_disp_new)
  plt.savefig('mean'+str(i))
  covariances = np.diag(np.diag(cf[k]))
  plt.imshow(covariances)
  plt.savefig('covariance'+str(i))

for i in range(K):
  mean_disp = np.dot( pcanft.components_.T,bnf[k]) + pcanft.mean_
  mean_disp_new = np.array(mean_disp).astype('uint8')
  mean_disp_new = np.reshape(mean_disp_new,(20,20,3))
  plt.imshow(mean_disp_new)
  plt.savefig('meannf'+str(i))
  covariances = np.diag(np.diag(cnf[k]))
  plt.imshow(covariances)
  plt.savefig('covariancenf'+str(i))



