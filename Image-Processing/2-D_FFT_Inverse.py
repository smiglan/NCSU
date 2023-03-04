#question2a
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import cv2 

import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack
import cv2 
import math
path = 'C:/Users/Shubham/Desktop/Fall19/imagin 558/Project02/'
img = cv2.imread('C:/Users/Shubham/Desktop/Fall19/imagin 558/Project02/wolves.png') 
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
def scale(image):
	max = np.max(image)
	min = np.min(image)
	L = 2
	resultnew = np.zeros(image.shape)
	for c1 in range(0,image.shape[0]):
		for c2 in range(0,image.shape[1]):
			resultnew[c1][c2] = ((image[c1][c2]-min)/(max-min))*(L-1)
	return resultnew
grayn = scale(gray)
def DFT2(image):
	oned = np.fft.fft(image)
	return np.transpose(np.fft.fft(oned.transpose()))
fshift = np.fft.fftshift(twod)
magnitude = 50*np.log(1+np.abs(fshift))
twod = DFT2(grayn)
def IDFT2(input):
	real = input.real
	imag = -1*input.imag
	Forwardnew = real+1j*imag
	Inverse = DFT2(Forwardnew)
	Inverse = Inverse/808500
	InverseReal = Inverse.real
	Inverseimag = -1*Inverse.imag
	Inversenew =  InverseReal+1j*Inverseimag
	return Inversenew
Inverses = IDFT2(twod)

plt.figure(1)
plt.imshow(grayn, cmap = 'gray')
plt.title('Input Image')
plt.axis('off')
plt.savefig(path+'input')
plt.figure(2)
plt.imshow(Inverses.real, cmap = 'gray')
plt.axis('off')
plt.title('Inverse')
plt.savefig(path+'Inverse')

plt.figure(3)
plt.imshow(magnitude, cmap = 'gray')
plt.title('Magnitude ')
plt.axis('off')
plt.savefig(path+'magnitudelena')
plt.figure(4)
phase = np.angle(fshift,deg=True)
plt.title('phase ')
plt.axis('off')
plt.imshow(phase)
plt.savefig(path+'phaselena')
#plt.show()





