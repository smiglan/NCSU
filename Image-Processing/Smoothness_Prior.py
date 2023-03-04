import time
start_time = time.time()
import cv2 
import numpy as np
import matplotlib.pyplot as plt

type = 'x+1','y' #define neighbour type

img = cv2.imread('C:/Users/Shubham/Desktop/Fall19/imagin 558/ECE558-HW01/ECE558-HW01/wolves.png')  

b,g,r = cv2.split(img)
bgr = np.ndarray(shape=(img.shape[0],img.shape[1]))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
graynew = np.ndarray(shape=(img.shape[0],img.shape[1]))

h,s,v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
hsv = np.ndarray(shape=(img.shape[0],img.shape[1]))

l,a,ba = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2Lab))
lab = np.ndarray(shape=(img.shape[0],img.shape[1]))


def error(a,b): #define error function
	v = (a-b)*(a-b)
	return v

def neighbour(a,b):
	if a == 'x+1' and b == 'y':
		return 1,0
	elif a == 'x+1' and b == 'y+1':
		return 1,1
	elif a == 'x' and b == 'y+1':
		return 0,1
	elif a == 'x-1' and b == 'y+1':
		return -1,1
	elif a == 'x-1' and b == 'y':
		return -1,0
	elif a == 'x-1' and b == 'y-1':
		return -1,-1
	elif a == 'x' and b == 'y-1':
		return 0,-1
	elif a == 'x+1' and b == 'y-1':
		return 1,-1




i,j = neighbour(type[0],type[1])
for c1 in range(0,539):
	for c2 in range(0,1500):
		if c1>=max(0,-i) and c2>=max(0,-j) and c1<=img.shape[1]-1-max(0,i) and c2<=img.shape[0]-1-max(0,j):
			if c1+j>=max(0,-i) and c2+i>=max(0,-j) and c1+j<=img.shape[1]-1-max(0,i) and c2+i<=img.shape[0]-1-max(0,j):
				graynew[c2][c1]=error(int(gray[c2][c1]),int(gray[c2+i][c1+j]))

for c1 in range(0,539):
	for c2 in range(0,1500):
		if c1>=max(0,-i) and c2>=max(0,-j) and c1<=img.shape[1]-1-max(0,i) and c2<=img.shape[0]-1-max(0,j):
			if c1+j>=max(0,-i) and c2+i>=max(0,-j) and c1+j<=img.shape[1]-1-max(0,i) and c2+i<=img.shape[0]-1-max(0,j):
				bgr[c2][c1]=error(int(b[c2][c1]),int(b[c2+i][c1+j]))+error(int(g[c2][c1]),int(g[c2+i][c1+j]))+error(int(r[c2][c1]),int(r[c2+i][c1+j]))

for c1 in range(0,539):
	for c2 in range(0,1500):
		if c1>=max(0,-i) and c2>=max(0,-j) and c1<=img.shape[1]-1-max(0,i) and c2<=img.shape[0]-1-max(0,j):
			if c1+j>=max(0,-i) and c2+i>=max(0,-j) and c1+j<=img.shape[1]-1-max(0,i) and c2+i<=img.shape[0]-1-max(0,j):

				hsv[c2][c1]=error(int(h[c2][c1]),int(h[c2+i][c1+j]))+error(int(s[c2][c1]),int(s[c2+i][c1+j]))+error(int(v[c2][c1]),int(v[c2+i][c1+j]))
			
for c1 in range(0,539):
	for c2 in range(0,1500):
		if c1>=max(0,-i) and c2>=max(0,-j) and c1<=img.shape[1]-1-max(0,i) and c2<=img.shape[0]-1-max(0,j):
			if c1+j>=max(0,-i) and c2+i>=max(0,-j) and c1+j<=img.shape[1]-1-max(0,i) and c2+i<=img.shape[0]-1-max(0,j):

				lab[c2][c1]=error(int(l[c2][c1]),int(l[c2+i][c1+j]))+error(int(a[c2][c1]),int(a[c2+i][c1+j]))+error(int(ba[c2][c1]),int(ba[c2+i][c1+j]))
	

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2,2)
ax1.hist(graynew.ravel())
ax1.set_title('gray')
ax2.hist(bgr.ravel())
ax2.set_title('bgr')
ax3.hist(hsv.ravel())
ax3.set_title('hsv')
ax4.hist(lab.ravel())
ax4.set_title('lab')
fig.tight_layout()
print("--- %s seconds ---" % (time.time() - start_time))
plt.show()
  
