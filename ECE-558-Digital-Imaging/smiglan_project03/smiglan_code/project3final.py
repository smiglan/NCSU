import tkinter as tk # this is in python 3.4. For python 2.x import Tkinter
from PIL import Image, ImageTk, ImageGrab
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
from tkinter import font

imgname = 'monalisa.png'
imgname1 = 'taylor.jpg'
img = cv2.imread(imgname)
img1 = cv2.imread(imgname1) 

size = 400

img = cv2.resize(img,(size,size))
img1 = cv2.resize(img1,(size,size))

def sample(image,factor):

    x1 = math.ceil(factor*image.shape[0])
    x2 = math.ceil(factor*image.shape[1])
    #print(int(x1),int(x2))
    sampledimage = np.zeros((x1,x2))

    for i in range(x1):
        for j in range(x2):
            p = image[math.floor(i/factor),math.floor(j/factor)]
            sampledimage[i][j] = p
    return sampledimage

b,g,r = cv2.split(img)
def loop(kernel,result):
    size0 = 1
    size1 = 1
    resultnew = np.zeros(result.shape)
    size0 = int((kernel.shape[0]-1)/2)
    size1 = int((kernel.shape[1]-1)/2)
    for c1 in range(size0,result.shape[0]-size0):
        for c2 in range(size1,result.shape[1]-size1):

            resultnew[c1,c2] = np.sum(np.multiply(kernel,result[c1-size0:c1+size0+1,c2-size1:c2+size1+1]))

            
    return resultnew
def padding(image,kernel,paddingtype):
    size0 = 1
    size1 = 1
    if kernel.shape[0]%2 >=1 and kernel.shape[1]%2>= 1: 
        size0 = int((kernel.shape[0]-1)/2)
        size1 = int((kernel.shape[1]-1)/2)
    if paddingtype == 'zero':
        result = np.zeros((image.shape[0]+2*size0,image.shape[1]+2*size1))
        result[size0:image.shape[0]+size0,size1:image.shape[1]+size1] += image
    return result.astype(np.uint8)  
def conv2(image,kernel,paddingtype):
    size0 = int((kernel.shape[0]-1)/2)

    size1 = 1
    if len(kernel.shape) != 1:
        size1 = int((kernel.shape[1]-1)/2)
    b,g,r = cv2.split(image)
    resultb = padding(b,kernel,paddingtype)
    resultb = loop(kernel,resultb)
    resultg = padding(g,kernel,paddingtype)
    resultg = loop(kernel,resultg)
    resultr = padding(r,kernel,paddingtype)
    resultr = loop(kernel,resultr)
    convimageb = resultb[size0:image.shape[0]+size0,size1:image.shape[1]+size1]
    convimageg = resultg[size0:image.shape[0]+size0,size1:image.shape[1]+size1]
    convimager = resultr[size0:image.shape[0]+size0,size1:image.shape[1]+size1]
    return cv2.merge((convimageb, convimageg, convimager))      
def gaussiankernel(shape=(5,5),sigma=1):

    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def parameters(factor):
    truncate = 4
    sigma = (2*factor)/6
    k = int(truncate*sigma+0.5)
    return sigma,k
k,sigma = parameters(1/2)


def gaussianpyramid(im,level):
    #gpyr[0] = im2double(img)
    gpyr = []
    gpyr.append(im.copy())
    for p in range(1,level+1):
        l = conv2(gpyr[p-1],gaussiankernel(),'zero')
        b1,g1,r1 = cv2.split(l)
        l = cv2.merge((sample(b1,1/2), sample(g1,1/2), sample(r1,1/2)))
        gpyr.append(np.float32(l))
    return gpyr


def laplacianpyramid(gpyrr,img,level):
    lpyr = []
    lpyr.append(gpyrr[level-1])

    for i in range(level-1,0,-1):

        l = conv2(gpyrr[i],gaussiankernel(),'zero')

        b1,g1,r1 = cv2.split(l)
        l = cv2.merge((sample(b1,2), sample(g1,2), sample(r1,2)))

        lpyr.append(np.subtract(gpyrr[i-1],l))
    return lpyr

def pyramidup(pi):


    l = conv2(pi,gaussiankernel(),'zero')
    b1,g1,r1 = cv2.split(l)
    l = cv2.merge((sample(b1,2), sample(g1,2), sample(r1,2)))

    return l

def blend(fore,back,mask,level):

    gpyr_fore = gaussianpyramid(fore,level)
    gpyr_back = gaussianpyramid(back,level)


    gpyr_mask = gaussianpyramid(mask,level)

    lpyr_fore = laplacianpyramid(gpyr_fore,fore,level)
    lpyr_back = laplacianpyramid(gpyr_back,back,level)
    
    lpyr_mask = [gpyr_mask[level-1]]
    for i in range(level-1,0,-1):
        lpyr_mask.append(gpyr_mask[i-1])
    BL = []
    for x,y,z in zip(lpyr_fore,lpyr_back,lpyr_mask):
        bl = x*z+y*(1.0-z)
        BL.append(bl)

    blr = BL[0]
    for i in range(1,level):
        blr = pyramidup(blr)
        blr = cv2.add(blr,BL[i] )
    return blr
        
startx = 0
endx = 0
starty = 0
endy = 0
typecore = 1

class ExampleApp(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        # frame = tk.Frame()
       
        self.canvas = tk.Canvas(width=1024, height=600)

        self.canvas.pack(side="top", fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)
        



        helv36 = tk.font.Font(family = 'Helvetica',size = 15)
        self.v = tk.IntVar()
        buttoncircle = tk.Radiobutton(self,text = "RECTANGLE", variable = self.v, value = 1,font = helv36)
        buttoncircle.configure(width = 15, activebackground = "#33B5E5",relief = 'flat')
        button_window_circle = self.canvas.create_window(200,10,window = buttoncircle)

        buttonellipse = tk.Radiobutton(self,text = "ELLIPSE", variable = self.v, value = 2,font = helv36)
        buttonellipse.configure(width = 15, activebackground = "#33B5E5",relief = 'flat')
        button_window_ellipse = self.canvas.create_window(200,50,window = buttonellipse)





        button = tk.Button(self,text = "BLEND", command = self.quit, anchor = 'w',font = helv36)
        button.configure(width = 50, activebackground = "#33B5E5",relief = 'flat')
        button_window = self.canvas.create_window(462,584,anchor = 'sw',window = button)

        


        self.rect = None

        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None


        self._draw_image()


    def _draw_image(self):
         self.im = Image.open(imgname)
         self.im = self.im.resize((400,400))
         self.tk_im = ImageTk.PhotoImage(self.im)
         self.im1 = Image.open(imgname1)
         self.im1 = self.im1.resize((400,400))
         self.tk_im1 = ImageTk.PhotoImage(self.im1)
         self.canvas.create_image(50,100,anchor="nw",image=self.tk_im)
         self.canvas.create_image(500,100,anchor="nw",image=self.tk_im1)



    def on_button_press(self, event):
        # save mouse drag start position
        self.start_x = event.x
        self.start_y = event.y
        global startx,starty
        startx = self.start_x
        starty = self.start_y
        # create rectangle if not yet exist
        if not self.rect:
            global typecore

            if self.v.get() == 1:
                typecore = 1
                self.rect = self.canvas.create_rectangle(self.x, self.y, 1, 1)#, fill="black")
            if self.v.get() == 2:
                typecore = 2
                self.rect = self.canvas.create_oval(self.x, self.y, 1, 1)#, fill="black")

    def on_move_press(self, event):
        curX, curY = (event.x, event.y)

        # expand rectangle as you drag the mouse
        self.canvas.coords(self.rect, self.start_x, self.start_y, curX, curY)



    def on_button_release(self, event):
        self.end_x = event.x
        self.end_y = event.y
        global endx,endy
        endx = self.end_x
        endy = self.end_y

        pass
    def getcoord(self):
        return self.start_x,self.start_y,self.end_x,self.end_y


if __name__ == "__main__":
    app = ExampleApp()
    app.mainloop()
    m = np.zeros_like(img, dtype='float32')

    if typecore ==1:
        m = np.zeros_like(img, dtype='float32')

        m[starty-100:endy-100,startx-50:endx-50] = 1  
    else:
        color = (1,1,1)
        centercoords = (startx+int((endx-startx)/2)-50,starty+int((endy-starty)/2)-100)
        axeslength  = (int((endx-startx)/2),int((endy-starty)/2))
        m = np.zeros_like(img, dtype='float32')
        m = cv2.ellipse(m,centercoords,axeslength,0,0,360,color,-1)
    cv2.imshow('mask',m)
    cv2.waitKey(0)
    blendedimage = blend(img,img1,m,5)
    cv2.imshow('asd',blendedimage.astype(np.uint8))
    cv2.waitKey(0)
    cv2.imwrite("mine.png",blendedimage)
