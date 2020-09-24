import time
start_time = time.time()
import numpy as np
import collections
import cv2 

img = cv2.imread('C:/Users/Shubham/Desktop/Fall19/imagin 558/ECE558-HW01/ECE558-HW01/wolves.png')  
V = [0,1] #set V for matrix, comment when using an image
#V = list(range(256)) #set V for image, uncomment when using an image
a = np.array([[3, 1, 2,1], [2, 2, 0,2], [1, 2, 1,1], [1,0,1,2]]) #comment when using an image

p = (3,0) #first point
q = (0,3) #end point 
#all paths will be printed together
#uncomment next two lines if required to use image
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
#a = gray

endp = q  
def mpath(grid, start,end):
    queue = collections.deque([[start]])
    seen = set([start])

    while queue:
        path = queue.popleft()
        x, y = path[-1]
        if y == end[0] and x == end[1]:
            return path
        listofpossible = []
        for (x1,y1) in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):
            if 0 <= x1 < width and 0 <= y1 < height:

                if x1==0:
                    listofpossible.append((0,y1))
                else:
                    listofpossible.append((x1,y1))
        for (x2, y2) in  ((x+1,y), (x-1,y), (x,y+1), (x,y-1),(x-1,y-1),(x+1,y+1),(x-1,y+1),(x+1,y-1)):
            if (x2,y2) in ((x-1,y-1),(x+1,y+1),(x-1,y+1),(x+1,y-1)):
                listofpossible2 = []
                if 0 <= x2 < width and 0 <= y2 < height:
                    for(x3,y3) in  ((x2+1,y2), (x2-1,y2), (x2,y2+1), (x2,y2-1)):
            
                        if 0 <= x3 < width and 0 <= y3 < height:

                            if x3==0:
                                listofpossible2.append((0,y3))
                            else:
                                listofpossible2.append((x3,y3))
                    inter = intersection(listofpossible,listofpossible2)  
                    if inter==0 and (grid[y2][x2] in V or y2 == end[0] and x2 == end[1]) and (x2, y2) not in seen:
                        queue.append(path + [(x2, y2)])
                        seen.add((x2, y2))
            if (x2,y2) in  ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):

                if 0 <= x2 < width and 0 <= y2 < height and (grid[y2][x2] in V or y2 == end[0] and x2 == end[1]) and (x2, y2) not in seen:
                    queue.append(path + [(x2, y2)])
                    seen.add((x2, y2))
def eightpath(grid, start,end):
    queue = collections.deque([[start]])
    seen = set([start])

    while queue:
        path = queue.popleft()
        x, y = path[-1]
        if y == end[0] and x == end[1]:
            return path
        for x2, y2 in ((x+1,y), (x-1,y), (x,y+1), (x,y-1),(x-1,y-1),(x+1,y+1),(x-1,y+1),(x+1,y-1)):
            if 0 <= x2 < width and 0 <= y2 < height and (grid[y2][x2] in V or y2 == end[0] and x2 == end[1]) and (x2, y2) not in seen:
                queue.append(path + [(x2, y2)])
                seen.add((x2, y2))

def fourpath(grid, start,end):
    queue = collections.deque([[start]])
    seen = set([start])
    
    while queue:
        path = queue.popleft()
        x, y = path[-1]
        if y == end[0] and x == end[1]:
            return path
        for x2, y2 in ((x+1,y), (x-1,y), (x,y+1), (x,y-1)):
            if 0 <= x2 < width and 0 <= y2 < height and (grid[y2][x2] in V or y2 == end[0] and x2 == end[1]) and (x2, y2) not in seen:
                queue.append(path + [(x2, y2)])
                seen.add((x2, y2))

    


width, height = a.shape[1], a.shape[0]

def Reverse(tuples): 
    new_tup = () 
    for k in reversed(tuples): 
        new_tup = new_tup + (k,) 
    return new_tup 

def Reversel(l): 
    if l != None:
        for i in range(len(l)):
            l[i] = Reverse(l[i])
        print('shortest path: ',l, 'length of the shortest path: ', len(l)-1)
    else: print('None') 
def intersection(r, s): 
    a_set = set(r) 
    b_set = set(s) 
    intersection = a_set & b_set
    values = []
    for i in intersection:
        values.append(a[i[1]][i[0]])
    for v in V:
        if v in values:
            return 1
    return 0

if 0 <= endp[1] < width and 0 <= endp[0] < height and 0 <= Reverse(p)[0] < height and 0 <= Reverse(p)[1] < width:

    fpath = fourpath(a, Reverse(p),endp) 
    epath = eightpath(a,Reverse(p),endp)
    mpath = mpath(a, Reverse(p),endp)
    print('4-path')
    Reversel(fpath)
    print('8-path')
    Reversel(epath)
    print('m-path')
    Reversel(mpath)
else:
    print('invalid start or end position')
print("--- Run time: %s seconds ---" % (time.time() - start_time))
