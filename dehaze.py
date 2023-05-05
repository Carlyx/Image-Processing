```

import numpy as np 
import cv2

def init(kernal):
    print(min(0.49390244, 0.9310345 ,0.80487806))
    global img,row,col,dep,M,m1,m2,m3,rr,cc
    img = cv2.imread('images/test2.png')
    # img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    row,col,dep = img.shape
    M = np.array(img)
    m1 = M[:,:,0]
    m2 = M[:,:,1]
    m3 = M[:,:,2]      
    rr = np.zeros((kernal-1)//2*col).reshape((kernal-1)//2,col)  #(2, 461)
    cc = np.zeros((row+kernal-1)*(kernal-1)//2).reshape(row+kernal-1,(kernal-1)//2)  #(304, 2)

def exmatrix(m):
    m0 = np.vstack((m,rr))
    m0 = np.vstack((rr,m0))
    m0 = np.hstack((m0,cc))
    m0 = np.hstack((cc,m0)) #(304, 465)
    return m0

def getGray(a,b,c):
    m0=a
    for i in range(0,a.shape[0]):
        for j in range(0,a.shape[1]):
            m0[i][j] = min(a[i][j],b[i][j],c[i][j])
    # print(m0)
    return m0

def minfilter(m,m0,kernal):
    return cv2.erode(m0,np.ones((2*kernal-1,2*kernal-1)),3)
    for i in range((kernal-1)//2,m.shape[0]-(kernal-1)//2):
        for j in range((kernal-1)//2,m.shape[1]-(kernal-1)//2):
            a=[]
            for ii in range(i-(kernal-1)//2,i+(kernal-1)//2+1):
                for jj in range(j-(kernal-1)//2,j+(kernal-1)//2+1):
                    a.append(m[ii][jj])
            m0[i-(kernal-1)//2][j-(kernal-1)//2] = min(a)
    return m0

def getA(m):
    ma = m.max()
    res = [0,0,0]
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if m[i][j]==ma:
                if m1[i][j]+m2[i][j]+m3[i][j]>res[0]+res[1]+res[2]:
                    res=[]
                    res.append(m1[i][j])
                    res.append(m2[i][j])
                    res.append(m3[i][j])
    return res

def getT(a,kernal):
    # print(a[0],a[1],a[2])
    T = m1*1.0 
    for i in range(m1.shape[0]):
        for j in range(m1.shape[1]):
            # print(m1[i][j]/a[0],m2[i][j]/a[1],m3[i][j]/a[2])
           T[i][j] = min(m1[i][j]/a[0],m2[i][j]/a[1],m3[i][j]/a[2])
        #    print(T[i][j])
    T = minfilter(exmatrix(T),T,kernal)
    return 1-0.95*T

def getJ(m,a,t):
    mm = m
    for i in range(0,mm.shape[0]):
        for j in range(0,mm.shape[1]):
            p = (mm[i][j]-a)/max(t[i][j],0.1)
            # print(p)
            if int(p)+a>255:
                mm[i][j]=255
            elif int(p)+a<0:
                mm[i][j]=0
            else:
                mm[i][j] = int(p)+a
    return mm

def output():
    cv2.imshow('img',img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def test(a,b,c):
    dst = np.dstack((a.astype(int),b.astype(int)))
    dst = np.dstack((dst,c.astype(int)))
    dst=dst.astype(np.uint8)
    cv2.imshow('dst',dst)
    return dst 

def guideFilter(I, p, winSize, eps):
    mean_I = cv2.blur(I, winSize)      
    mean_p = cv2.blur(p, winSize)      
    mean_II = cv2.blur(I * I, winSize)
    mean_Ip = cv2.blur(I * p, winSize)
    var_I = mean_II - mean_I * mean_I  
    cov_Ip = mean_Ip - mean_I * mean_p 
    a = cov_Ip / (var_I + eps)         
    b = mean_p - a * mean_I           
    mean_a = cv2.blur(a, winSize)    
    mean_b = cv2.blur(b, winSize)     
    q = mean_a * I + mean_b
    return q

if __name__ == "__main__":
    kernal = 3
    init(kernal)

    m_gray = getGray(m1,m2,m3)
    # print(m_gray.shape)
    m_min = minfilter(exmatrix(m_gray),m_gray,kernal)
    # cv2.imshow('m_gray',m_gray)
    cv2.imshow('m_min',m_min)
    

    A = getA(m_min)
    print(A)
    T = getT(A,kernal)
    print(T.max(),T.min())
    cv2.imshow('T',(T*255).astype(np.float32).astype(np.uint8))
    # print(T)
    # print(A[0])
    print(T)
    # TT = cv2.resize(T, None,fx=1, fy=1, interpolation=cv2.INTER_CUBIC)
    T =  guideFilter(T, T, (3,3), 1e6)
    print(T)
    cv2.imshow('T1',(T*255).astype(np.float32).astype(np.uint8))

    m1_last = getJ(m1,int(A[0]),T)
    # # # print(m1_last.shape)

    m2_last = getJ(m2,int(A[1]),T)
    # # # print(m2_t)
    
    m3_last = getJ(m3,int(A[2]),T)
    # # # print(m3_t)

    output()
    # # # print(m3_last.max())
    test(m1_last,m2_last,m3_last)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

```
