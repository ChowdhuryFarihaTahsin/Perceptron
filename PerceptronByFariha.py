# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 16:00:12 2019

@author: Fariha
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
df = pd.read_csv('train.txt ', sep= " " , header = None,  dtype = None )
#------------------------------TASK 1-----------------

#class 1 calculation
class1= df[2]==1  
w1 = df[class1]

np_w1 = w1.values
C1 = np_w1[:,0:2]

x1 = np_w1[:,0]   #x1 is the x points of class 1
y1 = np_w1[:,1]   #y1 is the y points of class 1 
plt.scatter(x1, y1, color = 'red', marker = '*', label = 'Class 1') #class1 marker is red

#class 2 calculation
class2= df[2]==2
w2 = df[class2]

np_w2 = w2.values
C2 = np_w2[:,0:2]

x2 = np_w2[:,0]
y2 = np_w2[:,1]
plt.scatter(x2, y2, color = 'blue', marker = 'D', label = 'Class 2')  #class 2 is blue
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Perceptron Class')
plt.legend()
plt.show()
#------------------------------TASK 2--------------------------    
Y= np.array([[x1[0]*x1[0],y1[0]*y1[0],x1[0]*y1[0],x1[0],y1[0],1],
            [x1[1]*x1[1],y1[1]*y1[1],x1[1]*y1[1],x1[1],y1[1],1],
            [x1[2]*x1[2],y1[2]*y1[2],x1[2]*y1[2],x1[2],y1[2],1],
 
            [-x2[0]*x2[0],-y2[0]*y2[0],-x2[0]*y2[0],-x2[0],-y2[0],-1],
            [-x2[1]*x2[1],-y2[1]*y2[1],-x2[1]*y2[1],-x2[1],-y2[1],-1],
            [-x2[2]*x2[2],-y2[2]*y2[2],-x2[2]*y2[2],-x2[2],-y2[2],-1]] )

print("Y----------------->")            
print (Y)
#--------------------------------Task 3 AND Task 4----------------------------
#-----------------------------for all ones one at a time--------------
oneAtaTimeOnes = []
alpha = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
flag = 0
for a in alpha:
    iterationo1 = 0
    W= np.ones(6)
    
    while iterationo1<1000:
        iterationo1=iterationo1+1
        flag = 0
        for i in Y:
            g = np.dot(i,W.T)
            if g <=0:                  
                    Update_factor=np.add(W,a*i)
                    W = Update_factor                            
            else:
                flag= flag+1      
        if flag==6:
            oneAtaTimeOnes.append(iterationo1)
            break
#-----------------------------for all ones many at a time-----------
manyAtaTimeOnes = []
alpha = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
flag = 0
for a in alpha:
    iterationM1 = 0
    w = 0 
    W=np.ones(6)
    
    while iterationM1<1000:
        iterationM1 = iterationM1+1
        flag=0
        for i in Y:
            g = np.dot(i,W.T)
            if g <=0:
                w=w+i                
            else:
                flag= flag+1              
        if flag==6:
            manyAtaTimeOnes.append(iterationM1)
            break
        W= W+a*w

print("          Initial weight All Ones")
print("learning rate  one at a time  many at a time")       
for l in range(len(alpha)):
    print(alpha[l],"\t\t",oneAtaTimeOnes[l],"\t\t",manyAtaTimeOnes[l])      
aa=np.arange(10)

bar_width = 0.35
plt.xlabel('Learning Rate')
plt.ylabel('Number of iterations')
plt.title('Initial Weight All Ones')
plt.xticks(aa+bar_width/2,labels=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])

plt.bar(aa,oneAtaTimeOnes,width=bar_width,label="one at a time",color="indigo")
plt.bar(aa+bar_width,manyAtaTimeOnes,width=bar_width,label="many at a time",color="pink")

plt.legend()
plt.show()
#-----------------------------for all zero one at a time-----------   
oneAtaTimeZeros = []        
alpha = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
flag = 0
for a in alpha:
    iterationo0 = 0
    W= np.zeros(6)
    
    while iterationo0<1000:
        iterationo0=iterationo0+1
        flag = 0
        for i in Y:
            g = np.dot(i,W.T)
            if g <=0:                 
                    Update_factor=np.add(W,a*i)
                    W = Update_factor
            else:
                flag= flag+1       
        if flag==6:
            oneAtaTimeZeros.append(iterationo0)
            break       
        
#-----------------------------for all zero many at a time-----------   
            
manyAtaTimeZeros = []   
alpha = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
flag = 0
for a in alpha:
    iterationM0 = 0
    w = 0  
    W=np.zeros(6)
    
    while iterationM0<1000:
        iterationM0 = iterationM0+1
        flag=0
        for i in Y:
            g = np.dot(i,W.T)
            if g <=0:
                w=w+i              
            else:
                flag= flag+1               
        if flag==6:
            manyAtaTimeZeros.append(iterationM0)
            break
        W= W+a*w
        
print("          Initial weight All Zeros")
print("learning rate  one at a time  many at a time")       
for l in range(len(alpha)):
    print(alpha[l],"\t\t",oneAtaTimeZeros[l],"\t\t",manyAtaTimeZeros[l])
    
bar_width = 0.35
plt.xlabel('Learning Rate')
plt.ylabel('Number of iterations')
plt.title('Initial Weight All Zeros')
plt.xticks(aa+bar_width/2,labels=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])

plt.bar(aa,oneAtaTimeZeros,width=bar_width,label="one at a time",color="indigo")
plt.bar(aa+bar_width,manyAtaTimeZeros,width=bar_width,label="many at a time",color="pink")

plt.legend()
plt.show()

#-----------------------------Random weight for one at a time-----------   
oneAtaTimeRandom = []
alpha = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
flag = 0
for a in alpha:
    iterationoR = 0
    np.random.seed(123)      
    W= np.random.randint(10, size=6)
    
    while iterationoR<1000:
        iterationoR=iterationoR+1
        flag = 0
        for i in Y:
            g = np.dot(i,W.T)
            if g <=0:                
                    Update_factor=np.add(W,a*i)
                    W = Update_factor    
            else:
                flag= flag+1       
        if flag==6:
            oneAtaTimeRandom.append(iterationoR)
            break

#-----------------------------Random weight for many at a time-----------   
manyAtaTimeRandom = [] 
alpha = np.array([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
flag = 0
for a in alpha:
    iterationMR = 0
    w = 0 
    np.random.seed(123)     
    W= np.random.randint(10, size=6)
    
    while iterationMR<1000:
        iterationMR = iterationMR+1
        flag=0
        for i in Y:
            g = np.dot(i,W.T)
            if g <=0:
                w=w+i               
            else:
                flag= flag+1               
        if flag==6:
            manyAtaTimeRandom.append(iterationMR) 
            break
        W= W+a*w
print("          Initial weight Random")
print("learning rate  one at a time  many at a time")       
for l in range(len(alpha)):
    print(alpha[l],"\t\t",oneAtaTimeRandom[l],"\t\t",manyAtaTimeRandom[l])

bar_width = 0.35
plt.xlabel('Learning Rate')
plt.ylabel('Number of iterations')
plt.title('Initial Weight Random')
plt.xticks(aa+bar_width/2,labels=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])

plt.bar(aa,oneAtaTimeRandom,width=bar_width,label="one at a time",color="indigo")
plt.bar(aa+bar_width,manyAtaTimeRandom,width=bar_width,label="many at a time",color="pink")

plt.legend()
plt.show()


