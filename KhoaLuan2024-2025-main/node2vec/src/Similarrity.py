#Chương trinh tính word similarrity dùng model text
import  io
import numpy as np
from scipy import spatial

fSimlex = 'word/Visim-400.txt'
fWord2vec = 'word/W2V_150.txt'
# reading vsimlex
f = open(fSimlex, 'r',encoding='utf-8')
vsl = f.readlines()
f.close()

# load model
f = open(fWord2vec, 'r')
model= dict({})
for line in f:
    tem=line.split()
 
    vtem=[]
    for i in range(1,len(tem)):
        vtem.append( float(tem[i].strip()))
    vec1=np.array(vtem)
    model.update({tem[0].strip():vec1})

 
rs=[]
v=[]
for i in vsl:
    s=i.split()
    u1 = s[0].strip()
    u2 = s[1].strip()

    if not(u1 in model):
        continue
    if not(u2 in model):
        continue
 
    v1=model[u1.strip()]
    v2=model[u2.strip()]
    if (len(v1)!=len(v2)):
        print(" loi vector khong cung chieu \n")
        continue
 
    k = ((2 - spatial.distance.cosine(v1, v2))/2)*10
	
    v.append(float(s[2].strip()))  
    print(u1 +"- "+u2+ " = "+str(k)+'\n')
    rs.append(k)

 
print(np.corrcoef(rs,v))
print(np.correlate(rs,v))
f.close()
