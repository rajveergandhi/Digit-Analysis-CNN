
# coding: utf-8

# In[1]:


import numpy as np
import math


# In[2]:


from sklearn.cluster import KMeans
import cv2
import numpy as np

def extract_letters(i):
    i = np.array(i).astype(np.uint8)
    chars = []

    t = 145
    ret, new_i = cv2.threshold(i, t, 255, cv2.THRESH_BINARY)

    while sum(sum(new_i / 255)) > 400 and t < 256:        
        ret, new_i = cv2.threshold(i,t, 255, cv2.THRESH_BINARY)
        #step size
        t += 15
        
    _ , contours, _ = cv2.findContours(new_i, 1, cv2.CHAIN_APPROX_SIMPLE)
    
    valid = []
    
    # Removing small bright spots that didn't get removed by the threshold
    for c in contours:
        if(cv2.contourArea(c) < 20):
            cv2.drawContours(new_i, [c], -1, 0, cv2.FILLED)
            continue
        else:
            valid.append(c)
    
    
    # Finding center of mass
    centers = []
    for c in valid:
        M = cv2.moments(c)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        centers.append([cx,cy])
    
    if(len(centers) < 3):
        return chars
    
    c = KMeans(n_clusters=3)
    c.fit(centers)
    s = 32 // 2 # cropped image "radius"
    labels = c.labels_

    point_cloud = np.array([[x,y] for y,r in enumerate(new_i) for x,p in enumerate(r) if p > 0])
    labels = np.array([c.predict([p])[0] for p in point_cloud])

    # Cutting characters from their location and placing them in new 32x32 image
    for index, (x,y) in enumerate(c.cluster_centers_):
        current_points = point_cloud[np.where(labels == index)]
        empty_patch = np.zeros((32,32))
        x = int(x - 16)
        y = int(y - 16)
        for px,py in current_points:
            empty_patch[max(0,min(31,py-y)),max(0,min(31,px-x))] = 1

        chars.append(empty_patch)

    return chars


# In[ ]:


X_train_unlabled = [] 
X_train_failed = []
#import unlabled data
with open('train_x_small.csv','r') as f:
    j = 0
    for i,l in enumerate(f):
        j += 1
        raw_image = np.array([int(float(a)) for a in l.split(',')])
        raw_image = raw_image.reshape(64,64)
        letters = extract_letters(raw_image)
        if(len(letters) != 3):
            X_train_failed.append(raw_image)
        else:
            X_train_unlabled += letters
        if j%1000 == 0:
            print(j)


# In[25]:


len(X_train_failed)


# In[24]:


sum(sum(X_train_failed[0] == X_train_failed[0]))


# In[26]:


index_unlabled = [] 
index_failed = []
flat_failed = np.array(X_train_failed).reshape(2160, 4096)
#import unlabled data
with open('train_x_small.csv','r') as f:
    for i,l in enumerate(f):
        raw_image = np.array([int(float(a)) for a in l.split(',')])
        found = False
        for fi in flat_failed:
            if((fi==raw_image).all()):
                found = True
                index_failed.append(i)
                break
        
        if(not found):
            index_unlabled.append(i)


# In[23]:


len(index_failed) == len(X_train_failed)


# In[43]:


with open('train_extracted_letters_data.csv','w') as df:
    with open('train_extracted_letters_label.csv','w') as lf:
        for i,d in enumerate(X_train_unlabled):
            d = np.array(d).reshape(-1)
            df.write(','.join([str(x) for x in d]) + '\n')
            lf.write(str(index_unlabled[int(math.floor(i / 3))]) + '\n')


# In[5]:


X_train_unlabled[0][0]

