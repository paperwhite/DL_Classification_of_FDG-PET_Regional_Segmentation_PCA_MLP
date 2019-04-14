import numpy as np
import json
import nibabel as nb
import os
import pickle
from scipy.stats import ttest_ind

path='/home/shibanis/Downloads/ss/build_lm_em'

with open('/home/shibanis/Downloads/ss/letterdict.pickle','rb') as handle:
    b=pickle.loads(handle.read())

img=nb.load('p_value_lmci_emci.nii.gz')
d=img.get_data()
img_arr=np.array(d)
img_arr[np.isnan(img_arr)]=0
print img_arr.shape
img_arr = np.reshape(img_arr,(79,95,79))

#print np.max(img_arr), np.min(img_arr)
print len(np.where(img_arr > 0.05))

print np.sum(img_arr)
img_arr[img_arr<=0.05]=1
img_arr[img_arr>0.05]=0

