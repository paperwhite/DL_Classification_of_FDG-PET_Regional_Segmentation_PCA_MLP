import numpy as np
import json
import nibabel as nb
import os
import pickle
from scipy.stats import ttest_ind

path='/home/shibanis/Downloads/ss/build_cu_lm'

with open('/home/shibanis/Downloads/ss/letterdict.pickle','rb') as handle:
    b=pickle.loads(handle.read())
print(b)

train_label=[]
test_label=[]
samplenum=0
mu_LMCI = np.zeros((79,95,79))
mu_Normal = np.zeros((79,95,79))
c_normal = 0
c_lmci = 0
normal_voxels = np.zeros((79*95*79, 186))
lmci_voxels = np.zeros((79*95*79, 158))
for filename in os.listdir(path):
    fullname=os.path.join(path,filename)
    img=nb.load(fullname)
    d=img.get_data()
    print(d.shape)
    uid = filename[6:12]
    print uid
    category = [k for k,v in b.iteritems() if uid in v]
    if(category[0] == 'Normal'):
        img_arr=np.array(d)
        img_arr[np.isnan(img_arr)]=0
        normal_voxels[:,c_normal] = np.reshape(img_arr,(79*95*79))
#        mu_LMCI = mu_LMCI + img_arr
        c_normal += 1
    else:
        img_arr=np.array(d)
        img_arr[np.isnan(img_arr)]=0
        lmci_voxels[:,c_lmci] = np.reshape(img_arr,(79*95*79))
#        mu_Normal = mu_Normal + img_arr
        c_lmci += 1

#print c_lmci, c_normal
#186,146

#mu_LMCI = mu_LMCI/c_lmci
#print mu_LMCI.shape
#mu_Normal = mu_Normal/c_normal

t = np.zeros((79*95*79))
p = np.zeros((79*95*79))

for x in range(0,79*95*79):
    #t[x,0], p[x,0]=ttest_ind(normal_voxels[x],lmci_voxels[x], equal_var=False)
    t[x], p[x]=ttest_ind(normal_voxels[x],lmci_voxels[x], equal_var=False)
    if(p[x]<=0.05):
        p[x]=1
    else:
        p[x]=0
#t, p = ttest_ind(mu_LMCI, mu_Normal, equal_var=False)
print p
    #print(np.isnan(img_arr).sum())
    #top_1percent_mean=np.mean(img_arr[np.argsort(img_arr)[-5928:]])
    #img_arr = img_arr/top_1percent_mean
#p[p<=0.05]=1
#p[p>0.05]=0
new_image_t = nb.Nifti1Image(np.reshape(t,(79,95,79)), affine=img.affine)
new_image_t.to_filename('t_value_cu_lm.nii.gz')
new_image_p = nb.Nifti1Image(np.reshape(p,(79,95,79)), affine=img.affine)
new_image_p.to_filename('thresholded_p_value_cu_lm.nii.gz')

