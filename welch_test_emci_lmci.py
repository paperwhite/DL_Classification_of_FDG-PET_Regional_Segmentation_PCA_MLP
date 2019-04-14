import numpy as np
import json
import nibabel as nb
import os
import pickle
from scipy.stats import ttest_ind

path='/home/shibanis/Downloads/ss/build_lm_em'

with open('/home/shibanis/Downloads/ss/letterdict.pickle','rb') as handle:
    b=pickle.loads(handle.read())
print(b)

train_label=[]
test_label=[]
samplenum=0
mu_AD = np.zeros((79,95,79))
mu_Normal = np.zeros((79,95,79))
c_lmci = 0
c_emci = 0
lmci_voxels = np.zeros((79*95*79, 158))
emci_voxels = np.zeros((79*95*79, 178))
for filename in os.listdir(path):
    fullname=os.path.join(path,filename)
    img=nb.load(fullname)
    d=img.get_data()
    print(d.shape)
    uid = filename[6:12]
    print uid
    category = [k for k,v in b.iteritems() if uid in v]
    if(category[0] == 'LMCI'):
        img_arr=np.array(d)
        img_arr[np.isnan(img_arr)]=0
        lmci_voxels[:,c_lmci] = np.reshape(img_arr,(79*95*79))
#        mu_AD = mu_AD + img_arr
        c_lmci += 1
    else:
        img_arr=np.array(d)
        img_arr[np.isnan(img_arr)]=0
        emci_voxels[:,c_emci] = np.reshape(img_arr,(79*95*79))
#        mu_Normal = mu_Normal + img_arr
        c_emci += 1

print c_lmci, c_emci
#158,178

#mu_AD = mu_AD/c_ad
#print mu_AD.shape
#mu_Normal = mu_Normal/c_normal

t = np.zeros((79*95*79))
p = np.zeros((79*95*79))
count = 0
for x in range(0,79*95*79):
    t[x], p[x]=ttest_ind(lmci_voxels[x],emci_voxels[x], equal_var=False)
    if(p[x]<=0.05):
        p[x]=1
    else:
        p[x]=0
print count
#t, p = ttest_ind(mu_AD, mu_Normal, equal_var=False)
print p
    #print(np.isnan(img_arr).sum())
    #top_1percent_mean=np.mean(img_arr[np.argsort(img_arr)[-5928:]])
    #img_arr = img_arr/top_1percent_mean
#p[p<=0.05]=1
#p[p>0.05]=0
print np.sum(p)
new_image_t = nb.Nifti1Image(np.reshape(t,(79,95,79)), affine=img.affine)
new_image_t.to_filename('t_value_lmci_emci.nii.gz')
new_image_p = nb.Nifti1Image(np.reshape(p,(79,95,79)), affine=img.affine)
new_image_p.to_filename('thresholded_p_value_lmci_emci.nii.gz')

