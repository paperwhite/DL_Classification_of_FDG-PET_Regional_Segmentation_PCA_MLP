import numpy as np
import json
import nibabel as nb
import os
import pickle

path='/home/shibanis/Downloads/ss/all_files_AD_Normal'

with open('/home/shibanis/Downloads/ss/letterdict.pickle','rb') as handle:
    b=pickle.loads(handle.read())
print(b)

for filename in os.listdir(path):
    if (os.path.isfile(os.path.join('build','res_'+filename))):
        print filename+" file exists!"
        
    else:
        fullname=os.path.join(path,filename)
        img=nb.load(fullname)
        d=img.get_data()
        print(d.shape)
        img_arr=np.array(d)
        img_arr[np.isnan(img_arr)]=0

    #print(np.isnan(img_arr).sum())
        top_1percent_mean=np.mean(img_arr[np.argsort(img_arr)[-5928:]])
        img_arr = img_arr/top_1percent_mean

        new_image = nb.Nifti1Image(img_arr, affine=img.affine)
        new_image.to_filename(os.path.join('build','res_'+filename))
    
