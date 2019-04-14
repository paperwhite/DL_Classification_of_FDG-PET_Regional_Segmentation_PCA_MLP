import numpy as np
import json
import nibabel as nb
import os
import pickle
from scipy.stats import ttest_ind
import itertools as it

path='/home/shibanis/Downloads/ss/aal_regional_segmentation/'

with open('/home/shibanis/Downloads/ss/letterdict.pickle','rb') as handle:
    b=pickle.loads(handle.read())
print(b)

sample_image=nb.load('/home/shibanis/Downloads/ss/aal_regional_segmentation/1/res_wI422643.nii.gz')
train_label=[]
test_label=[]
samplenum=0
mu_AD = np.zeros((79,95,79))
mu_Normal = np.zeros((79,95,79))
lmci_voxels = np.zeros((79*95*79, 158))
emci_voxels = np.zeros((79*95*79, 178))
normal_voxels = np.zeros((79*95*79, 186))
ad_voxels = np.zeros((79*95*79, 146))

# Process Welch T Test for a region and specified categories
def two_feature_sets(path, region, category1, category2):
    c_set_1 = 0
    c_set_2 = 0
    set1 = b[category1]
    set2 = b[category2]
    if (category1=='LMCI'):
        set1_voxels = lmci_voxels
    elif (category1=="EMCI"):
        set1_voxels = emci_voxels
    elif (category1=="AD"):
        set1_voxels = ad_voxels
    else :
        set1_voxels = normal_voxels

    if (category2=='LMCI'):
        set2_voxels = lmci_voxels
    elif (category2=="EMCI"):
        set2_voxels = emci_voxels
    elif (category2=="AD"):
        set2_voxels = ad_voxels
    else :
        set2_voxels = normal_voxels
    
    for filename in set1:
        fullname=os.path.join(path+str(region),"res_wI"+filename+".nii.gz")
        img=nb.load(fullname)
        d=img.get_data()
        img_arr=np.array(d)
        img_arr[np.isnan(img_arr)]=0
        set1_voxels[:,c_set_1] = np.reshape(img_arr,(79*95*79))
        c_set_1 += 1

    for filename in set2:
        fullname=os.path.join(path+str(region),"res_wI"+filename+".nii.gz")
        img=nb.load(fullname)
        d=img.get_data()
        img_arr=np.array(d)
        img_arr[np.isnan(img_arr)]=0
        set2_voxels[:,c_set_2] = np.reshape(img_arr,(79*95*79))
        c_set_2 += 1

    print(c_set_1, c_set_2)
    return set1_voxels, set2_voxels
#158,178

#mu_AD = mu_AD/c_ad
#print mu_AD.shape
#mu_Normal = mu_Normal/c_normal
def welch_test(set1_voxels, set2_voxels):
    t = np.zeros((79*95*79))
    p = np.zeros((79*95*79))
    count = 0
    for x in range(0,79*95*79):
        t[x], p[x]=ttest_ind(set1_voxels[x],set2_voxels[x], equal_var=False)
        if(p[x]<=0.05):
            p[x]=1
            count+=1
        else:
            p[x]=0
    print("P Value lt 0..05 count: "+str(count))
    #t, p = ttest_ind(mu_AD, mu_Normal, equal_var=False)
    print(p)
    print(np.sum(p))
    return t,p
    #print(np.isnan(img_arr).sum())
    #top_1percent_mean=np.mean(img_arr[np.argsort(img_arr)[-5928:]])
    #img_arr = img_arr/top_1percent_mean
#p[p<=0.05]=1
#p[p>0.05]=0

def saveAsImage(t, p, path, postfix):
    new_image_t = nb.Nifti1Image(np.reshape(t,(79,95,79)), affine=sample_image.affine)
    new_image_t.to_filename(os.path.join(path,'t_value_'+postfix+'.nii.gz'))
    new_image_p = nb.Nifti1Image(np.reshape(p,(79,95,79)), affine=sample_image.affine)
    new_image_p.to_filename(os.path.join(path,'thresholded_p_value_'+postfix+'.nii.gz'))

def welch_all_regions():
    for region in it.chain(range(1, 90), range(109, 117)):
        print("Processing region: "+ str(region))
       
        # print "AD/CU"
        # set1, set2 = two_feature_sets(path, region, "AD", "Normal")
        # t_ad_cu, p_ad_cu = welch_test(set1, set2)
        # saveAsImage(t_ad_cu,p_ad_cu,os.path.join(path,str(region)), 'ad_cu')
        # print "EMCI/LMCI"
        # set1, set2 = two_feature_sets(path, region, "EMCI", "LMCI")
        # t_em_lm, p_em_lm = welch_test(set1, set2)
        # saveAsImage(t_em_lm,p_em_lm,os.path.join(path,str(region)), 'em_lm')
        print("AD/EMCI")
        set1, set2 = two_feature_sets(path, region, "AD", "EMCI")
        t_ad_cu, p_ad_cu = welch_test(set1, set2)
        saveAsImage(t_ad_cu,p_ad_cu,os.path.join(path,str(region)), 'ad_em')
        print("CU/EMCI")
        set1, set2 = two_feature_sets(path, region, "Normal", "EMCI")
        t_ad_cu, p_ad_cu = welch_test(set1, set2)
        saveAsImage(t_ad_cu,p_ad_cu,os.path.join(path,str(region)), 'cu_em')
        print("CU/LMCI")
        set1, set2 = two_feature_sets(path, region, "Normal", "LMCI")
        t_ad_cu, p_ad_cu = welch_test(set1, set2)
        saveAsImage(t_ad_cu,p_ad_cu,os.path.join(path,str(region)), 'cu_lm')

welch_all_regions()
'''
        set1, set2 = two_feature_test(path, region, "EMCI", "AD")
        t_em_ad, p_em_ad = welch_test(set1, set2)
        saveAsImage(t_em_ad,p_em_ad,os.path.join(path,region), 'em_ad')
        set1, set2 = two_feature_test(path, region, "Normal", "LMCI")
        t_cu_lm, p_cu_lm = welch_test(set1, set2)
        saveAsImage(t_cu_lm,p_cu_lm,os.path.join(path,region), 'cu_lm')
'''
