# Author - Shibani Singh

import math
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA, IncrementalPCA
import sklearn as sk
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
#from sklearn.cross_validation import train_test_split
import pickle
import itertools as it
import numpy as np
import nibabel as nb
import os
path='/home/shibanis/Downloads/ss/aal_regional_segmentation/'

with open('/home/shibanis/Downloads/ss/letterdict.pickle','rb') as handle:
    b=pickle.loads(handle.read())
#print(b)
lmci_voxels = np.zeros((79*95*79, 158))
emci_voxels = np.zeros((79*95*79, 178))
normal_voxels = np.zeros((79*95*79, 186))
ad_voxels = np.zeros((79*95*79, 146))

ad_cu_data_set=np.zeros(())

def unison_shuffled_copies(a,b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]
def prep_all_regions(category1, category2):
    count_values = 0
    num_values, set1_train,set2_train = prep_region(1, category1,category2)
    count_values+=num_values
    for region in it.chain(range(2, 91), range(109, 117)):
        print category1,"/",category2," : Region - ",region
        num_values, set1, set2 = prep_region(region, category1,category2)
        
        print np.shape(set1)
        set1_train = np.append(set1_train,set1,axis=1)
        set2_train = np.append(set2_train,set2,axis=1)
        count_values+=num_values
        print count_values
        print np.shape(set1_train)
        print np.shape(set2_train)
    np.save('all_regions_' + category1+'__'+category1+category2+'_'+'.npy', set1_train)
    np.save('all_regions_' + category2+'__'+category1+category2+'_'+'.npy', set2_train)

    print "num_features_for_all_regions: ",count_values
    return set1_train, set2_train
#Multiply region with the P-value for all subjects and load into data set
def prep_region(region_num, category1, category2):
    if(category1=="AD" and category2=="Normal"):
        welch_filter = nb.load(path+str(region_num)+"/thresholded_p_value_ad_cu.nii.gz")
    if(category1=="EMCI" and category2=="LMCI"):
        welch_filter = nb.load(path+str(region_num)+"/thresholded_p_value_em_lm.nii.gz")
    if(category1=="Normal" and category2=="EMCI"):
        welch_filter = nb.load(path+str(region_num)+"/thresholded_p_value_cu_em.nii.gz")
    if(category1=="Normal" and category2=="LMCI"):
        welch_filter = nb.load(path+str(region_num)+"/thresholded_p_value_cu_lm.nii.gz")
    if(category1=="AD" and category2=="EMCI"):
        welch_filter = nb.load(path+str(region_num)+"/thresholded_p_value_ad_em.nii.gz")
    welch_filter_array = welch_filter.get_fdata()
    welch_filter_array[np.isnan(welch_filter_array)]=0
    num_values = 0
    for val in np.reshape(welch_filter_array,(79*95*79)):
        if (val!=0):
            num_values+=1
    print num_values
    lmci_voxels = np.zeros((158,num_values))
    emci_voxels = np.zeros((178,num_values))
    normal_voxels = np.zeros((186,num_values))
    ad_voxels = np.zeros((146,num_values))
 
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
        fullname=os.path.join(path+str(region_num),"res_wI"+filename+".nii.gz")
    #for filename in os.listdir(os.path.join(path,region_num)):
        #fullname=os.path.join(path,region+'/'+filename)
        img=nb.load(fullname)
        d=img.get_fdata()
        img_arr=np.array(d)
        img_arr[np.isnan(img_arr)]=0
        img_arr = img_arr[welch_filter_array == 1]
#        print(np.shape(img_arr))
#        img_arr=img_arr*welch_filter_array
        #print(d.shape)
        uid = filename[6:12]
#        print uid
#        img_arr = np.argwhere(img_arr.flatten())
        set1_voxels[c_set_1] = np.reshape(img_arr,(num_values))
        c_set_1 = c_set_1+1
#        category = [k for k,v in b.iteritems() if uid in v]
    for filename in set2:
        fullname=os.path.join(path+str(region_num),"res_wI"+filename+".nii.gz")
    #for filename in os.listdir(os.path.join(path,region_num)):
        #fullname=os.path.join(path,region+'/'+filename)
        img=nb.load(fullname)
        d=img.get_fdata()
        img_arr=np.array(d)
        img_arr[np.isnan(img_arr)]=0
        img_arr = img_arr[welch_filter_array == 1]
 #       print(np.shape(img_arr))
        uid = filename[6:12]
 #       print uid
        #img_arr = np.argwhere(img_arr.flatten())
        set2_voxels[c_set_2] = np.reshape(img_arr,(num_values))
        c_set_2 += 1

    print "Voxel Sum: "+str(np.sum(set1_voxels))+"Max Voxel Value: "+str(np.max(set1_voxels))
#    return num_values,set1_train, set1_test,set2_train, set2_test
    return num_values,set1_voxels, set2_voxels

def create_labels(set1, set2):
    labels=np.append(np.zeros(np.shape(set1)[0]),np.ones(np.shape(set2)[0]))
    return labels

def create_train_test_split(set1_voxels, set2_voxels, test_size):

    labels=np.append(np.zeros(np.shape(set1_voxels)[0]),np.ones(np.shape(set2_voxels)[0]))
    X, y = unison_shuffled_copies(np.append(set1_voxels,set2_voxels,axis=0), labels) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=2222)
    return X_train, X_test, y_train, y_test

def mlp_thesis(set1, set2, nc):
#    print np.shape(set1), np.shape(set2)
    all_sets = np.append(set1,set2,axis=0)

    all_labels = create_labels(set1,set2)
    nf = 'auto'
    print "number of Principal Components: ${}",nc
    pca = PCA(n_components=nc)
    #pca = PCA()
    pca.fit(all_sets)
    all_sets = pca.transform(all_sets)
#    all_sets = np.append(all_sets,labels.T)
    kf = KFold(n_splits=10,shuffle=True,random_state=0)
    kf.get_n_splits(all_sets)
    
    mat = [[0,0],[0,0]]

    for train_index, test_index in kf.split(all_sets):
        #all_labels = all_sets[:,-1]
        #all_sets = all_sets[:,:-1]

        train_set, test_set = all_sets[train_index], all_sets[test_index]
        print("dimensions of train set are {}".format(train_set.shape))
        train_label, test_label = all_labels[train_index], all_labels[test_index]
#    mlp = Classifier(
    #	layers=[Layer("ExpLin", units=nf),Layer("Softmax")], learning_rate=0.001, n_iter=200
    #	) 
        #mlp = MLPClassifier(hidden_layer_sizes=(nf+1, 500, 150, 50, 5), solver='lbfgs',random_state=2)#, learning_rate='adaptive')# shuffle=True)
        # mlp = MLPClassifier(hidden_layer_sizes=(5,25,5))
        #mlp = MLPClassifier(hidden_layer_sizes=(700,500,250), solver='adam')
        mlp = MLPClassifier(hidden_layer_sizes=(900,20), solver='adam')

        mlp.fit(train_set,train_label)
        pred = mlp.predict(test_set)
        mat = mat + confusion_matrix(test_label, pred) 
#        print confusion_matrix(test_label, pred)
#        print classification_report(test_label, pred)
    print mat
    #print mat[0,0]
    prec =  float(mat[0,0])/(float(mat[0,0])+ float(mat[0,1]))
    rec =  float(mat[0,0])/(float(mat[0,0])+float(mat[1,0]))
    f1_score = float(2*mat[0,0])/(float(2*mat[0,0]) + float(mat[0,1]) + float(mat[1,0]))
    print "Precision: ",prec 
    print "Recall: ",rec
    print "F1 score: ",f1_score
    return f1_score, prec, rec


ad = "AD"
cu = "Normal"
em = "EMCI"
lm = "LMCI"

#if (os.path.isfile('all_regions_AD.npy')) and  (os.path.isfile('all_regions_Normal.npy')):
#    set1 = np.load('all_regions_AD.npy')
#    set2 = np.load('all_regions_Normal.npy')
#    mlp_thesis(set1, set2)
#else:
'''
set1 = np.load('all_regions_AD__ADNormal_.npy')
set2 = np.load('all_regions_Normal__ADNormal_.npy')

#set1, set2 = prep_all_regions(ad,cu)
mlp_thesis(set1, set2)

set1 = np.load('all_regions_AD__ADEMCI_.npy')
set2 = np.load('all_regions_EMCI__ADEMCI_.npy')

#set1, set2 = prep_all_regions(ad,em)
mlp_thesis(set1, set2)

#set1 = np.load('all_regions_Normal__NormalEMCI.npy')
#set2 = np.load('all_regions_EMCI__NormalEMCI.npy')
#mlp_thesis(set1, set2)
'''

#set1 = np.load('all_regions_Normal__NormalLMCI_.npy')
#set2 = np.load('all_regions_LMCI__NormalLMCI_.npy')
#set1 = np.load('all_regions_Normal__NormalEMCI_.npy')
#set2 = np.load('all_regions_EMCI__NormalEMCI_.npy')
#for ncomp in range(100,np.shape(set1)[0]+1,50):
#set1 = np.load('all_regions_EMCI__EMCILMCI_.npy')
#set2 = np.load('all_regions_LMCI__EMCILMCI_.npy')
#set1 = np.load('all_regions_AD__ADEMCI_.npy')
#set2 = np.load('all_regions_EMCI__ADEMCI_.npy')
set1 = np.load('all_regions_AD__ADNormal_.npy')
set2 = np.load('all_regions_Normal__ADNormal_.npy')
f1 = np.zeros((1,30))
prec = np.zeros((1,30))
rec = np.zeros((1,30))
for i in range(0,30):
#set1, set2 = prep_all_regions(cu,em)
    f1[0,i],prec[0,i],rec[0,i] = mlp_thesis(set1, set2,np.shape(set1)[0]-50)

mean_f1 = np.mean(f1)
std_dev_f1 = np.std(f1)
print "f1 mean:${} ",mean_f1," f1 std_dev: ${}", std_dev_f1

mean_prec = np.mean(prec)
std_dev_prec = np.std(prec)
print "prec mean:${} ",mean_prec," prec std_dev: ${}", std_dev_prec
mean_rec = np.mean(rec)
std_dev_rec = np.std(rec)
print "rec mean:${} ",mean_rec," rec std_dev: ${}", std_dev_rec
'''
set1 = np.load('all_regions_Normal__NormalLMCI_.npy')
set2 = np.load('all_regions_LMCI__NormalLMCI_.npy')

#set1, set2 = prep_all_regions(cu,lm)

mlp_thesis(set1, set2)

set1 = np.load('all_regions_EMCI__EMCILMCI_.npy')
set2 = np.load('all_regions_LMCI__EMCILMCI_.npy')

#set1, set2 = prep_all_regions(em,lm)
mlp_thesis(set1, set2)

'''
#mlp_thesis(set1, set2)
