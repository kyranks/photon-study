#!/cvmfs/sft.cern.ch/lcg/views/LCG_103swan/x86_64-centos7-gcc11-opt/bin/python3

import uproot
import pandas as pd
import numpy as np
import boost_histogram as bh
import matplotlib.pyplot as plt
import pickle
import atlasplots as ap
import sys
import time
import gc


#args: entry_start, entry_stop, version   #FIGURE THIS OUT WITH argparse LATER

if sys.argv[1] == 'None':
    entry_start = None
else:
    entry_start = int(sys.argv[1])
#----------------------------------------
if sys.argv[2] == 'None':
    entry_stop = None
else:
    entry_stop = int(sys.argv[2])
#----------------------------------------
version = sys.argv[3]


folder = '/eos/atlas/atlascerngroupdisk/perf-egamma/InclusivePhotons'
branches = ['evtWeight', 'mcWeight', 'mcTotWeight', 'yWeight', 'y_passOQ', 'y_pt', 'y_eta', 
            'y_isTruthMatchedPhoton', 'y_convType', 'y_Rhad1', 'y_Rhad',       
            'y_Reta', 'y_weta2', 'y_Rphi', 'y_wtots1', 'y_weta1', 'y_fracs1', 'y_deltae', 'y_Eratio', 'y_f1',
            'y_iso_FixedCutLoose', 'y_iso_FixedCutTight', 'y_iso_FixedCutTightCaloOnly', 'y_topoetcone20ptCorrection', 'y_topoetcone30ptCorrection', 'y_topoetcone40ptCorrection', 'y_IsLoose', 'y_IsLoosePrime2', 'y_IsLoosePrime3', 'y_IsLoosePrime4', 'y_IsTight', 'y_IsEMTight']


# entry_start = -25000001  #None if want default all, neg if counting from end (ex. -1000)
# entry_stop = -1      #None if want defalut all, neg if counting from end (ex. -1)





# LOADING IN ROOT FILES

starttime = time.time()

print('loading in ROOT files:')
print('starting at entry ',entry_start,' ending at entry ',entry_stop)
print('time: ', time.time() - starttime)
    
df_mc20a_gjfull = ap.fileloader(folder+'/mc20_gammajet_v09/PyPt8_inf_mc20a_p5536_Rel22_AB22.2.97_v09.root',branches,entry_stop=entry_stop,entry_start=entry_start)
print("df_mc20a_gjfull:", df_mc20a_gjfull.shape, 'time: ',time.time() - starttime)
df_mc20a_jjfull = ap.fileloader(folder+'/mc20_jetjet_v09/Py8_jetjet_mc20a_p5536_Rel22_AB22.2.97_v09.root',branches,entry_stop=entry_stop,entry_start=entry_start)
print("df_mc20a_jjfull:", df_mc20a_jjfull.shape, 'time: ',time.time() - starttime)
df_mc20d_gjfull = ap.fileloader(folder+'/mc20_gammajet_v09/PyPt8_inf_mc20d_p5536_Rel22_AB22.2.97_v09.root',branches,entry_stop=entry_stop,entry_start=entry_start)
print("df_mc20d_gjfull:", df_mc20d_gjfull.shape, 'time: ',time.time() - starttime)
df_mc20d_jjfull = ap.fileloader(folder+'/mc20_jetjet_v09/Py8_jetjet_mc20d_p5536_Rel22_AB22.2.97_v09.root',branches,entry_stop=entry_stop,entry_start=entry_start)
print("df_mc20d_jjfull:", df_mc20d_jjfull.shape, 'time: ',time.time() - starttime)
df_mc20e_gjfull = ap.fileloader(folder+'/mc20_gammajet_v09/PyPt8_inf_mc20e_p5536_Rel22_AB22.2.97_v09.root',branches,entry_stop=entry_stop,entry_start=entry_start)
print("df_mc20e_gjfull:", df_mc20e_gjfull.shape, 'time: ',time.time() - starttime)
df_mc20e_jjfull = ap.fileloader(folder+'/mc20_jetjet_v09/Py8_jetjet_mc20e_p5536_Rel22_AB22.2.97_v09.root',branches,entry_stop=entry_stop,entry_start=entry_start)
print("df_mc20e_jjfull:", df_mc20e_jjfull.shape, 'time: ',time.time() - starttime)

print('files imported.')
print('time taken: ',time.time() - starttime)
print()


# COMBINING a, d, e SLICES

stepstart = time.time()

df_mc20_gjfull = pd.concat([df_mc20a_gjfull, df_mc20d_gjfull, df_mc20e_gjfull])
df_mc20_gjfull.index = list(range(len(df_mc20_gjfull)))

del df_mc20a_gjfull
del df_mc20d_gjfull
del df_mc20e_gjfull

df_mc20_jjfull = pd.concat([df_mc20a_jjfull, df_mc20d_jjfull, df_mc20e_jjfull])
df_mc20_jjfull.index = list(range(len(df_mc20_jjfull)))

del df_mc20a_jjfull
del df_mc20d_jjfull
del df_mc20e_jjfull

gc.collect()

print('a,d,e slices combined. time taken: ',time.time() - stepstart)


# PASSING OBJECT QUALITY & DELETING "full"

stepstart = time.time()

df_mc20_gj = df_mc20_gjfull[df_mc20_gjfull.y_passOQ]
df_mc20_jj = df_mc20_jjfull[df_mc20_jjfull.y_passOQ]

del df_mc20_gjfull
del df_mc20_jjfull
gc.collect()

print('\nafter OQ pass gj: ', df_mc20_gj.shape)
print('after OQ pass jj: ', df_mc20_jj.shape)
print('OQ done. time taken: ', time.time() - stepstart)
print()


# APPLYING E_T PRESELECTION
# ET > 25 GeV --> signal enriched sample

stepstart = time.time()

ETpresel_gj = df_mc20_gj.y_pt > 25
ETpresel_jj = df_mc20_jj.y_pt > 25

df_mc20_gj = df_mc20_gj[ETpresel_gj]
df_mc20_jj = df_mc20_jj[ETpresel_jj]

print('\nafter E_T presel gj: ', df_mc20_gj.shape)
print('after E_T presel jj: ', df_mc20_jj.shape)
print('E_T presel done. time taken: ', time.time() - stepstart)
print()



# APPLYING ETA PRESELECTION
#1.37 <= |eta| <= 1.52 & |eta| < 2.37 

stepstart = time.time()

etapresel_gj = ((abs(df_mc20_gj.y_eta) <= 1.37) | (abs(df_mc20_gj.y_eta) >= 1.52)) & (abs(df_mc20_gj.y_eta) < 2.37)
etapresel_jj = ((abs(df_mc20_jj.y_eta) <= 1.37) | (abs(df_mc20_jj.y_eta) >= 1.52)) & (abs(df_mc20_jj.y_eta) < 2.37)

df_mc20_gj = df_mc20_gj[etapresel_gj]
df_mc20_jj = df_mc20_jj[etapresel_jj]

print('\nafter eta presel gj: ', df_mc20_gj.shape)
print('after eta presel jj: ', df_mc20_jj.shape)
print('eta presel done. time taken: ', time.time() - stepstart)
print()


# TRUTH MATCHING THE REAL AND FAKE photons

stepstart = time.time()

df_mc20_gj = df_mc20_gj[df_mc20_gj.y_isTruthMatchedPhoton]
df_mc20_jj = df_mc20_jj[~df_mc20_jj.y_isTruthMatchedPhoton]

print('\nafter Truth matching gj: ', df_mc20_gj.shape)
print('after Truth matching jj: ', df_mc20_jj.shape)
print('truth matching done. time taken: ', time.time() - stepstart)
print()


# CREATING GOOD WEIGHTS WITHOUT PHOTON SFs

stepstart = time.time()

df_mc20_gj['goodWeight'] = df_mc20_gj['mcTotWeight']/df_mc20_gj['yWeight']
df_mc20_jj['goodWeight'] = df_mc20_jj['mcTotWeight']/df_mc20_jj['yWeight']

print('goodWeight created. time taken: ', time.time() - stepstart)
print()

# CREATING HadLeakage VARIABLE:

stepstart = time.time()

df_mc20_gj['HadLeakage'] = ap.makehadlist(df_mc20_gj)
df_mc20_jj['HadLeakage'] = ap.makehadlist(df_mc20_jj)

print('HadLeakage created. time taken: ', time.time() - stepstart)
print()

# df_mc20_gj.index = list(range(len(df_mc20_gj)))   #resetting indices
# df_mc20_jj.index = list(range(len(df_mc20_gj)))   #resetting indices


# COMBINING, SHUFFLING, RESTTING INDICES

stepstart = time.time()

df_mc20 = pd.concat([df_mc20_gj,df_mc20_jj])
df_mc20 = df_mc20.sample(frac=1).reset_index(drop=True)    #shuffling and resetting indices
del df_mc20_gj
del df_mc20_jj
gc.collect()

print('DFs combined, shuffled. time taken: ', time.time() - stepstart)
print()

# CREATING STANDARDIZED VARIABLES

stepstart = time.time()

branchlist = ap.branchlist[2:]
minmaxlist = ap.minmaxlist[2:]
labellist = ap.labellist[2:]

for i in range(len(branchlist)):
    branchname = branchlist[i]
    label = labellist[i]
    minmax = minmaxlist[i]
    data = np.array(df_mc20[branchname])
    standlist = (data - np.mean(data))/np.std(data)
    df_mc20[branchname+'_stand'] = standlist

gc.collect()
# print('\nstand. var.s created\n')
print('stand. var.s created. time taken: ', time.time() - stepstart)
print()


# SEPARATING CONVERTED AND UNCONVERTED

stepstart = time.time()

df_conv = df_mc20[df_mc20.y_convType > 0]     #!= 0; ==1, or ==2, or ...      ######################
df_conv.index = list(range(len(df_conv))) #resetting indices
df_unconv = df_mc20[df_mc20.y_convType < 1]   # == 0
df_unconv.index = list(range(len(df_unconv))) #resetting indices

#can i delete df_mc20 here?
del df_mc20
gc.collect()

print('conv and unconv separated')
print('conv., unconv. : ', df_conv.shape, df_unconv.shape)
print('time taken: ', time.time() - stepstart)
print()


# REWEIGHTING TO EQUALIZE ETA and E_T DISTRIBUTIONS
#should i do this for the whole combined? and it'll equal out? for now, separatig conv. and unconv.
# also to-do: vectorize the ap.reweight function

stepstart = time.time()

binedgesETA = [0,0.6,0.8,1.15,1.37,1.52,1.81,2.01,2.37]
binedgesET = [0,25,30,35,40,45,50,60,80,100,120,200,500,10000,float("inf")]
# From Florian: [20, 25), [25, 30), [30, 40), [40, 50), [50, 60), [60, 80), [80, 100), [100, 250), [250, 1000), [1000, ∞)

df_conv['abs_eta'] = abs(df_conv.y_eta)
df_conv['newWeight'] = ap.weightmaker(df_conv,'abs_eta',binedgesETA,'goodWeight')
df_conv['finalWeight'] = ap.weightmaker(df_conv,'y_pt', binedgesET, 'newWeight')

print('\nETA and E_T weights applied to converted')
print('time taken: ', time.time() - stepstart)
secondtime = time.time()

df_unconv['abs_eta'] = abs(df_unconv.y_eta)
df_unconv['newWeight'] = ap.weightmaker(df_unconv,'abs_eta',binedgesETA,'goodWeight')
df_unconv['finalWeight'] = ap.weightmaker(df_unconv,'y_pt', binedgesET, 'newWeight')

print('\nETA and E_T weights applied to unconverted')
print('time taken: ', time.time() - secondtime)
print('time for both: ', time.time() - stepstart)
print()

#SEPARATING EVEN AND ODD

stepstart = time.time()

df_evenc, df_oddc = ap.evenodd(df_conv)
df_evenu, df_oddu = ap.evenodd(df_unconv)

del df_conv
del df_unconv
gc.collect()

print('\neven and odd and conv. and unconv. separated')
print('conv. (even, odd) -- ', df_evenc.shape, df_oddc.shape)
print('unconv. (even, odd) -- ', df_evenu.shape, df_oddu.shape)
print('time taken: ', time.time() - stepstart)
print()

# CLEANUP ------------------------------------------------------------------------------------

stepstart = time.time()

df_evenu = df_evenu[['mcTotWeight','goodWeight', 'finalWeight',
                      'y_pt', 'y_eta', 'y_isTruthMatchedPhoton', 'y_convType',
                               'HadLeakage', 'y_Reta', 'y_weta2', 'y_Rphi', 'y_wtots1', 
                               'y_weta1', 'y_fracs1', 'y_deltae', 'y_Eratio', 'y_f1', 
                               'HadLeakage_stand', 'y_Reta_stand', 'y_Rphi_stand', 'y_weta2_stand',
                               'y_wtots1_stand', 'y_weta1_stand', 'y_fracs1_stand', 'y_deltae_stand',
                               'y_Eratio_stand', 'y_f1_stand',
                    'y_iso_FixedCutLoose', 'y_iso_FixedCutTight', 'y_iso_FixedCutTightCaloOnly', 'y_topoetcone20ptCorrection', 'y_topoetcone30ptCorrection', 'y_topoetcone40ptCorrection', 'y_IsLoose', 'y_IsLoosePrime2', 'y_IsLoosePrime3', 'y_IsLoosePrime4', 'y_IsTight', 'y_IsEMTight']]
df_evenc = df_evenc[['mcTotWeight','goodWeight', 'finalWeight',
                      'y_pt', 'y_eta', 'y_isTruthMatchedPhoton', 'y_convType',
                               'HadLeakage', 'y_Reta', 'y_weta2', 'y_Rphi', 'y_wtots1', 
                               'y_weta1', 'y_fracs1', 'y_deltae', 'y_Eratio', 'y_f1', 
                               'HadLeakage_stand', 'y_Reta_stand', 'y_Rphi_stand', 'y_weta2_stand',
                               'y_wtots1_stand', 'y_weta1_stand', 'y_fracs1_stand', 'y_deltae_stand',
                               'y_Eratio_stand', 'y_f1_stand',
                    'y_iso_FixedCutLoose', 'y_iso_FixedCutTight', 'y_iso_FixedCutTightCaloOnly', 'y_topoetcone20ptCorrection', 'y_topoetcone30ptCorrection', 'y_topoetcone40ptCorrection', 'y_IsLoose', 'y_IsLoosePrime2', 'y_IsLoosePrime3', 'y_IsLoosePrime4', 'y_IsTight', 'y_IsEMTight']]
df_oddu = df_oddu[['mcTotWeight','goodWeight', 'finalWeight',
                      'y_pt', 'y_eta', 'y_isTruthMatchedPhoton', 'y_convType',
                               'HadLeakage', 'y_Reta', 'y_weta2', 'y_Rphi', 'y_wtots1', 
                               'y_weta1', 'y_fracs1', 'y_deltae', 'y_Eratio', 'y_f1', 
                               'HadLeakage_stand', 'y_Reta_stand', 'y_Rphi_stand', 'y_weta2_stand',
                               'y_wtots1_stand', 'y_weta1_stand', 'y_fracs1_stand', 'y_deltae_stand',
                               'y_Eratio_stand', 'y_f1_stand',
                  'y_iso_FixedCutLoose', 'y_iso_FixedCutTight', 'y_iso_FixedCutTightCaloOnly', 'y_topoetcone20ptCorrection', 'y_topoetcone30ptCorrection', 'y_topoetcone40ptCorrection', 'y_IsLoose', 'y_IsLoosePrime2', 'y_IsLoosePrime3', 'y_IsLoosePrime4', 'y_IsTight', 'y_IsEMTight']]
df_oddc = df_oddc[['mcTotWeight','goodWeight', 'finalWeight',
                      'y_pt', 'y_eta', 'y_isTruthMatchedPhoton', 'y_convType',
                               'HadLeakage', 'y_Reta', 'y_weta2', 'y_Rphi', 'y_wtots1', 
                               'y_weta1', 'y_fracs1', 'y_deltae', 'y_Eratio', 'y_f1', 
                               'HadLeakage_stand', 'y_Reta_stand', 'y_Rphi_stand', 'y_weta2_stand',
                               'y_wtots1_stand', 'y_weta1_stand', 'y_fracs1_stand', 'y_deltae_stand',
                               'y_Eratio_stand', 'y_f1_stand',
                  'y_iso_FixedCutLoose', 'y_iso_FixedCutTight', 'y_iso_FixedCutTightCaloOnly', 'y_topoetcone20ptCorrection', 'y_topoetcone30ptCorrection', 'y_topoetcone40ptCorrection', 'y_IsLoose', 'y_IsLoosePrime2', 'y_IsLoosePrime3', 'y_IsLoosePrime4', 'y_IsTight', 'y_IsEMTight']]

#reset indices?, nah rn they're even and odd

print('columns cleaned up. time taken: ', time.time() - stepstart)
print()

# WRITING FILES ------------------------------------------------------------------------------------


ap.picklewrite(df_evenc,version+'df_mc20_conv_even.pickle',filepath='/eos/user/k/kyklazek/SWAN_projects/UVic-Photons/photon-study/picklefiles/')
ap.picklewrite(df_oddc,version+'df_mc20_conv_odd.pickle',filepath='/eos/user/k/kyklazek/SWAN_projects/UVic-Photons/photon-study/picklefiles/')
ap.picklewrite(df_evenu,version+'df_mc20_unconv_even.pickle',filepath='/eos/user/k/kyklazek/SWAN_projects/UVic-Photons/photon-study/picklefiles/')
ap.picklewrite(df_oddu,version+'df_mc20_unconv_odd.pickle',filepath='/eos/user/k/kyklazek/SWAN_projects/UVic-Photons/photon-study/picklefiles/')

print("files saved! - to version ",version)
print('time taken for whole program: ', time.time() - starttime)