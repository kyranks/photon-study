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

if sys.argv[1] == 'None'
    entry_start = None
else:
    entry_start = int(sys.argv[1])
#----------------------------------------
if sys.argv[2] == 'None'
    entry_stop = None
else:
    entry_stop = int(sys.argv[2])
#----------------------------------------
version = sys.argv[3]


folder = '/eos/atlas/atlascerngroupdisk/perf-egamma/InclusivePhotons'
branches = ['evtWeight', 'mcWeight', 'mcTotWeight', 'yWeight', 'y_passOQ', 'y_pt', 'y_eta', 
            'y_isTruthMatchedPhoton', 'y_convType', 'y_Rhad1', 'y_Rhad',       #do we need y_convType thats not truth?
            'y_Reta', 'y_weta2', 'y_Rphi', 'y_wtots1', 'y_weta1', 'y_fracs1', 'y_deltae', 'y_Eratio', 'y_f1']


# entry_start = -25000001  #None if want default all, neg if counting from end (ex. -1000)
# entry_stop = -1      #None if want defalut all, neg if counting from end (ex. -1)





# LOADING IN ROOT FILES

print('loading in ROOT files:')
print('starting at entry ',entry_start,' ending at entry ',entry_stop)
    
df_mc20a_gjfull = ap.fileloader(folder+'/mc20_gammajet_v09/PyPt8_inf_mc20a_p5536_Rel22_AB22.2.97_v09.root',branches,entry_stop=entry_stop,entry_start=entry_start)
print("df_mc20a_gjfull:", df_mc20a_gjfull.shape)
df_mc20a_jjfull = ap.fileloader(folder+'/mc20_jetjet_v09/Py8_jetjet_mc20a_p5536_Rel22_AB22.2.97_v09.root',branches,entry_stop=entry_stop,entry_start=entry_start)
print("df_mc20a_jjfull:", df_mc20a_jjfull.shape)
df_mc20d_gjfull = ap.fileloader(folder+'/mc20_gammajet_v09/PyPt8_inf_mc20d_p5536_Rel22_AB22.2.97_v09.root',branches,entry_stop=entry_stop,entry_start=entry_start)
print("df_mc20d_gjfull:", df_mc20d_gjfull.shape)
df_mc20d_jjfull = ap.fileloader(folder+'/mc20_jetjet_v09/Py8_jetjet_mc20d_p5536_Rel22_AB22.2.97_v09.root',branches,entry_stop=entry_stop,entry_start=entry_start)
print("df_mc20d_jjfull:", df_mc20d_jjfull.shape)
df_mc20e_gjfull = ap.fileloader(folder+'/mc20_gammajet_v09/PyPt8_inf_mc20e_p5536_Rel22_AB22.2.97_v09.root',branches,entry_stop=entry_stop,entry_start=entry_start)
print("df_mc20e_gjfull:", df_mc20e_gjfull.shape)
df_mc20e_jjfull = ap.fileloader(folder+'/mc20_jetjet_v09/Py8_jetjet_mc20e_p5536_Rel22_AB22.2.97_v09.root',branches,entry_stop=entry_stop,entry_start=entry_start)
print("df_mc20e_jjfull:", df_mc20e_jjfull.shape)

print('files imported \n')

# COMBINING a, d, e SLICES

df_mc20_gjfull = pd.concat([df_mc20a_gjfull, df_mc20d_gjfull, df_mc20e_gjfull])
df_mc20_gjfull.index = list(range(len(df_mc20_gjfull)))

df_mc20_jjfull = pd.concat([df_mc20a_jjfull, df_mc20d_jjfull, df_mc20e_jjfull])
df_mc20_jjfull.index = list(range(len(df_mc20_jjfull)))


# PASSING OBJECT QUALITY

df_mc20_gjfull = df_mc20_gjfull[df_mc20_gjfull.y_passOQ]
df_mc20_jjfull = df_mc20_jjfull[df_mc20_gjfull.y_passOQ]


# TRUTH MATCHING THE REAL AND FAKE photons

df_mc20_gj = df_mc20_gjfull[df_mc20_gjfull.y_isTruthMatchedPhoton]
# df_mc20_gj.index = list(range(len(df_mc20_gj)))   #resetting indices
df_mc20_jj = df_mc20_jjfull[~df_mc20_jjfull.y_isTruthMatchedPhoton]
# df_mc20_jj.index = list(range(len(df_mc20_gj)))   #resetting indices


# CREATING GOOD WEIGHTS WITHOUT PHOTON SFs

df_mc20_gj['goodWeight'] = df_mc20_gj['mcTotWeight']/df_mc20_gj['yWeight']
df_mc20_jj['goodWeight'] = df_mc20_jj['mcTotWeight']/df_mc20_jj['yWeight']


# CREATING HadLeakage VARIABLE:

df_mc20_gj['HadLeakage'] = ap.makehadlist(df_mc20_gj)
df_mc20_jj['HadLeakage'] = ap.makehadlist(df_mc20_jj)




# APPLYING ETA PRESELECTION
#1.37 <= |eta| <= 1.52 & |eta| < 2.37 

etapresel_gj = ((abs(df_mc20_gj.y_eta) <= 1.37) | (abs(df_mc20_gj.y_eta) >= 1.52)) & (abs(df_mc20_gj.y_eta) < 2.37)
etapresel_jj = ((abs(df_mc20_jj.y_eta) <= 1.37) | (abs(df_mc20_jj.y_eta) >= 1.52)) & (abs(df_mc20_jj.y_eta) < 2.37)

df_mc20_gj = df_mc20_gj[etapresel_gj]
df_mc20_jj = df_mc20_jj[etapresel_jj]

print('\nafter eta presel gj: ', df_mc20_gj.shape)
print('after eta presel jj: ', df_mc20_jj.shape)

# APPLYING E_T PRESELECTION

# ET > 25 GeV --> signal enriched sample
# ? is E_T the same as p_T? TO BE DONE LATER

ETpresel_gj = df_mc20_gj.y_pt > 25
ETpresel_jj = df_mc20_jj.y_pt > 25

df_mc20_gj = df_mc20_gj[ETpresel_gj]
df_mc20_jj = df_mc20_jj[ETpresel_jj]

print('after E_T presel gj: ', df_mc20_gj.shape)
print('after E_T presel jj: ', df_mc20_jj.shape)
print()

# COMBINING gj AND jj  (for correct standardization)

df_mc20_all = pd.concat([df_mc20_gj,df_mc20_jj])


# CREATING STANDARDIZED VARIABLES

branchlist = ap.branchlist[2:]
minmaxlist = ap.minmaxlist[2:]
labellist = ap.labellist[2:]

for i in range(len(branchlist)):
    branchname = branchlist[i]
    label = labellist[i]
    minmax = minmaxlist[i]
    datagj = np.array(df_mc20_gj[branchname])
    datajj = np.array(df_mc20_jj[branchname])
    data = np.array(df_mc20_all[branchname])
    standlistgj = (datagj - np.mean(data))/np.std(data)
    standlistjj = (datajj - np.mean(data))/np.std(data)
    df_mc20_gj[branchname+'_stand'] = standlistgj
    df_mc20_jj[branchname+'_stand'] = standlistjj

print('\nstand vars created\n')

#TAKING ONLY WANTED COLUMNS

df_mc20_gj_clean = df_mc20_gj[['mcTotWeight','goodWeight','y_pt',
                               'y_eta', 'y_isTruthMatchedPhoton', 'y_convType',
                               'HadLeakage', 'y_Reta', 'y_weta2', 'y_Rphi', 'y_wtots1', 
                               'y_weta1', 'y_fracs1', 'y_deltae', 'y_Eratio', 'y_f1', 
                               'HadLeakage_stand', 'y_Reta_stand', 'y_Rphi_stand', 'y_weta2_stand',
                               'y_wtots1_stand', 'y_weta1_stand', 'y_fracs1_stand', 'y_deltae_stand',
                               'y_Eratio_stand', 'y_f1_stand']]
df_mc20_gj_clean.index = list(range(len(df_mc20_gj_clean)))   #resetting indices

df_mc20_jj_clean = df_mc20_jj[['mcTotWeight','goodWeight','y_pt',
                               'y_eta', 'y_isTruthMatchedPhoton', 'y_convType',
                               'HadLeakage', 'y_Reta', 'y_weta2', 'y_Rphi', 'y_wtots1', 
                               'y_weta1', 'y_fracs1', 'y_deltae', 'y_Eratio', 'y_f1', 
                               'HadLeakage_stand', 'y_Reta_stand', 'y_Rphi_stand', 'y_weta2_stand',
                               'y_wtots1_stand', 'y_weta1_stand', 'y_fracs1_stand', 'y_deltae_stand',
                               'y_Eratio_stand', 'y_f1_stand']]
df_mc20_jj_clean.index = list(range(len(df_mc20_jj_clean)))   #resetting indices


# COMBINING, SHUFFLING RESTTING INDICES

df_mc20_clean_ordered = pd.concat([df_mc20_gj_clean,df_mc20_jj_clean])
df_mc20_clean = df_mc20_clean_ordered.sample(frac=1).reset_index(drop=True)    #shuffling and resetting indices


# SEPARATING CONVERTED AND UNCONVERTED

df_conv = df_mc20_clean[df_mc20_clean.y_convType > 0]     #!= 0; ==1, or ==2, or ...      ######################
df_conv.index = list(range(len(df_conv))) #resetting indices
df_unconv = df_mc20_clean[df_mc20_clean.y_convType < 1]   # == 0
df_unconv.index = list(range(len(df_unconv))) #resetting indices


#SEPARATING EVEN AND ODD

def evenodd(df_name):
    '''separates input dataframe df_name into even events and odd events, by index'''
    evenlist = []
    oddlist = []
    for i in range(len(df_name)):
        if i%2 == 0:
            evenlist.append(i)
        else:
            oddlist.append(i)
            
    even = df_name.loc[evenlist]
    odd =  df_name.loc[oddlist]
    
    return even, odd

df_evenc,df_oddc = evenodd(df_conv)
df_evenu, df_oddu = evenodd(df_unconv)

print('\neven and odd and conv. and unconv. separated')
print('conv. (even, odd) -- ', df_evenc.shape, df_oddc.shape)
print('unconv. (even, odd) -- ', df_evenu.shape, df_oddu.shape)
print()

# REWEIGHTING TO EQUALIZE ETA and E_T DISTRIBUTIONS ------------------------------------------------

binedgesETA = [0,0.6,0.8,1.15,1.37,1.52,1.81,2.01,2.37]
binedgesET = [0,25,30,35,40,45,50,60,80,100,120,200,500,10000]

# FIRST, even unconverted

df_evenu['abs_eta'] = abs(df_evenu.y_eta)
df_evenu['newWeight'] = ap.weightmaker(df_evenu,'abs_eta',binedgesETA,'goodWeight')
df_evenu['finalWeight'] = ap.weightmaker(df_evenu,'y_pt', binedgesET, 'newWeight')

print('\nweights applied to even unconverted\n')

# SECOND, even converted

df_evenc['abs_eta'] = abs(df_evenc.y_eta)
df_evenc['newWeight'] = ap.weightmaker(df_evenc,'abs_eta',binedgesETA,'goodWeight')
df_evenc['finalWeight'] = ap.weightmaker(df_evenc,'y_pt', binedgesET, 'newWeight')

print('\nweights applied to even converted\n')

# THIRD, odd unconverted

df_oddu['abs_eta'] = abs(df_oddu.y_eta)
df_oddu['newWeight'] = ap.weightmaker(df_oddu,'abs_eta',binedgesETA,'goodWeight')
df_oddu['finalWeight'] = ap.weightmaker(df_oddu,'y_pt', binedgesET, 'newWeight')

print('\nweights applied to odd unconverted\n')

#LAST, odd converted

df_oddc['abs_eta'] = abs(df_oddc.y_eta)
df_oddc['newWeight'] = ap.weightmaker(df_oddc,'abs_eta',binedgesETA,'goodWeight')
df_oddc['finalWeight'] = ap.weightmaker(df_oddc,'y_pt', binedgesET, 'newWeight')

print('\nweights applied to odd converted\n')

# CELANUP AGAIN ------------------------------------------------------------------------------------

#c for clean
cdf_evenu = df_evenu[['mcTotWeight','goodWeight', 'finalWeight',
                      'y_pt', 'y_eta', 'y_isTruthMatchedPhoton', 'y_convType',
                               'HadLeakage', 'y_Reta', 'y_weta2', 'y_Rphi', 'y_wtots1', 
                               'y_weta1', 'y_fracs1', 'y_deltae', 'y_Eratio', 'y_f1', 
                               'HadLeakage_stand', 'y_Reta_stand', 'y_Rphi_stand', 'y_weta2_stand',
                               'y_wtots1_stand', 'y_weta1_stand', 'y_fracs1_stand', 'y_deltae_stand',
                               'y_Eratio_stand', 'y_f1_stand']]
cdf_evenc = df_evenc[['mcTotWeight','goodWeight', 'finalWeight',
                      'y_pt', 'y_eta', 'y_isTruthMatchedPhoton', 'y_convType',
                               'HadLeakage', 'y_Reta', 'y_weta2', 'y_Rphi', 'y_wtots1', 
                               'y_weta1', 'y_fracs1', 'y_deltae', 'y_Eratio', 'y_f1', 
                               'HadLeakage_stand', 'y_Reta_stand', 'y_Rphi_stand', 'y_weta2_stand',
                               'y_wtots1_stand', 'y_weta1_stand', 'y_fracs1_stand', 'y_deltae_stand',
                               'y_Eratio_stand', 'y_f1_stand']]
cdf_oddu = df_oddu[['mcTotWeight','goodWeight', 'finalWeight',
                      'y_pt', 'y_eta', 'y_isTruthMatchedPhoton', 'y_convType',
                               'HadLeakage', 'y_Reta', 'y_weta2', 'y_Rphi', 'y_wtots1', 
                               'y_weta1', 'y_fracs1', 'y_deltae', 'y_Eratio', 'y_f1', 
                               'HadLeakage_stand', 'y_Reta_stand', 'y_Rphi_stand', 'y_weta2_stand',
                               'y_wtots1_stand', 'y_weta1_stand', 'y_fracs1_stand', 'y_deltae_stand',
                               'y_Eratio_stand', 'y_f1_stand']]
cdf_oddc = df_oddc[['mcTotWeight','goodWeight', 'finalWeight',
                      'y_pt', 'y_eta', 'y_isTruthMatchedPhoton', 'y_convType',
                               'HadLeakage', 'y_Reta', 'y_weta2', 'y_Rphi', 'y_wtots1', 
                               'y_weta1', 'y_fracs1', 'y_deltae', 'y_Eratio', 'y_f1', 
                               'HadLeakage_stand', 'y_Reta_stand', 'y_Rphi_stand', 'y_weta2_stand',
                               'y_wtots1_stand', 'y_weta1_stand', 'y_fracs1_stand', 'y_deltae_stand',
                               'y_Eratio_stand', 'y_f1_stand']]

# WRITING FILES ------------------------------------------------------------------------------------

# ap.picklewrite(df_mc20_gj_clean,'ALLdf_mc20_gj.pickle')  #gj
# ap.picklewrite(df_mc20_jj_clean,'ALLdf_mc20_jj.pickle')  #jj

# version = 'BACKW25mil_TESTwWEIGHT'

# ap.picklewrite(df_mc20_clean,version+'df_mc20.pickle',filepath='/eos/user/k/kyklazek/SWAN_projects/UVic-Photons/photon-study/picklefiles/')  #all combined and shuffled

ap.picklewrite(cdf_evenc,version+'df_mc20_conv_even.pickle',filepath='/eos/user/k/kyklazek/SWAN_projects/UVic-Photons/photon-study/picklefiles/')
ap.picklewrite(cdf_oddc,version+'df_mc20_conv_odd.pickle',filepath='/eos/user/k/kyklazek/SWAN_projects/UVic-Photons/photon-study/picklefiles/')
ap.picklewrite(cdf_evenu,version+'df_mc20_unconv_even.pickle',filepath='/eos/user/k/kyklazek/SWAN_projects/UVic-Photons/photon-study/picklefiles/')
ap.picklewrite(cdf_oddu,version+'df_mc20_unconv_odd.pickle',filepath='/eos/user/k/kyklazek/SWAN_projects/UVic-Photons/photon-study/picklefiles/')

print("files saved! - to version ",version)