#!/bin/env python3

import uproot
import pandas as pd
import numpy as np
import boost_histogram as bh
import matplotlib.pyplot as plt
import pickle
import atlasplots as ap


folder = '/eos/atlas/atlascerngroupdisk/perf-egamma/InclusivePhotons'
branches = ['evtWeight', 'mcWeight', 'mcTotWeight', 'yWeight', 'y_passOQ', 'y_pt', 'y_eta', 
            'y_isTruthMatchedPhoton', 'y_truth_convType', 'y_Rhad1', 'y_Rhad',       #do we need y_convType thats not truth?
            'y_Reta', 'y_weta2', 'y_Rphi', 'y_wtots1', 'y_weta1', 'y_fracs1', 'y_deltae', 'y_Eratio', 'y_f1']



# LOADING IN ROOT FILES
    
df_mc20a_gjfull = ap.fileloader(folder+'/mc20_gammajet_v09/PyPt8_inf_mc20a_p5536_Rel22_AB22.2.97_v09.root',branches,entry_stop=None)
df_mc20a_jjfull = ap.fileloader(folder+'/mc20_jetjet_v09/Py8_jetjet_mc20a_p5536_Rel22_AB22.2.97_v09.root',branches,entry_stop=None)
df_mc20d_gjfull = ap.fileloader(folder+'/mc20_gammajet_v09/PyPt8_inf_mc20d_p5536_Rel22_AB22.2.97_v09.root',branches,entry_stop=None)
df_mc20d_jjfull = ap.fileloader(folder+'/mc20_jetjet_v09/Py8_jetjet_mc20d_p5536_Rel22_AB22.2.97_v09.root',branches,entry_stop=None)
df_mc20e_gjfull = ap.fileloader(folder+'/mc20_gammajet_v09/PyPt8_inf_mc20e_p5536_Rel22_AB22.2.97_v09.root',branches,entry_stop=None)
df_mc20e_jjfull = ap.fileloader(folder+'/mc20_jetjet_v09/Py8_jetjet_mc20e_p5536_Rel22_AB22.2.97_v09.root',branches,entry_stop=None)


# COMBINING a, d, e SLICES

df_mc20_gjfull = pd.concat([df_mc20a_gjfull, df_mc20d_gjfull, df_mc20e_gjfull])
df_mc20_jjfull = pd.concat([df_mc20a_jjfull, df_mc20d_jjfull, df_mc20e_jjfull])


# PASSING OBJECT QUALITY

df_mc20_gjfull = df_mc20_gjfull[df_mc20_gjfull.y_passOQ]
df_mc20_jjfull = df_mc20_jjfull[df_mc20_gjfull.y_passOQ]


# TRUTH MATCHING THE REAL AND FAKE photons

df_mc20_gj = df_mc20_gjfull[df_mc20_gjfull.y_isTruthMatchedPhoton]
# df_mc20_gj.index = list(range(len(df_mc20_gj)))   #resetting indices
df_mc20_jj = df_mc20_jjfull[~df_mc20_jjfull.y_isTruthMatchedPhoton]
# df_mc20_jj.index = list(range(len(df_mc20_gj)))   #resetting indices


# CREATING GOOD WEIGHTS WITHOUT PHOTON SFs

df_mc20_gjfull['goodWeight'] = df_mc20_gjfull['mcTotWeight']/df_mc20_gjfull['yWeight']
df_mc20_jjfull['goodWeight'] = df_mc20_jjfull['mcTotWeight']/df_mc20_jjfull['yWeight']


# CREATING HadLeakage VARIABLE:

df_mc20_gjfull['HadLeakage'] = ap.makehadlist(df_mc20_gjfull)
df_mc20_jjfull['HadLeakage'] = ap.makehadlist(df_mc20_jjfull)


# APPLYING ETA PRESELECTION
#1.37 <= |eta| <= 1.52 & |eta| < 2.37 

etapresel_gj = ((abs(df_mc20_gj.y_eta) <= 1.37) | (abs(df_mc20_gj.y_eta) >= 1.52)) & (abs(df_mc20_gj.y_eta) < 2.37)
etapresel_jj = ((abs(df_mc20_jj.y_eta) <= 1.37) | (abs(df_mc20_jj.y_eta) >= 1.52)) & (abs(df_mc20_jj.y_eta) < 2.37)

df_mc20_gj = df_mc20_gj[etapresel_gj]
df_mc20_jj = df_mc20_jj[etapresel_jj]


# APPLYING E_T PRESELECTION

# ET > 25 GeV --> signal enriched sample
# ? is E_T the same as p_T? TO BE DONE LATER


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


#TAKING ONLY WANTED COLUMNS

df_mc20_gj_clean = df_mc20_gj[['mcTotWeight','goodWeight','y_pt',
                               'y_eta', 'y_isTruthMatchedPhoton', 'y_truth_convType',
                               'HadLeakage', 'y_Reta', 'y_weta2', 'y_Rphi', 'y_wtots1', 
                               'y_weta1', 'y_fracs1', 'y_deltae', 'y_Eratio', 'y_f1', 
                               'HadLeakage_stand', 'y_Reta_stand', 'y_Rphi_stand', 'y_weta2_stand',
                               'y_wtots1_stand', 'y_weta1_stand', 'y_fracs1_stand', 'y_deltae_stand',
                               'y_Eratio_stand', 'y_f1_stand']]
df_mc20_gj_clean.index = list(range(len(df_mc20_gj_clean)))   #resetting indices

df_mc20_jj_clean = df_mc20_jj[['mcTotWeight','goodWeight','y_pt',
                               'y_eta', 'y_isTruthMatchedPhoton', 'y_truth_convType',
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

df_conv = df_mc20_clean[df_mc20_clean.y_truth_convType == 0]
df_conv.index = list(range(len(df_conv))) #resetting indicies
df_unconv = df_mc20_clean[df_mc20_clean.y_truth_convType == 1]
df_unconv.index = list(range(len(df_unconv))) #resetting indicies


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


# WRITING FILES

# ap.picklewrite(df_mc20_gj_clean,'ALLdf_mc20_gj.pickle')  #gj
# ap.picklewrite(df_mc20_jj_clean,'ALLdf_mc20_jj.pickle')  #jj

ap.picklewrite(df_mc20_clean,'ALLdf_mc20.pickle',filepath='/eos/user/k/kyklazek/SWAN_projects/UVic-Photons/photon-study/picklefiles/')  #all combined and shuffled

ap.picklewrite(df_evenc,'ALLdf_mc20_conv_even.pickle',filepath='/eos/user/k/kyklazek/SWAN_projects/UVic-Photons/photon-study/picklefiles/')
ap.picklewrite(df_oddc,'ALLdf_mc20_conv_odd.pickle',filepath='/eos/user/k/kyklazek/SWAN_projects/UVic-Photons/photon-study/picklefiles/')
ap.picklewrite(df_evenu,'ALLdf_mc20_unconv_even.pickle',filepath='/eos/user/k/kyklazek/SWAN_projects/UVic-Photons/photon-study/picklefiles/')
ap.picklewrite(df_oddu,'ALLdf_mc20_unconv_odd.pickle',filepath='/eos/user/k/kyklazek/SWAN_projects/UVic-Photons/photon-study/picklefiles/')
