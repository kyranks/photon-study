#Mine
import uproot
import pandas as pd
import numpy as np
import boost_histogram as bh
import matplotlib.pyplot as plt
import pickle
import gc

import sys
 
# setting path
sys.path.append('/eos/user/k/kyklazek/SWAN_projects/UVic-Photons/photon-study/')
 
# importing
import atlasplots as ap

print('everything imported')


# arguments:  version, conv/unconv, even/odd
#--------------------------------------------------------------------------------
version = sys.argv[1]     #eg. full_v02
converted = sys.argv[2]   #'conv' or 'unconv'
evenodd = sys.argv[3]     #'even' or 'odd'
firstlast = 2000000  #does first and last ___ events
firstlaststr = '2mil'
#--------------------------------------------------------------------------------


with open('/eos/user/k/kyklazek/SWAN_projects/UVic-Photons/photon-study/picklefiles/'+version+'df_mc20_'+converted+'_'+evenodd+'.pickle', 'rb') as file:
    df_evenc = pickle.load(file)
    file.close()

print('opened picklefiles/'+version+'df_mc20_'+converted+'_'+evenodd+'.pickle')
    

# rn made to do first and last 2mil events
    
ap.picklewrite(df_evenc.iloc[:firstlast], version+'_'+evenodd[0]+converted[0]+firstlaststr+'_a.pickle', filepath='/eos/user/k/kyklazek/SWAN_projects/UVic-Photons/photon-study/TRAININGTEST/data/')   #e/o=even/odd, c/u=(un)converted, 2mil=number of events, a=first 2mil events

ap.picklewrite(df_evenc.iloc[-firstlast:], version+'_'+evenodd[0]+converted[0]+firstlaststr+'_z.pickle', filepath='/eos/user/k/kyklazek/SWAN_projects/UVic-Photons/photon-study/TRAININGTEST/data/')  #e/o=even/odd, c/u=(un)converted, 2mil=number of events, z=last 2mil events


print('saved to data/'+version+'_'+evenodd[0]+converted[0]+firstlaststr+'_a.pickle')
print('saved to data/'+version+'_'+evenodd[0]+converted[0]+firstlaststr+'_z.pickle')
