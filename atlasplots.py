import uproot
import pandas as pd
import numpy as np
import boost_histogram as bh
import matplotlib.pyplot as plt
import pickle

#---------------------------------------------------------------------------

branchlist = ['y_Rhad1',
  'y_Rhad',
  'HadLeakage',
  'y_Reta',
  'y_Rphi',
  'y_weta2',
  'y_wtots1',
  'y_weta1',
  'y_fracs1',
  'y_deltae',
  'y_Eratio',
  'y_f1']

labellist = [r'$R_{had1}$',
  r'$R_{had}$',
  r'Had. Leakage',
  r'$R_{\eta}$',
  r'$R_{\phi}$',
  r'$w_{\eta_2}$',
  r'$w_{s \hspace{.4} tot}$',
  r'$w_{s \hspace{.4}3}$',
  r'$f_{side}$',
  r'$\Delta E_s$',
  r'$E_{ratio}$',
  r'$f_1$']

minmaxlist =[(-0.5, 5),
  (-0.5, 5),
  (-0.5, 5),
  (0.1, 1.25),
  (0.1, 1.25),
  (0.002, 0.023),
  (-0.2, 15),
  (0, 0.9),
  (-0.2, 2),
  (-200, 4000),
  (-0.3, 1.1),
  (-0.05, 0.9)]

#----------------------------------------------------------------------------

def fileloader(filepath,branches,TTree='SinglePhoton',entry_stop=1000,entry_start=None):
    '''for Single Photon root files
    returns a pandas DataFrame'''
    
    with uproot.open(filepath) as file:
#         file = uproot.open(filepath)
        fileSP = file[TTree]
        dataframe = fileSP.arrays(branches,library='pd',entry_stop=entry_stop,entry_start=entry_start)
    
        return dataframe


#----------------------------------------------------------------------------

def atlasstyle():
    import matplotlib.pyplot as plt
    plt.style.use('/eos/user/k/kyklazek/ATLAS.mplstyle') 

    import matplotlib.font_manager as font_manager
    font_dirs = ['/eos/user/k/kyklazek/helvetica_font/']
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)

    for font_file in font_files:
        try:
            font_manager.fontManager.addfont(font_file)
            print ("added",font_file)
        except:
            print("cannot add",font_file)
    return


#-------------------------------------------------------------------------

def makebh(dataframe,branchname,minmax,bins=100,boolslice=[],weightname='mcTotWeight'):
    '''a general version of 'makebh', spits out only one boostHistogram
    
    dataframe is the the dataframe (pandas DataFrame)
    branchname is the name of the wanted branch (str)
    minmax is a tuple: (min,max) = (float,float)
    bins is the number of bins (int)
    boolslice is an optional boolean (list/array/Series) argument (boolean mask), if a slice of the data is wanted,
        for example: boolslice = dataframe.y_convType == 0 for converted
    weightname is the (str) of the key for the weights in the DataFrame dataframe.
    
    '''
    minn = minmax[0]
    maxx = minmax[1]
    
    histo = bh.Histogram(bh.axis.Regular(bins,minn,maxx),storage=bh.storage.Weight()) 
    
    if bool(list(boolslice)) == True:
        histo.fill(dataframe[branchname][boolslice], weight=dataframe[weightname][boolslice])
    else:
        histo.fill(dataframe[branchname], weight=dataframe[weightname])
    
    return histo

#--------------------------------------------------------------------------------------


def makebhvar(dataframe,branchname,binedges,boolslice=[],weightname='goodWeight'):
    '''a variable binning version of 'makebh', spits out only one boostHistogram
    
    dataframe is the the dataframe (pandas DataFrame)
    branchname is the name of the wanted branch (str)
    binedges is a list of bin edges, one more than number of bins
    boolslice is an optional boolean (list/array/Series) argument (boolean mask), if a slice of the data is wanted,
        for example: boolslice = dataframe.y_convType == 0 for converted
    weightname is the (str) of the key for the weights in the DataFrame dataframe.
    
    '''
    histo = bh.Histogram(bh.axis.Variable(binedges),storage=bh.storage.Weight()) 
    
    if bool(list(boolslice)) == True:
        histo.fill(dataframe[branchname][boolslice], weight=dataframe[weightname][boolslice])
    else:
        histo.fill(dataframe[branchname], weight=dataframe[weightname])
    
    return histo


#--------------------------------------------------------------------------------------


def ATLAShist(hist1,hist2,label='variable name',minmax=[-2,2],figname=False,log=True,norm=True,save=False,label1=r'single $\gamma$',label2='fake photons'):
    '''
    right now, hist1 is for single photons (gj), hist2 is for fakes (jj)  - boosthistograms
    
    minmax is for x axis: minmax=[minx,maxx]   -  list or tuple
    
    label is variable name. ex.: label=r"$R_{had1}$"
    
    figname is name of .pdf file, if anything is written in.
    
    ***STILL TO DO:***
                - Figure out consistency of top/ bottom for
                                    - log=True, norm=False
                                    - log=False
                - Figure out ATLAS label for log=False
                - add in number for 'xx fb^-1'
                - maybe add option for dpi/format change for savefig
    '''
    
    integral1 = hist1.sum().value
    integral2 = hist2.sum().value
    minn = minmax[0]
    maxx = minmax[1]
    
    plt.close('all')
    plt.figure(figsize=(9,6))
    
    # draw errobars, use the sqrt error. You can use what you want there
    # poissonian 1 sigma intervals would make more sense
    if norm==True:
        plt.errorbar(hist1.axes[0].centers, hist1.view().value/integral1, yerr=np.sqrt(hist1.view().variance)/integral1,
                 fmt='^',mec='purple',mfc='purple',ecolor='purple',ms=7,label=label1)
        plt.errorbar(hist2.axes[0].centers, hist2.view().value/integral2, yerr=np.sqrt(hist2.view().variance)/integral2,
                 fmt='ro',mec='skyblue',mfc='skyblue',ecolor='skyblue',ms=7,label=label2)
    elif norm==False:
        plt.errorbar(hist1.axes[0].centers, hist1.view().value, yerr=np.sqrt(hist1.view().variance),
                 fmt='^',mec='purple',mfc='purple',ecolor='purple',ms=7,label=label1)
        plt.errorbar(hist2.axes[0].centers, hist2.view().value, yerr=np.sqrt(hist2.view().variance),
                 fmt='ro',mec='skyblue',mfc='skyblue',ecolor='skyblue',ms=7,label=label2)
    else:
        print('Please choose True or False for input norm')
        return


    ax = plt.gca()
    ax.set_xlabel(label,position=(1,1),horizontalalignment='right')
    if norm==True:
        ax.set_ylabel("Fraction of Events",position=(0,1),horizontalalignment='right')
    elif norm==False:
        ax.set_ylabel("Number of Events",position=(0,1),horizontalalignment='right')
    ax.legend()
    if log==True:
        plt.yscale('log')
    
    if log==True:
        if norm==True:
            top = 20*max(max(hist1.view().value/integral1),max(hist2.view().value/integral2))
            bottom = 0.000001;
        elif norm==False:
            top = 25*max(max(hist1.view().value),max(hist2.view().value))
            bottom = 0.1;
        left = minn; right = maxx
        ax.set_ylim(bottom=bottom); ax.set_ylim(top=top); ax.set_xlim(left=left); ax.set_xlim(right=right)
    elif log==False:
        if norm==True:
            top = 1.1*max(hist1.view().value/integral1)
            bottom = 0;
        elif norm==False:
            top = 1.8*max(max(hist1.view().value),max(hist2.view().value))
            bottom = 0;
        left = minn; right = maxx
        ax.set_ylim(bottom=bottom); ax.set_ylim(top=top); ax.set_xlim(left=left); ax.set_xlim(right=right)
    else:
        print('Please choose True or False for input log')
        return

    if log==True:
        ax.text((right-left)*0.02+left, (top-bottom)*0.3+bottom, 'ATLAS',style = 'italic',fontweight='bold',fontsize=20,horizontalalignment='left')
        ax.text((right-left)*0.165+left, (top-bottom)*0.3+bottom, 'Simulation Internal',fontsize=20,horizontalalignment='left')
        ax.text((right-left)*0.02+left, (top-bottom)*0.10+bottom, r'$\sqrt{s} =$'+'13.6 TeV, xx fb'+r'$^{-1}$',fontsize=20,horizontalalignment='left')
    elif log==False:
        print('have not figured out ATLAS Label for non-log yet')
        
    if bool(figname)==True:
        plt.savefig(figname+'.png',dpi=360,format='png')
        
    plt.show()
    
        
    return


#-----------------------------------------------------------------------------------------------------------


def ATLAShist4(hist1,hist2,h1nc,h2nc,label='variable name',minmax=[-2,2],figname=False,log=True,norm=True,totalnorm=True,label1=r'single $\gamma$ conv.',label2=r'fake photons conv.',label3=r'single $\gamma$ unconv.', label4=r'fake photons unconv.'):
    '''
    right now, hist1 is for converted single photons (gj), hist2 is for converted fakes (jj)  - boosthistograms
    and h1nc is nonconverted single photons, h2nc is nonconverted fakes
            hist1 and h1nc, as well as hist2 and h2nc, pattern together colour-wise.
    
    minmax is for x axis: minmax=[minx,maxx]   -  list or tuple
    
    label is variable name. ex.: label=r"$R_{had1}$"
    
    figname is name of .pdf file, if anything is written in. If left False, no file will be created
    
    totalnorm = True divides the converted and unconverted rates by total: (sum of conv. & unconv.)
              = False divides each rate by just the sum of itself. (separates conv. and unconv.)
    
    ***STILL TO DO:***
                - maybe add a **kwargs functionality for things like colors, legend labels etc. (how to have a default val.?)
                - FIX log==False
                - Figure out consistency of top/ bottom for
                                    - log=True, norm=False
                                    - log=False
                - Figure out ATLAS label for log=False
                - add in number for 'xx fb^-1'
                - maybe add option for dpi/format change for savefig
    '''
    
    integral1 = hist1.sum().value
    integral2 = hist2.sum().value
    integral1nc = h1nc.sum().value
    integral2nc = h2nc.sum().value
    minn = minmax[0]
    maxx = minmax[1]
    
    if totalnorm==True:
        integral1 = integral1 + integral1nc
        integral2 = integral2 + integral2nc
        integral1nc = integral1
        integral2nc = integral2
    
    plt.close('all')
    plt.figure(figsize=(9,6))
    
    # draw errobars, use the sqrt error. You can use what you want there
    # poissonian 1 sigma intervals would make more sense
    if norm==True:
        plt.errorbar(hist1.axes[0].centers, hist1.view().value/integral1, yerr=np.sqrt(hist1.view().variance)/integral1,
                 fmt='^',mec='purple',mfc='purple',ecolor='purple',ms=7,label=label1)
        plt.errorbar(hist2.axes[0].centers, hist2.view().value/integral2, yerr=np.sqrt(hist2.view().variance)/integral2,
                 fmt='ro',mec='skyblue',mfc='skyblue',ecolor='skyblue',ms=7,label=label2)
        plt.errorbar(h1nc.axes[0].centers, h1nc.view().value/integral1nc, yerr=np.sqrt(h1nc.view().variance)/integral1nc,
                 fmt='^',mec='purple',mfc='none',ecolor='purple',ms=7,label=label3)
        plt.errorbar(h2nc.axes[0].centers, h2nc.view().value/integral2nc, yerr=np.sqrt(h2nc.view().variance)/integral2nc,
                 fmt='ro',mec='skyblue',mfc='none',ecolor='skyblue',ms=7,label=label4)
    elif norm==False:
        plt.errorbar(hist1.axes[0].centers, hist1.view().value, yerr=np.sqrt(hist1.view().variance),
                 fmt='^',mec='purple',mfc='purple',ecolor='purple',ms=7,label=label1)
        plt.errorbar(hist2.axes[0].centers, hist2.view().value, yerr=np.sqrt(hist2.view().variance),
                 fmt='ro',mec='skyblue',mfc='skyblue',ecolor='skyblue',ms=7,label=label2)
        plt.errorbar(h1nc.axes[0].centers, h1nc.view().value, yerr=np.sqrt(h1nc.view().variance),
                 fmt='^',mec='purple',mfc='none',ecolor='purple',ms=7,label=label3)
        plt.errorbar(h2nc.axes[0].centers, h2nc.view().value, yerr=np.sqrt(h2nc.view().variance),
                 fmt='ro',mec='skyblue',mfc='none',ecolor='skyblue',ms=7,label=label4)
    else:
        print('Please choose True or False for input norm')
        return


    ax = plt.gca()
    ax.set_xlabel(label,position=(1,1),horizontalalignment='right')
    if norm==True:
        ax.set_ylabel("Fraction of Events",position=(0,1),horizontalalignment='right')
    elif norm==False:
        ax.set_ylabel("Number of Events",position=(0,1),horizontalalignment='right')
    ax.legend()
    if log==True:
        plt.yscale('log')
    
    if log==True:
        if norm==True:
            top = 20*max(max(hist1.view().value/integral1),max(hist2.view().value/integral2))
            bottom = 0.000001;
        elif norm==False:
            top = 25*max(max(hist1.view().value),max(hist2.view().value))
            bottom = 0.1;
        left = minn; right = maxx
        ax.set_ylim(bottom=bottom); ax.set_ylim(top=top); ax.set_xlim(left=left); ax.set_xlim(right=right)
    elif log==False:
        if norm==True:
            top = 1.1*max(hist1.view().value/integral1)
            bottom = 0;
        elif norm==False:
            top = 1.8*max(max(hist1.view().value),max(hist2.view().value))
            bottom = 0;
        left = minn; right = maxx
        ax.set_ylim(bottom=bottom); ax.set_ylim(top=top); ax.set_xlim(left=left); ax.set_xlim(right=right)
    else:
        print('Please choose True or False for input log')
        return

    if log==True:
        ax.text((right-left)*0.02+left, (top-bottom)*0.3+bottom, 'ATLAS',style = 'italic',fontweight='bold',fontsize=20,horizontalalignment='left')
        ax.text((right-left)*0.165+left, (top-bottom)*0.3+bottom, 'Simulation Internal',fontsize=20,horizontalalignment='left')
        ax.text((right-left)*0.02+left, (top-bottom)*0.10+bottom, r'$\sqrt{s} =$'+'13.6 TeV, xx fb'+r'$^{-1}$',fontsize=20,horizontalalignment='left')
    elif log==False:
        print('have not figured out ATLAS Label for non-log yet')
        
    if bool(figname)==True:
        plt.savefig(figname+'.png',dpi=360,format='png')
        
    plt.show()
    
        
    return


#------------------------------------------------------------------------------------------------------------


def makehadlist(df_name):
    '''Makes the array of values for the combined HadLeakage variable
    where HadLeakage = Rhad if 0.8<|eta|<1.37   and   
                       Rhad1 elsewhere
    df_name is the dataframe
    
    !now vectorized!'''
    
    hadleaklist = [False] * len(df_name['y_eta'])
    
    abseta = np.array(abs(df_name['y_eta']))
    bins = [0, 0.8, 1.37, float('inf')]
    labels = [False,True,False]  #True if Rhad, False if Rhad1
    
    Rhadbool = np.array(pd.cut(abseta,bins,labels=labels,include_lowest=True,ordered=False))  #True if Rhad, False if Rhad1
    Rhad1list = np.array(df_name['y_Rhad1'])
    Rhadlist = np.array(df_name['y_Rhad'])
    
    hadleaklist = np.array([0.]*len(abseta))
    hadleaklist[Rhadbool] = Rhadlist[Rhadbool]
    hadleaklist[~Rhadbool] = Rhad1list[~Rhadbool]
         
    return hadleaklist
        
    
#-------------------------------------------------------------------------------------------------------



def picklewrite(file,filename,filepath='picklefiles/'):
    '''writes <file> to a pickle file with name <filename> (str)
    automatically into picklefiles folder (<filepath>)
    
    if want file in present directory, set filepath='' '''
    pickle.dump(file,open(filepath+filename, 'wb'))
    
    
    
#--------------------------------------------------------------------------------------------------------


def evenodd(df_name):
    '''separates input dataframe df_name into even events and odd events, by index
    returns (even_df, odd_df)'''
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


#-------------------------------------------------------------------------------------------------------


def histvals(hist1,norm=False):
    '''returns the values of the histogram bins of hist (bh_Histogram)
    
    norm = False (default) means its the number of events
    norm = True means its the fraction of events'''
    
    integral1 = hist1.sum().value
    
    if norm:
        vals = hist1.view().value/integral1
    elif ~norm:
        vals = hist1.view().value
    return vals


#--------------------------------------------------------------------------------------------------------



def reweight(df,binedges,ratiolist,PTorETA='y_pt',sigorbkg='sig'):
    '''returns array of weights associated with applying
    binned weighting to the events in df_sig
    
    !now vectorized!
    
    this function chooses which events to apply which weights to
    
    binedges is list of bin edges, length i+1
    ratiolist is list of reweighting ratios for each bin, length i
    PTorETA can be 'y_pt' (default) or 'y_eta' (or 'abs_eta' if made)
    sigorbkg can be 'sig' (default) or 'bkg'
        - this chooses whether the weighting is applied to the signal events or bkg events
    '''
    
    if sigorbkg == 'sig':
        truthsig = True
    elif sigorbkg == 'bkg':
        truthsig = False
    else:
        print('please pick \'sig\' or \'bkg\' for <sigorbkg>')
        return None
    
    np.nan_to_num(ratiolist, copy=False, nan=1., posinf=1.)
    
    eventlist = np.array(df[PTorETA])
    sigbkglist = np.array(df['y_isTruthMatchedPhoton'])
    
    reweightlist = np.array(pd.cut(eventlist,binedges,labels=ratiolist,include_lowest=True,ordered=False))
    
    if truthsig == True:
        reweightlist[~sigbkglist] = 1.
    elif truthsig == False:
        reweightlist[sigbkglist] = 1.
    else:
        print('it should never get here, fix your code')
        return None
    
    return reweightlist


#--------------------------------------------------------------------------------------------------------

def weightmaker(df,variable,binedges,inputweight='goodWeight'):
    '''returns an array of weights, such that
    the signal and background of DataFrame <df> are equal in counts
    of p_t (E_T) bins or eta bins
    
    df is DataFrame
    variable (str) is name of column to equalize counts to (e.g. 'abs_eta' or 'y_pt')
    binedges (list) is a list of the bin edges
    inputweight (str) is the name of the weight to start with (usually either 'goodWeight' or 'newWeight')
    
    reweights the signal events to match background events in binned counts of <variable>
    '''
    
    sigvar = makebhvar(df,variable,binedges,boolslice=df.y_isTruthMatchedPhoton,weightname=inputweight)
    bkgvar = makebhvar(df,variable,binedges,boolslice=~df.y_isTruthMatchedPhoton,weightname=inputweight)
    bkgvals = histvals(bkgvar)    #not normed, right?
    sigvals = histvals(sigvar)
    ratioforsig = bkgvals/sigvals #will have some nans and infs
    df[variable+'Weight'] = reweight(df,binedges,ratioforsig,variable)
    
    return np.multiply( np.array(df[inputweight]), np.array(df[variable+'Weight']) )


#---------------------------------------------------------------------------------------------------------