import uproot
import pandas as pd
import numpy as np
import boost_histogram as bh
import matplotlib.pyplot as plt

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

#doesn't work because df_gj and df_jj files aren't in this file

# def makebh(branchname,minmax,bins=100):
#     '''makes bh historgrams
    
#     with ***data frame variable names df_gj and df_jj***
    
#     minmax = (minx,maxx)
    
#     returns gamma-jet histogram and jet-jet histogram, in that order'''
    
    
#     minn = minmax[0]
#     maxx = minmax[1]
    
#     histgj = bh.Histogram(bh.axis.Regular(bins,minn,maxx),storage=bh.storage.Weight()) 
#     exec('histgj.fill(df_gj.'+branchname+', weight=df_gj.evtWeight)')

#     histjj = bh.Histogram(bh.axis.Regular(bins,minn,maxx),storage=bh.storage.Weight()) 
#     exec('histjj.fill(df_jj.'+branchname+', weight=df_jj.evtWeight)')
    
#     return histgj, histjj


#--------------------------------------------------------------------------------------


def ATLAShist(hist1,hist2,label='variable name',minmax=[-2,2],figname=False,log=True,norm=True,save=False):
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
                 fmt='^',mec='purple',mfc='purple',ecolor='purple',ms=7,label=r'single $\gamma$')
        plt.errorbar(hist2.axes[0].centers, hist2.view().value/integral2, yerr=np.sqrt(hist2.view().variance)/integral2,
                 fmt='ro',mec='skyblue',mfc='skyblue',ecolor='skyblue',ms=7,label=r'fake photons')
    elif norm==False:
        plt.errorbar(hist1.axes[0].centers, hist1.view().value, yerr=np.sqrt(hist1.view().variance),
                 fmt='^',mec='purple',mfc='purple',ecolor='purple',ms=7,label=r'single $\gamma$')
        plt.errorbar(hist2.axes[0].centers, hist2.view().value, yerr=np.sqrt(hist2.view().variance),
                 fmt='ro',mec='skyblue',mfc='skyblue',ecolor='skyblue',ms=7,label=r'fake photons')
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