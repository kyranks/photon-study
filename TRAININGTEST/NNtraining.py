import time
starttime = time.time()

import uproot
import pandas as pd
import numpy as np
import boost_histogram as bh
import matplotlib.pyplot as plt
import pickle
import gc

import tensorflow as tf
import os
import sys
from sklearn.metrics import roc_curve, auc
from datetime import datetime
import statistics as stat
import ROOT
import hist
from hist import Hist
now = datetime.utcnow().strftime("%y%m%d%H%M%S")

# setting path
sys.path.append('/eos/user/k/kyklazek/SWAN_projects/UVic-Photons/photon-study/')
# importing
import atlasplots as ap

print('modules imported')
print('time taken: ', time.time() - starttime)
print()

if sys.argv[1] == 'conv':
    converted = True
elif sys.argv[1] == 'unconv':
    converted = False
#------------------------------------------------------------------------------------------------------------------------
#CHANGABLE PARAMETERS

version = 'full_v02'
# converted = True   #True or False
first = 1000000   #only takes first <first> events
firststr = '1mil'  #for filenames
stand = True   #if using standardized variable or not, True or False
weightstr = 'finalWeight'    #either 'finalWeight' for E_T and eta renorm. or 'goodWeight' for not  (can be equWeight if have made that)
method = 'train'
#------------------------------------------------------------------------------------------------------------------------


print('loading data...');print()
stepstart = time.time()
currentpath = '/eos/user/k/kyklazek/SWAN_projects/UVic-Photons/photon-study/TRAININGTEST/'

if converted == True:
    convlabel = 'conv'
    with open(currentpath+'data/'+version+'_ec2mil_a.pickle', 'rb') as file:    #this can be changed to ec2mil_z or concat. both or ec1mil
        df_even = pickle.load(file)
        df_even = df_even.iloc[:first]
        print(currentpath+'data/'+version+'_ec2mil_a.pickle loaded')
    with open(currentpath+'data/'+version+'_oc2mil_a.pickle', 'rb') as file:    #this can be changed to ec2mil_z or concat both or ec1mil
        df_odd = pickle.load(file)
        df_odd = df_odd.iloc[:first]
        print(currentpath+'data/'+version+'_oc2mil_a.pickle loaded')
elif converted == False:
    convlabel = 'unconv'
    with open(currentpath+'data/'+version+'_eu2mil_a.pickle', 'rb') as file:    #this can be changed to eu2mil_z or concat both or eu1mil
        df_even = pickle.load(file)
        df_even = df_even.iloc[:first]
        print(currentpath+'data/'+version+'_eu2mil_a.pickle loaded')
    with open(currentpath+'data/'+version+'_ou2mil_a.pickle', 'rb') as file:    #this can be changed to eu2mil_z or concat both or eu1mil
        df_odd = pickle.load(file)
        df_odd = df_odd.iloc[:first]
        print(currentpath+'data/'+version+'_ou2mil_a.pickle loaded')
else:
    print('code should never get here. fix!')

gc.collect()
first = firststr
print('files loaded with first '+str(first)+' events each, time taken:',time.time()-stepstart);print()

outmodel = version+convlabel+'_'+str(first)
if stand == False:
    outmodel = version+convlabel+'_'+str(first)+'nonstand'
print('outmodel label:',outmodel)


features_stand = ['HadLeakage_stand', 'y_Reta_stand', 'y_Rphi_stand', 'y_weta2_stand',
                   'y_wtots1_stand', 'y_weta1_stand', 'y_fracs1_stand', 'y_deltae_stand',
                   'y_Eratio_stand', 'y_f1_stand']
features = ['HadLeakage', 'y_Reta', 'y_weta2', 'y_Rphi', 'y_wtots1', 
            'y_weta1', 'y_fracs1', 'y_deltae', 'y_Eratio', 'y_f1']



#separating features and labels################################################
stepstart = time.time()

if stand == True:
    featlist = features_stand
elif stand == False:
    featlist = features

train_features_even = np.array(df_even[featlist])
train_labels_even   = np.array(df_even['y_isTruthMatchedPhoton'])
train_weights_even  = np.array(df_even[weightstr])
test_features_even = np.array(df_odd[featlist])
test_labels_even   = np.array(df_odd['y_isTruthMatchedPhoton'])
test_weights_even  = np.array(df_odd[weightstr])

train_features_odd = np.array(df_odd[featlist])
train_labels_odd   = np.array(df_odd['y_isTruthMatchedPhoton'])
train_weights_odd  = np.array(df_odd[weightstr])
test_features_odd = np.array(df_even[featlist])
test_labels_odd   = np.array(df_even['y_isTruthMatchedPhoton'])
test_weights_odd  = np.array(df_even[weightstr])

print('features and labels separated, time taken:', time.time()-stepstart);print()


#############################################################################

print('a naive prediction (all true) would result in:')
print('without weights (even, odd):', np.sum(train_labels_even)/len(train_labels_even), np.sum(test_labels_even)/len(test_labels_even))
print('with weights (even, odd):', np.sum(np.multiply(train_labels_even*1,train_weights_even))/np.sum(train_weights_even), np.sum(np.multiply(test_labels_even*1,test_weights_even))/np.sum(test_weights_even))
print()

############################################################################
#functions#

def train_model(train_features,train_labels,train_weights,valid_features,valid_labels,valid_weights):
    """
    Trains a binary classification model using the provided training and validation data.

    Args:
        train_features (numpy.ndarray): The input features for training the model.
        train_labels (numpy.ndarray): The target labels for training the model.
        train_weights (numpy.ndarray): The sample weights for training the model.
        valid_features (numpy.ndarray): The input features for validating the model.
        valid_labels (numpy.ndarray): The target labels for validating the model.
        valid_weights (numpy.ndarray): The sample weights for validating the model.

    Returns:
        tuple: A tuple containing the trained model and a dictionary of loss and accuracy values during training.

    """

    inputs = tf.keras.layers.Input(dtype=tf.float64,shape=(train_features.shape[1],))
    Dx = inputs
    Dx = tf.keras.layers.BatchNormalization()(Dx)
    Dx = tf.keras.layers.Dense(512,input_shape = (train_features.shape[1],))(Dx)
    Dx = tf.keras.layers.LeakyReLU()(Dx)
    Dx = tf.keras.layers.Dense(512,input_shape = (train_features.shape[1],))(Dx)
    Dx = tf.keras.layers.LeakyReLU()(Dx)
    Dx = tf.keras.layers.Dense(512,input_shape = (train_features.shape[1],))(Dx)
    Dx = tf.keras.layers.Dense(1,activation="sigmoid")(Dx)
    f = tf.keras.models.Model([inputs],[Dx])

    print('Classifier Summary')
    f.summary()

    ###########################################
    # Train model
    ############################################
    f.compile(loss = tf.keras.losses.binary_crossentropy, optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5),metrics=["accuracy"])

    callback_es = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience = 10)
    callback_nan = tf.keras.callbacks.TerminateOnNaN()

    losses = {"train":[],"accuracy":[],"validation":[]}
    hist = f.fit(x=train_features,
           y=train_labels,
           sample_weight=np.abs(train_weights),
           callbacks=[callback_nan,callback_es],
           batch_size=256,
           epochs=200,
           verbose=1,
           validation_data=(valid_features,valid_labels,np.abs(valid_weights)))
    losses["train"].append(hist.history['loss'])
    losses["accuracy"].append(hist.history['accuracy'])
    losses["validation"].append(hist.history['val_loss'])

    return f,losses


def plot_losses(losses,name,label):
    """
    Plots the training and validation losses over epochs.

    Args:
        losses (dict): A dictionary containing the training and validation losses.
        name (str): The name of the plot file to be saved.
        label (str): The label for the plot.

    Returns:
        None
    """
    train_losses = losses['train']
    validation_losses = losses['validation']
    fig1, ax1 = plt.subplots(1)
    plt.plot(range(len(train_losses[0])), train_losses[0], color="blue",label="Training Loss")
    plt.plot(range(len(validation_losses[0])), validation_losses[0],color="red",label='Validation Loss')
    ax1.legend(loc='upper right')
    ax1.title.set_text(label)
    ax1.text(0.85, 0.77, 'ATLAS Internal', weight='bold',horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    fig1.savefig("/eos/user/k/kyklazek/SWAN_projects/UVic-Photons/photon-study/TRAININGTEST/plots/"+name+".pdf",bbox_inches='tight')
    plt.clf()
    plt.cla()
    
    
def plot_roc(model,test_features,test_labels,name,label):
    """
    Plots the Receiver Operating Characteristic (ROC) curve for a binary classification model.

    Args:
        model (tf.keras.Model): The trained model used for prediction.
        test_features (numpy.ndarray): The input features for testing the model.
        test_labels (numpy.ndarray): The true labels for the test data.
        name (str): The name of the plot file to be saved.
        label (str): The label for the plot.

    Returns:
        None

    Raises:
        None

    """
    y_pred = model.predict(test_features).ravel()

    # Compute the false positive rate, true positive rate, and thresholds for the ROC curve
    fpr, tpr, thresholds = roc_curve(test_labels,y_pred)

    f,ax = plt.subplots(1)
    plt.plot([1,0],[0,1],'k--')
    plt.plot(1-fpr, tpr)
    plt.xlabel('Bkg Rejection')
    plt.ylabel('Signal Efficiency')
    plt.title(label)
    plt.text(0.85, 0.95, 'ATLAS Internal', weight='bold',horizontalalignment='center', verticalalignment='center')
    #plt.show()
    f.savefig("/eos/user/k/kyklazek/SWAN_projects/UVic-Photons/photon-study/TRAININGTEST/plots/"+name+".pdf",bbox_inches='tight')
    plt.clf()
    plt.cla()


def validation_plot(model,test_features,train_features,train_labels,test_labels,train_weights,test_weights,name):
    """
    Creates a validation plot comparing the output of a neural network model for signal and background events.

    Args:
        model (tf.keras.Model): The trained model used for prediction.
        test_features (numpy.ndarray): The input features for testing the model.
        train_features (numpy.ndarray): The input features for training the model.
        train_labels (numpy.ndarray): The target labels for training the model.
        test_labels (numpy.ndarray): The target labels for testing the model.
        train_weights (numpy.ndarray): The sample weights for training the model.
        test_weights (numpy.ndarray): The sample weights for testing the model.
        name (str): The name of the plot file to be saved.

    Returns:
        None

    Raises:
        None

    """
    y_pred = model.predict(test_features).ravel()
    y_pred_train = model.predict(train_features).ravel()
    ROOT.gStyle.SetOptStat(0)
    hist_sig = ROOT.TH1F("h_sig",";NN output; Normalized Events",20,0,1)
    hist_bkg = ROOT.TH1F("h_bkg",";NN output;Normalized Events",20,0,1)
    hist_sig_train = ROOT.TH1F("h_sig_train","NN output;Normalized Events",20,0,1)
    hist_bkg_train = ROOT.TH1F("h_bkg_train","NN output;Normalized Events",20,0,1)

    for i in range(len(test_features)):
        if (test_labels[i]):
            hist_sig.Fill(y_pred[i],test_weights[i])
        else:
            hist_bkg.Fill(y_pred[i],test_weights[i])

    for i in range(len(train_features)):
        if (train_labels[i]):
            hist_sig_train.Fill(y_pred_train[i],train_weights[i])
        else:
            hist_bkg_train.Fill(y_pred_train[i],train_weights[i])

    can = ROOT.TCanvas("can","can",800,600)

    hist_sig.SetLineColor(1)
    hist_sig.SetFillColor(1)
    hist_sig.SetFillStyle(3005)
    hist_sig.SetMarkerStyle(0)
    hist_bkg.SetLineColor(8)
    hist_bkg.SetFillColor(8)
    hist_bkg.SetMarkerStyle(0)
    hist_sig_train.SetMarkerColor(1)
    hist_sig_train.SetLineColor(1)
    hist_sig_train.SetMarkerSize(1)
    hist_sig_train.SetMarkerStyle(8)
    hist_bkg_train.SetMarkerColor(209)
    hist_bkg_train.SetLineColor(209)
    hist_bkg_train.SetMarkerStyle(8)
    h = hist_bkg.DrawNormalized("HIST E")
    h.SetMinimum(0)
    h.SetMaximum(h.GetMaximum()*3)
    hist_sig.DrawNormalized("HIST SAME E")
    hist_bkg_train.DrawNormalized("pe SAME")
    hist_sig_train.DrawNormalized("pe SAME")
    leg = ROOT.TLegend(0.6,0.6,0.95,0.9)
    leg.SetTextSize(.035)
    leg.AddEntry(hist_sig_train,"MC Signal (train)","pe")
    leg.AddEntry(hist_bkg_train,"MC Background (train)","pe")
    leg.AddEntry(hist_sig,"MC Signal (test)","F")
    leg.AddEntry(hist_bkg,"MC Background (test)","F")
    leg.SetBorderSize(0)
    leg.Draw()
#     ROOT.ATLASLabel(0.2,0.88,"Internal");   
    can.Print("/eos/user/k/kyklazek/SWAN_projects/UVic-Photons/photon-study/TRAININGTEST/plots/"+name+".pdf")
    

def ratioplot(df, branchname, binedges, title='Plot Title', savefig=False, ratiominmax=(0.8,1.2), leftright=(0,100), log=True, weightname='finalWeight',uncert_type="bar"):
    '''plots and saves a figure of the ratio of bkg and signal
    for the selected branchname, in the bins of binedges
    ratiominmax is the min and max for the ratio axis. tuple
    
    savefig -- False or 'fig_name'
    ratiominmax -- min and max for the ratio axis. tuple
    leftright -- tuple min max of x axis
    log -- True or False for y axis
    weightname -- string
    uncert_type -- "line" or "bar"
    '''
    bh_sig = ap.makebhvar(df,branchname,binedges,boolslice=df.y_isTruthMatchedPhoton,weightname=weightname)
    bh_bkg = ap.makebhvar(df,branchname,binedges,boolslice=~df.y_isTruthMatchedPhoton,weightname=weightname)
    
    h1 = Hist(bh_sig)
    h2 = Hist(bh_bkg)
    
    fig, axs = plt.subplots(2, 1, sharex=True, sharey=False,figsize=(10,8), gridspec_kw={'height_ratios': [3, 1]})
    axes_dict = {"main_ax": axs[0], "ratio_ax": axs[1]}
    main_ax_artists, sublot_ax_arists = h1.plot_ratio(
        h2,
        rp_ylabel=r"Ratio",
        rp_num_label="signal",
        rp_denom_label="background",
        rp_uncert_draw_type=uncert_type,  # line or bar
        ax_dict = axes_dict
    )
    
    top = 1.1*max(max(h2.values()),max(h1.values()))
    bottom = 0; 
    left = leftright[0]; right = leftright[1]
    axs[0].set_xlabel("")
    axs[1].set_xlabel(branchname,position=(1,1),horizontalalignment='right')
    axs[0].set_ylabel("Number of events",position=(0,1),horizontalalignment='right')
    axs[0].legend()
    
    axs[0].set_ylim(bottom=bottom); axs[0].set_ylim(top=top); 
    axs[0].set_xlim(left=left); axs[0].set_xlim(right=right)
    axs[1].set_xlim(left=left); axs[1].set_xlim(right=right)
    axs[1].set_ylim(bottom=ratiominmax[0]); axs[1].set_ylim(top=ratiominmax[1])
    if log == True:
        axs[0].set_xscale('log'); axs[1].set_xscale('log')
        axs[0].text((right-left)*0.02+left, (top-bottom)*0.9+bottom, 'ATLAS',style = 'italic',fontweight='bold',fontsize=20,horizontalalignment='left')
        axs[0].text((right-left)*0.05+left, (top-bottom)*0.9+bottom, 'Simulation Internal',fontsize=20,horizontalalignment='left')
        axs[0].text((right-left)*0.02+left, (top-bottom)*0.8+bottom, r'$\sqrt{s} =$'+'13 TeV, 139 fb'+r'$^{-1}$',fontsize=20,horizontalalignment='left')
    else:
        axs[0].text((right-left)*0.02+left, (top-bottom)*0.9+bottom, 'ATLAS',style = 'italic',fontweight='bold',fontsize=20,horizontalalignment='left')
        axs[0].text((right-left)*0.165+left, (top-bottom)*0.9+bottom, 'Simulation Internal',fontsize=20,horizontalalignment='left')
        axs[0].text((right-left)*0.02+left, (top-bottom)*0.8+bottom, r'$\sqrt{s} =$'+'13 TeV, 139 fb'+r'$^{-1}$',fontsize=20,horizontalalignment='left')
    
    axs[0].set_title(title)
    
    fig.subplots_adjust(hspace=0)
    
    if savefig != False:
        fig.savefig("/eos/user/k/kyklazek/SWAN_projects/UVic-Photons/photon-study/TRAININGTEST/plots/"+savefig+".pdf",bbox_inches='tight')
    return


#---------------------------------------------------------------------------------------------------------------------------------------

stepstart = time.time()
print('making ratio plots')

binedgesETA = [0,0.6,0.8,1.15,1.37,1.52,1.81,2.01,2.37]
binedgesET = [0,25,30,35,40,45,50,60,80,100,120,200,500,10000,1000000000]

ratioplot(df_odd,'y_pt',binedgesET, title='odd '+convlabel+', first '+str(first),savefig=outmodel+'oddETratio',leftright=(20,1000))
ratioplot(df_even,'y_pt',binedgesET, title='even '+convlabel+', first '+str(first),savefig=outmodel+'evenETratio',leftright=(20,1000))
df_even['abs_eta']=abs(df_even.y_eta)
df_odd['abs_eta']=abs(df_odd.y_eta)
ratioplot(df_odd,'abs_eta',binedgesETA, title='odd '+convlabel+', first '+str(first),savefig=outmodel+'oddETAratio',leftright=(0,2.37), log=False)
ratioplot(df_even,'abs_eta',binedgesETA, title='even '+convlabel+', first '+str(first),savefig=outmodel+'evenETAratio',leftright=(0,2.37), log=False)

print('ratio plots made, time taken:',time.time()-stepstart);print()
#---------------------------------------------------------------------------------------------------------------------------------------



if method == 'train':
    stepstart = time.time()
    f_even, losses_even = train_model(train_features_even,train_labels_even,train_weights_even,test_features_even,test_labels_even,test_weights_even)
    f_odd, losses_odd = train_model(train_features_odd,train_labels_odd,train_weights_odd,test_features_odd,test_labels_odd,test_weights_odd)
    
    print('models trained, time taken:',time.time()-stepstart); print()
    
    stepstart = time.time()
    plot_losses(losses_even,outmodel+"_losses_even","Even NN")
    plot_losses(losses_odd,outmodel+"_losses_odd","Odd NN")
    plot_roc(f_even,test_features_even,test_labels_even,outmodel+"_roc_even","Even NN")
    plot_roc(f_odd,test_features_odd,test_labels_odd,outmodel+"_roc_odd","Odd NN")
    validation_plot(f_even,test_features_even,train_features_even,train_labels_even,test_labels_even,train_weights_even,test_weights_even, outmodel+"_validation_even")
    validation_plot(f_odd,test_features_odd,train_features_odd,train_labels_odd,test_labels_odd,train_weights_odd,test_weights_odd, outmodel+"_validation_odd")

    
    y_pred_even = f_even.predict(test_features_even).ravel()
    fpr_even, tpr_even, thresholds_even = roc_curve(test_labels_even,y_pred_even)
    auc_nom_even = auc(fpr_even,tpr_even)

    y_pred_odd = f_odd.predict(test_features_odd).ravel()
    fpr_odd, tpr_odd, thresholds_odd = roc_curve(test_labels_odd,y_pred_odd)
    auc_nom_odd = auc(fpr_odd,tpr_odd)
    
    print('plots made, time taken:',time.time()-stepstart);print()


datafile = version+convlabel
#####################################################
# Save model 
#####################################################
model_json_classif_even = f_even.to_json()
with open("/eos/user/k/kyklazek/SWAN_projects/UVic-Photons/photon-study/TRAININGTEST/models/fclass_"+outmodel+"_even.json", "w") as json_file:
    json_file.write(model_json_classif_even)
f_even.save_weights("/eos/user/k/kyklazek/SWAN_projects/UVic-Photons/photon-study/TRAININGTEST/models/fclass_"+outmodel+"_even.h5")
np.savez("/eos/user/k/kyklazek/SWAN_projects/UVic-Photons/photon-study/TRAININGTEST/models/losses_"+outmodel+"_even.npz",losses=losses_even)

model_json_classif_odd = f_odd.to_json()
with open("/eos/user/k/kyklazek/SWAN_projects/UVic-Photons/photon-study/TRAININGTEST/models/fclass_"+outmodel+"_odd.json", "w") as json_file:
    json_file.write(model_json_classif_odd)
f_odd.save_weights("/eos/user/k/kyklazek/SWAN_projects/UVic-Photons/photon-study/TRAININGTEST/models/fclass_"+outmodel+"_odd.h5")
np.savez("/eos/user/k/kyklazek/SWAN_projects/UVic-Photons/photon-study/TRAININGTEST/models/losses_"+outmodel+"_odd.npz",losses=losses_odd)

print("EVEN AUC: "+str(auc_nom_even))
print("ODD AUC: "+str(auc_nom_odd))
print("Saved model to disk")
print("Saved model timestamp: models/fclass_"+outmodel)
print("Trained on data: "+datafile)
print("Saved loss: models/losses_"+outmodel+".npz")