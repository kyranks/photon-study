# photon-study

Kyran Klazek-Schryer UVic Summer Project, working with using Neural Networks for ATLAS photon detection. This project is on the CERN servers under the path `/eos/user/k/kyklazek/SWAN_projects/UVic-Photons/photon-study/`.

Single Photon Events (signal) were taken from the mc20 files in `/eos/atlas/atlascerngroupdisk/perf-egamma/InclusivePhotons/mc20_gammajet_v09/` and Dijet Events (background) were taken similarly from `/eos/atlas/atlascerngroupdisk/perf-egamma/InclusivePhotons/mc20_jetjet_v09/`. These events were compiled and cuts applied in the `makepickledata-gc_time.py` script. This script was run on the `lxplus` condor batch system, using the `.sub` file contained in the `condor/makepickledata/` folder. Many of the functions made for this photon study exploration can be found in the `atlasplots.py` file, which is also imported into each script as `ap`.

## script: `makepickledata-gc_time.py`
*Note: `makepickledata.py` is an old deprecated version of this script, without garbage collection, nor timing print statements.*

To run `makepickledata-gc_time.py`, see `condor/makepickledata/`. This script takes three arguments, the index of entry start (can be negative), the index of entry stop (can be negative) and the name of the version of outputted files. The entry start and stops determine how many events are skimmed from each file. If all events from all files are wanted, `entry_start=None` and `entry_stop=None`.

An example of running the script, with the last 1 000 000 events from each ntuple file looks like: `python makepickledata-gc_time.py -1000000 None v_last1mil`

An example of running the script, with the all events from each ntuple file looks like: `python makepickledata-gc_time.py None None v_full`

The script performs these tasks, in order:
- imports the `.root` ntuple files from the `/eos/atlas/atlascerngroupdisk/perf-egamma/InclusivePhotons/` folder to pandas DataFrames, with the desried selection of events, and the desired selection of branches, stated in line 31 of the script.
- combines the `mc20a`, `mc20d`, and `mc20e` imported DataFrames (DFs), to full gamma-jet (`gj`) and dijet (`jj`) DFs.
- only selects events which pass object quality, branchname `y_passOQ`.
- an $E_T$ preselection is applied, only taking events with `y_pt`$>25GeV$. This signal enriching cut was taken from [Florian Kirfel's slides](https://indico.cern.ch/event/1076972/contributions/4531976/attachments/2310873/3932455/Photon%20ID%20ML.pdf) from a similar project.
- an $\eta$ preselection is also applied, in accordance to the detector sensitivity. Only events with ($1.37 ≥$ |`y_eta`| OR |`y_eta`| $≥ 1.52$) & (|`y_eta`| $< 2.37$) were kept.
- the real (signal) and fake (backgroud) photons are truth matched, such that gamma-jet samples are only `y_isTruthMatchedPhoton == True` and dijet samples are only `y_isTruthMatchedPhoton == False`.
- new weights are created, labelled `goodWeight`, which are comprised of the `mcTotWeight`/`yWeight`, to remove photon Scale Factors.
- the new variable `HadLeakage` is created out of the variables `y_Rhad` and `y_Rhad1`, with certain $\eta$ selections. To see in detail, look at the `makehadlist` function in `atlasplots.py`.
- the events of the DataFrames are combined into one full DF (gj and jj), shuffled and, indices reset.
- standardized versions of the trainable variables (`'y_Rhad1', 'y_Rhad', 'y_Reta', 'y_weta2', 'y_Rphi', 'y_wtots1', 'y_weta1', 'y_fracs1', 'y_deltae', 'y_Eratio', 'y_f1'`) are created, as is usually done in Neural Network data prep.
- the converted and unconverted events are separated, via `y_convType > 0` and `y_convType == 0`.
- all the signal events were reweighted, such that the binned $E_T$ and $\eta$ distributions of signal and background matched in weighted event count. This creates a new column of weights called `finalWeight`. This was also taken from [Florian's slides](https://indico.cern.ch/event/1076972/contributions/4531976/attachments/2310873/3932455/Photon%20ID%20ML.pdf) to prevent the NN from learning real vs. fake from $E_T$ or $\eta$.
- events were split by Even and Odd indices (event numbers, but after resetting after shuffling). This was done to be able to use [John McGowan's Neural Network training functions and scripts](https://gitlab.cern.ch/atlas-physics/sm/ew/wgamma-vbs-run2/analysis_scripts/-/tree/master/NN_training).
- only certain branches of the final DataFrames were saved, as seen in lines 283-314. These reduced files were then saved to the `picklefiles/` folder, labelled according to version, conversion, odd/even.

*Note: although not in this git repository, the files created from this script are on the `eos` file system under `/eos/user/k/kyklazek/SWAN_projects/UVic-Photons/photon-study/picklefiles`.*

The outputted print statements of the time taken for each step can be seen in `condor/makepickledata/out/`.

*Note: from this script and all others, version `full_v01` contains all events but just certain branches, and version `full_v02` contains all events with added branches for TightID and IsLoose and isolation variables as well.*

-----------------------------------------------------------------------------------------

Next, the files from `picklefiles/` were skimmed into smaller files (containng only the first and last 1 million to 2 million events) so that they could reasonably be used for the Neural Network training script. This was done with the script `TRAININGTEST/sizereduction.py`, and was also run on the condor batch system, as seen in `condor/sizeredu/`.

## script: `TRAININGTEST/sizereduction.py`

To run `TRAININGTEST/sizereduction.py`, see `condor/sizeredu/`. This script takes three arguments: the version name from `makepickledata-gc_time.py` (e.g. `full_v02`), specification of converted/unconverted files (either `conv` or `unconv`), and specification of even/odd files (either `even` or `odd`). The number of events skimmed can also be changed in the script (in the variables `firstlast` and `firstlaststr` in lines 26-27) and is set to 2 million events right now. These variables make the script pick the first and last `firstlast` events, and name the output files with `firstlaststr` string.

To run this script, say for the even unconverted *full* pickle file from version 'full_v02', the command would be: `python TRAININGTEST/sizereduction.py full_v02 unconv even`.

This would output the files `TRAININGTEST/data/full_v02_eu2mil_a.pickle` and `TRAININGTEST/data/full_v02_eu2mil_z.pickle` with the first 2 million and last 2 million events, respectively, of the original full pickle file.

The script performs these tasks in this order
- opens the full pickle file saved by the above script, with the specifications provided.
- separates the first `firstlast` events and last `firstlast` events into two different DataFrames
- save these smaller DataFrames to pickle files in the folder `TRAININGTEST/data/`.

The naming convention of the saved files in `TRAININGTEST/data/` is as follows. For example in the saved file `full_v02_eu2mil_a.pickle`, the `full_v02` represents the version name, the `e` represents **e**ven events, the `u` represents **u**nconverted events, the `2mil` represents the number of events in the file, and the `a` represents that it is the *first* 2mil events from the original file (As opposed to `z` for the *last* 2mil).

The outputted print statements can be seen in `condor/sizeredu/out/`.

-----------------------------------------------------------------------------------------
After these files were created the Nerual Network was trained, using the `NNtraining.py` script. This script takes functions and NN architecture from [John McGowan's NN scripts](https://gitlab.cern.ch/atlas-physics/sm/ew/wgamma-vbs-run2/analysis_scripts/-/tree/master/NN_training), used for other HEP NN training. These scripts were adapted for this photon identification use. This script was run many tiems on the condor batch system, as seen in `condor/NNtrain/`.

## script: `TRAININGTEST/NNtraining.py`

To run `TRAININGTEST/NNtraining.py`, see `condor/NNtrain`. This script takes no arguments (although in some previous version it took one, as seen in `condor/NNtrain/NNtrain.sh`) but has changeable parameters in the code (lines 39-45). The parameters are `version` which is the version string, `converted` which can be `True` or `False` for converted or unconverted, `first` which mean the script only trains on the first `first` events, `firststr` which is a string version of `first` for output filenames, `stand` which can be `True` or `False` for using standardized or regular variables for training, `weightstr` which is the string of the weight name used for training, and `method` which was brought over from John's code and for the purpose of this exploration always equals `'train'`.

To run this script, run `python TRAININGTEST/NNtraining.py`

The script performs these tasks in order:
- it does something too
- explain all the plots it makes and where it saves them
- saves plots (Loss over Epochs, ROC Curves, and Validation Plots) to `TRAININGTEST/plots/`

where it saves everything, and where to find output print files of the training losses etc. (condor/NNtrain/out/)

-----------------------------------------------------------------------------------------
Then, mostly doing plotting in Notebooks. Explain the plotting, kind of in order. Talk about what plots were made how and where they were saved.   PUT IN `atlasplots.py` SOMEWHERE

### - more plots were made (in the `./plots/` folder) via the notebooks `12ExploringAfterNN.ipynb`, `14MoreExploringAfterNN_ROCandPlots.ipynb`, and `03PhotonPlottingFunction-withfullfiles.ipynb`.
### - `atlasplots.py` contains most of the functions defined for this project, with documentation of function use in the definition/docstring of each function. It is imported into most scripts as `ap`.

Then after, you can make a list of Notebooks that weren't mentioned and what they do / what their purpose is. And mention if theyre deprecated/old code.