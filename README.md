# photon-study

Kyran Klazek-Schryer UVic Summer Project, working with using Neural Networks for ATLAS photon detection.

Single Photon Events (signal) were taken from the mc20 files in `/eos/atlas/atlascerngroupdisk/perf-egamma/InclusivePhotons/mc20_gammajet_v09/` and Dijet Events (background) were taken similarly from `/eos/atlas/atlascerngroupdisk/perf-egamma/InclusivePhotons/mc20_jetjet_v09/`. These events were compiled and cuts applied in the `makepickledata-gc_time.py` script. This script was run on the `lxplus` condor batch system, using the `.sub` file contained in the `condor/makepickledata/` folder. Many of the functions made for this photon study exploration can be found in the `atlasplots.py` file, which is also imported into each script as `ap`.

## script: `makepickledata-gc_time.py`
Note: `makepickledata.py` is an old deprecated version of this script, without garbage collection, nor timing print statements.

To run `makepickledata-gc_time.py`, see `condor/makepickledata/`. This script takes three arguments, the index of entry start (can be negative), the index of entry stop (can be negative) and the name of the version of outputted files. The entry start and stops determine how many events are skimmed from each file. If all events from all files are wanted, `entry_start=None` and `entry_stop=None`.

An example of running the script, with the last 1 000 000 events from each ntuple file looks like: `python makepickledata-gc_time.py -1000000 None v_last1mil`

An example of running the script, with the all events from each ntuple file looks like: `python makepickledata-gc_time.py None None v_full`

The script performs these tasks, in order:
- imports the `.root` ntuple files from the `/eos/atlas/atlascerngroupdisk/perf-egamma/InclusivePhotons/` folder to pandas DataFrames, with the desried selection of events, and the desired selection of branches, stated in line 31 of the script.
- combines the `mc20a`, `mc20d`, and `mc20e` imported DataFrames (DFs), to full gamma-jet and dijet DFs.
- only selects events which pass object quality, branchname `y_passOQ`.
- an $E_T$ preselection is applied, only taking events with $`y_pt`>25GeV$. This signal enriching cut was taken from [Florian Kirfel's slides](https://indico.cern.ch/event/1076972/contributions/4531976/attachments/2310873/3932455/Photon%20ID%20ML.pdf) from a similar project.
- An $\eta$ preselection is also applied, in accordance to the detector sensitivity. Only events with ($1.37 ≥ |`y_eta`|$ OR $|`y_eta`| ≥ 1.52$) & ($|`y_eta`| < 2.37$) were kept.
- The real and fake photons are truth matched, such that gamma-jet samples are only `y_isTruthMatchedPhoton==True` and dijet samples are only `y_isTruthMatchedPhoton==False`.
- New weights are created, labelled `goodWeight`, which are comprised of the `mcTotWeight`/`yWeight`, to remove photon Scale Factors.
- The new variable `HadLeakage` is created out of the variables `y_Rhad` and `y_Rhad1`, with certain $\eta$ selections. To see in detail, look at the `atlasplots.py` function `makehadlist`.
- 

-----------------------------------------------------------------------------------------

Next, the files --> smaller as to make them appropriate size for training script. this is `sizereduction.py`, i think?

## script: `TRAININGTEST/sizereduction.py`
- what it do
- saves into `TRAININGTEST/data`

-----------------------------------------------------------------------------------------
Then, I trained the NN.. explain the architecture, got from John McGowan, link here etc. in `TRAININGTEST/NNtraining.py`

## script: `TRAININGTEST/NNtraining.py`
- it does something too
- explain all the plots it makes and where it saves them

-----------------------------------------------------------------------------------------
Then, mostly doing plotting in Notebooks. Explain the plotting, kind of in order. Mention the new script, if have it done. Talk about what plots were made how and where they were saved.   PUT IN ATLASPLOTS.PY SOMEWHERE

Then after, you can make a list of Notebooks that weren't mentioned and what they do / what their purpose is. And mention if theyre depracated/old code.

The `condor` folder should also be added, with all the submission code fro batch jobs.