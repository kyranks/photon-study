# photon-study

Kyran Klazek-Schryer UVic Summer Project, working with using Neural Networks for ATLAS photon detection.

Single Photon Events (signal) were taken from the mc20 files in `/eos/atlas/atlascerngroupdisk/perf-egamma/InclusivePhotons/mc20_gammajet_v09/` and Dijet Events (background) were taken similarly from `/eos/atlas/atlascerngroupdisk/perf-egamma/InclusivePhotons/mc20_jetjet_v09/`. These events were compiled and cuts applied in the `makepickledata-gc_time.py` script. This script was run on the `lxplus` condor batch system, using the `.sub` file contained in the `condor/makepickledata/` folder.

## script: `makepickledata-gc_time.py`
Note: `makepickledata.py` is an old depracated version of this script, without garbage collection, nor timing print statements.

- HERE: WHAT THIS SCRIPT DOES and what parameters it takes (how to run it)
- etc.

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
Then, mostly doing plotting in Notebooks. Explain the plotting, kind of in order. Mention the new script, if have it done. Talk about what plots were made how and where they were saved. 

Then after, you can make a list of Notebooks that weren't mentioned and what they do / what their purpose is. And mention if theyre depracated/old code.

The `condor` folder should also be added, with all the submission code fro batch jobs.