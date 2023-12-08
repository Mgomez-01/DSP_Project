# DSP_Project
Repo to contain all the work we do for the DSP project at the University of Utah 

## Commit from Miguel to start the repo

[Link to the Piazza post](https://piazza.com/class/llj01zibs9l1ky/post/44)

[Link to octave band filtering PDF](https://dspfirst.gatech.edu/chapters/07ztrans/labs/OctaveBandFiltLab/OctaveBandFiltLab.pdf)

[Link to the Z-transform stuff from piazza](https://dspfirst.gatech.edu/chapters/07ztrans/overview.html)


# NOTE!!!
You must first add the util directory to `/LabP14_Exercise5/pythonStuff/` *BEFORE* running any of the python code!! This was accidentally left out of the pythonStuff directory and is crucial to the execution of all code calling the setup of the filters!!

# requirements
must have the following installed python libs, shown under the program

## Run testOBF.py
- running tesOBF.py will call the filter class and sets things up. will call to calc coeffs and then apply the filter to data.
```
python3 testOBF.py
```
``` pip install numpy matplotlib rich```


~## Run create_FIR_filter.py~
~- running create_FIR_filter.py like shown below will show the help option.~
~```~
~python3 create_FIR_filter.py -h~
~```~
~~
~- example run with 32 elements and fmin=2, fmax=8~
~```~
~python3 create_FIR_filter.py -N 32 -F 2 8~
~```~
~~


## Run create_FIR_filter.py
- running Create_FIR_FIlter_Example.py
```
python3 Create_FIR_FIlter_Example.py
```


## Run testOBF.py
- running test_audio_input.py will call the filter class and sets things up, then start attempting to read from a default mic. may not work exactly since the thresholds are dependent on my default, but worth a shot if you want to take a look. will call to calc coeffs and then apply the filter to data live as it comes through the mic.
```
python3 test_audio_input.py
```

``` pip install numpy matplotlib rich pyaudio```

## this is another version of the above. 
```
python3 live_audio_vis.py
```

``` pip install numpy matplotlib pyaudio```



