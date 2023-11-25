# DSP_Project
Repo to contain all the work we do for the DSP project at the University of Utah 

## Commit from Miguel to start the repo

[Link to the Piazza post](https://piazza.com/class/llj01zibs9l1ky/post/44)

[Link to octave band filtering PDF](https://dspfirst.gatech.edu/chapters/07ztrans/labs/OctaveBandFiltLab/OctaveBandFiltLab.pdf)

[Link to the Z-transform stuff from piazza](https://dspfirst.gatech.edu/chapters/07ztrans/overview.html)


## Run testOBF.py
- running tesOBF.py will call the filter class and sets things up. will call to calc coeffs and then apply the filter to data.
```
python3 testOBF.py
```

## Run create_FIR_filter.py
- running create_FIR_filter.py like shown below will show the help option.
```
python3 create_FIR_filter.py -h
```

- example run with 32 elements and fmin=2, fmax=8
```
python3 create_FIR_filter.py -N 32 -F 2 8
```


