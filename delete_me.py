# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 19:13:28 2023

@author: benja
"""

from util.OctaveBandFilt import OctaveBandFilter, ourOctaveBandFilter
from util.FIR_filter import FIRFilter
import numpy as np
import matplotlib.pyplot as plt
from rich import print

x = 10
f1 = FIRFilter(N=50, fmin=x, fmax=x+10, padding_factor=10)
f1.plot_filter2()
