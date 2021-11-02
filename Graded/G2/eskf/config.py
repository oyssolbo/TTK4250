import numpy as np
# Set to 'sim' or 'real'
RUN = 'sim'

# Set to False for speedup, skips sanity checks for in dataclasses
DEBUG = False and __debug__

# Set to True for speedup as matrix exponential is approximated in
# ESKF.get_van_loan_matrix()
DO_APPROXIMATIONS = True

# Max running time set to np.inf to run through all the data
MAX_TIME = 600
