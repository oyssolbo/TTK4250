import numpy as np
# set to 'sim' or 'real'
RUN = 'real'

# set to False for speedup, skips sanity checks for in dataclasses
DEBUG = False and __debug__

# set to True for speedup as matrix exponential is approximated in
# ESKF.get_van_loan_matrix()
DO_APPROXIMATIONS = True

# max unning time set to np.inf to run through all the data
MAX_TIME = 600
