# %%
import pymc3 as pm 
import numpy as np 
import scipy.stats as stats 
import pandas as pd 
import theano.tensor as tt 
import matplotlib.pyplot as plt 
import arviz as az 

az.stype.use('arviz-darkgrid')
np.random.seed(42)

# %%
