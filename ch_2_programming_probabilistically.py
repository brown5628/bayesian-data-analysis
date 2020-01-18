# %%
import matplotlib.pyplot as plt 
import scipy.stats as stats 
import numpy as np 
import pandas as pd 
import seaborn as sns 
import pymc3 as pm 
import arviz as az 


# %%
np.random.seed(123)
trials = 4
theta_real = .35 
data = stats.bernoulli.rvs(p=theta_real, size = trials)

# %%
with pm.Model() as out_first_modedl:
    θ = pm.Beta('θ', alpha=1.,beta=1.)
    y = pm.Bernoulli('y', p=θ, observed=data)
    trace = pm.sample(10000, random_seed=123)

# %%
az.plot_trace(trace)

# %%
az.summary(trace)

# %%
az.plot_posterior(trace)

# %%
