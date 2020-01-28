# %%
import pymc3 as pm 
import numpy as np 
import pandas as pd 
import theano.tensor as tt 
import seaborn as sns 
import scipy.stats as stats 
from scipy.special import expit as logistic 
import matplotlib.pyplot as plt 
import arviz as az 
az.style.use('arviz-darkgrid')

# %%
z = np.linspace(-8, 8) 
plt.plot(z, 1/(1+ np.exp(-z)))
plt.xlabel('z')
plt.ylabel('logistic(z)')
plt.show()

# %%
iris = pd.read_csv('/home/brown5628/projects/bayesian-data-analysis/data/iris.csv')
iris.head()
# %%
sns.stripplot(x="species", y="sepal_length", data=iris, jitter=True)

# %%
sns.pairplot(iris, hue='species', diag_kind='kde')

# %%
df = iris.query("species == ('setosa', 'versicolor')")
y_0 = pd.Categorical(df['species']).codes
x_n ='sepal_length'
x_0 = df[x_n].values
x_c = x_0 - x_0.mean() 

# %%
with pm.Model() as model_0:
    α = pm.Normal('α', mu=0, sd=10)
    β = pm.Normal('β', mu=0, sd=10)

    μ = α + pm.math.dot(x_c, β)
    θ = pm.Deterministic('θ', pm.math.sigmoid(μ))
    bd = pm.Deterministic('bd', -α/β)

    yl = pm.Bernoulli('yl', p=θ, observed=y_0)

    trace_0 = pm.sample(1000)

# %%
theta = trace_0['θ'].mean(axis=0)
idx = np.argsort(x_c)
plt.plot(x_c[idx], theta[idx], color='C2', lw=3)
plt.vlines(trace_0['bd'].mean(), 0, 1, color='k')
bd_hpd = az.hpd(trace_0['bd'])
plt.fill_betweenx([0, 1], bd_hpd[0], bd_hpd[1], color='k', alpha=0.5)

plt.scatter(x_c, np.random.normal(y_0, 0.02),
            marker='.', color=[f'C{x}' for x in y_0])
az.plot_hpd(x_c, trace_0['θ'], color='C2')

plt.xlabel(x_n)
plt.ylabel('θ', rotation=0)
# use original scale for xticks
locs, _ = plt.xticks()
plt.xticks(locs, np.round(locs + x_0.mean(), 1))

# %%
df = iris.query("species == ('setosa', 'versicolor')")
y_1 = pd.Categorical(df['species']).codes 
x_n = ['sepal_length', 'sepal_width']
x_1 = df[x_n].values

# %%
with pm.Model() as model_1:
    alpha = pm.Normal('alpha', mu=0, sd=10)
    beta = pm.Normal('beta', mu=0, sd=2, shape=len(x_n))

    mu = alpha + pm.math.dot(x_1, beta)
    theta = pm.Deterministic('theta', 1 / (1 + pm.math.exp(-mu)))
    bd = pm.Deterministic('bd', -alpha/beta[1] - beta[0]/beta[1] * x_1[:,0])

    y1 = pm.Bernoulli('y1', p=theta, observed=y_1)

    trace_1 = pm.sample(2000)

# %%
idx = np.argsort(x_1[:,0])
bd = trace_1['bd'].mean(0)[idx]
plt.scatter(x_1[:,0], x_1[:, 1], c=[f'C{x}' for x in y_0])
plt.plot(x_1[:,0][idx], bd, color='k')

az.plot_hpd(x_1[:,0], trace_1['bd'], color='k')

plt.xlabel(x_n[0])
plt.ylabel(x_n[1])

# %%
probability = np.linspace(.01, 1, 100)
odds = probability / (1 - probability)

_, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(probability, odds, 'C0')
ax2.plot(probability, np.log(odds), 'C1')

ax1.set_xlabel('probability')
ax1.set_ylabel('odds', color='C0')
ax2.set_ylabel('log-odds', color='C1')
ax1.grid(False)
ax2.grid(False)

# %%
df = az.summary(trace_1)
df

# %%

# %%
