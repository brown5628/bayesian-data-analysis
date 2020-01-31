# %% 
import pymc3 as pm 
import numpy as np 
import scipy.stats as stat 
import matplotlib.pyplot as plt 
import arviz as az 

az.style.use('arviz-darkgrid')

# %%
dummy_data = np.loadtxt('/home/brown5628/projects/bayesian-data-analysis/data/dummy.csv')
x_1 = dummy_data[:,0]
y_1 = dummy_data[:,1]

order = 2
x_1p = np.vstack([x_1**i for i in range(1, order+1)])
x_1s = (x_1p - x_1p.mean(axis=1, keepdims=True)) / x_1p.std(axis=1, keepdims=True)
y_1s = (y_1 - y_1.mean()) / y_1.std()
plt.scatter(x_1s[0], y_1s)
plt.xlabel('x')
plt.ylabel('y')

# %%
with pm.Model() as model_1:
    alpha = pm.Normal('alpha', mu=0, sd=1)
    beta = pm.Normal('beta', mu=0, sd=10)
    error = pm.HalfNormal('error', 5)

    mu = alpha + beta * x_1s[0]

    y_pred = pm.Normal('y_pred', mu=mu, sd=error, observed=y_1s)

    trace_1 = pm.sample(2000)

# %%
with pm.Model() as model_p:
    alpha = pm.Normal('alpha', mu=0, sd=1)
    beta = pm.Normal('beta', mu=0, sd=10, shape=order)
    error = pm.HalfNormal('error', 5)

    mu = alpha + pm.math.dot(beta, x_1s)

    y_pred = pm.Normal('y_pred', mu=mu, sd=error, observed=y_1s)

    trace_p = pm.sample(2000)

# %%
x_new = np.linspace(x_1s[0].min(), x_1s[0].max(), 100)

alpha_1_post = trace_1['alpha'].mean()
beta_1_post = trace_1['beta'].mean(axis=0)
y_1_post = alpha_1_post + beta_1_post * x_new 

plt.plot(x_new, y_1_post, 'C1', label='linear model')

alpha_p_post = trace_p['alpha'].mean()
beta_p_post = trace_p['beta'].mean(axis=0)
idx = np.argsort(x_1s[0])
y_p_post = alpha_p_post + np.dot(beta_p_post, x_1s)

plt.plot(x_1s[0][idx], y_p_post[idx], 'C2', label=f'model order {order}')

alpha_p_post = trace_p['alpha'].mean() 
beta_p_post = trace_p['beta'].mean(axis=0)
x_new_p = np.vstack([x_new**i for i in range(1, order+1)])
y_p_post = alpha_p_post + np.dot(beta_p_post, x_new_p)

plt.scatter(x_1s[0], y_1s, c='C0', marker='.')
plt.legend()
plt.show()

# %%
y_1 = pm.sample_posterior_predictive(trace_1, 200, model=model_1)['y_pred']
y_p = pm.sample_posterior_predictive(trace_p, 2000, model=model_p)['y_pred']

# %%
plt.figure(figsize=(8,3))
data = [y_1s, y_1, y_p]
labels = ['data', 'linear model', 'order 2']
for i, d in enumerate(data):
    mean = d.mean()
    err = np.percentile(d, [25,75])
    plt.errorbar(mean, -i, xerr=[[-err[0]], [err[1]]], fmt='o')
    plt.text(mean, -i+.2, labels[i], ha='center', fontsize=14)
plt.ylim([-i-.5, .5])
plt.yticks([])

# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 3), constrained_layout=True)


def iqr(x, a=0):
    return np.subtract(*np.percentile(x, [75, 25], axis=a))


for idx, func in enumerate([np.mean, iqr]):
    T_obs = func(y_1s)
    ax[idx].axvline(T_obs, 0, 1, color='k', ls='--')
    for d_sim, c in zip([y_1, y_p], ['C1', 'C2']):
        T_sim = func(d_sim, 1)
        p_value = np.mean(T_sim >= T_obs)
        az.plot_kde(T_sim, plot_kwargs={'color': c},
                    label=f'p-value {p_value:.2f}', ax=ax[idx])
    ax[idx].set_title(func.__name__)
    ax[idx].set_yticks([])
    ax[idx].legend()
    plt.show()

# %%
x = np.array([4.,5.,6.,9.,12,14.])
y = np.array([4.2, 6., 6., 9., 10, 10.])

plt.figure(figsize=(10,5))
order = [0,1,2,5]
plt.plot(x, y, 'o')
for i in order:
    x_n = np.linspace(x.min(), x.max(), 100)
    coeffs = np.polyfit(x, y, deg=i)
    ffit = np.polyval(coeffs, x_n)

    p =np.poly1d(coeffs)
    yhat = p(x)
    ybar = np.mean(y)
    ssreg = np.sum((yhat-ybar)**2)
    sstot = np.sum((y - ybar)**2)
    r2 = ssreg / sstot 

    plt.plot(x_n, ffit, label=f'order {i}, $R^2$= {r2:.2f}')

plt.legend(loc=2)
plt.xlabel('x')
plt.ylabel('y', rotation=0)

# %%
waic_1 = az.waic(trace_1)
waic_1
# %%
cmp_df = az.compare({'model_1':trace_1,'model_p':trace_p}, method='BB-pseudo-BMA')
cmp_df

# %%
az.plot_compare(cmp_df)

# %%
