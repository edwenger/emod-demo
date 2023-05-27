import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from emodlib.malaria import IntrahostComponent


def run_challenge(duration):

    asexuals = np.zeros(duration)
    gametocytes = np.zeros(duration)

    ic = IntrahostComponent.create()
    ic.challenge()

    for t in range(duration):
        ic.update(dt=1)
        asexuals[t] = ic.parasite_density
        gametocytes[t] = ic.gametocyte_density

    return pd.DataFrame({'days': range(duration),
                         'parasite_density': asexuals,
                         'gametocyte_density': gametocytes}).set_index('days')


def multiple_challenges(n_people, duration):
    
    asexuals = np.zeros((n_people, duration))
    gametocytes = np.zeros((n_people, duration))
    pp = [IntrahostComponent.create() for _ in range(n_people)]
    _ = [p.challenge() for p in pp]

    for t in range(duration):
        for i, p in enumerate(pp):
            p.update(dt=1)
            asexuals[i, t] = p.parasite_density
            gametocytes[i, t] = p.gametocyte_density
            
    da = xr.DataArray(dims=('individual', 'time', 'channel'),
                      coords=(range(n_people), range(duration), ['parasite_density', 'gametocyte_density']))
                      
    da.loc[dict(channel='parasite_density')] = asexuals
    da.loc[dict(channel='gametocyte_density')] = gametocytes
                      
    return da


def plot_timeseries(df):
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    df.plot(ax=ax, color=dict(parasite_density='navy', gametocyte_density='darkgreen'))
    ax.set(yscale='log')
    fig.set_tight_layout(True)
    

if __name__ == '__main__':

    df = run_challenge(duration=300)
    print(df.head(10))

    plot_timeseries(df)

    plt.show()
    
