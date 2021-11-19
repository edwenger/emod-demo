import yaml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from emodlib.malaria import *


def configure_from_file(config_path):

    with open(config_path) as cfg:
        params = yaml.load(cfg, Loader=yaml.FullLoader)

    # print(yaml.dump(params))

    IntrahostComponent.configure(params)
    
    return params


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


if __name__ == '__main__':

    configure_from_file('config.yaml')

    df = run_challenge(duration=300)
    print(df.head(10))

    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    df.plot(ax=ax, color=dict(parasite_density='navy', gametocyte_density='darkgreen'))
    ax.set(yscale='log')
    fig.set_tight_layout(True)

    plt.show()
    
