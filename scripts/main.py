"""
Script to reproduce exemplary figures from

Gerster, Moritz, et al. "Patient-specific network connectivity combined with a
next generation neural mass model to test clinical hypothesis of seizure
propagation." bioRxiv (2021).
"""
from sim import simulation
from plot import fig13, fig14, fig20, fig25

# Sim returns dataframe
df = simulation()

# Plots
fig13(df)
fig14(df)
fig20(df)
fig25(df)
