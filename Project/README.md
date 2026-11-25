# Macro regimes and factor allocation

Detects macroeconomic regimes (PCA + Gaussian mixture) from FRED indicators and tests whether a regime-aware tabular Q-learning agent allocates across Fama-French factors better than equal weights or a simple rule.

## Install
pip install -r requirements.txt

## Run (from the code/ folder, in this order)
1. 01_data_download.ipynb: downloads FRED and Kenneth French data into data/
2. 02_regimes.ipynb: PCA, GMM, regimes, NBER comparison (tables in results/)
3. 03_backtest.ipynb: simple regime rule vs equal weights
4. 04_qlearning.ipynb: Q-learning with and without regimes, sensitivity
5. 05_figures.ipynb: figures in figures/

Shared code is in code/helpers.py. Random seeds are fixed.
Running time: a few minutes in total (the sensitivity analysis in notebook 04 takes 1-2 minutes). The provided Nuvolos workspace is sufficient.

## Data
FRED (Federal Reserve Bank of St. Louis) and the Kenneth R. French Data Library. Saved copies are in data/. Re-downloading may give slightly different values because FRED revises its series. Licence: [to complete after reading each site's terms].

## Tools
Parts of the code were debugged and refined with the assistance of Claude (Anthropic).