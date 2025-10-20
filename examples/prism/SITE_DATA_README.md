# PRISM Site-Specific Data Files

The PRISM cohort study included three sites in Uganda with different malaria transmission intensities:

## Data Files

- `prism_cleaned_nagongera.csv` - **High transmission** (Tororo district)
- `prism_cleaned_walukuba.csv` - **Medium transmission** (Jinja district)  
- `prism_cleaned_kihihi.csv` - **Low transmission** (Kanungu district)

## Site Statistics

### Nagongera (High Transmission)
- Households: 107
- Participants: 489
- Observations: 23,044
- **Microscopy prevalence: 22.22%**
- Median parasite density (when positive): 3,020 parasites/µL
- Date range: 2011-08-13 to 2017-07-06

### Walukuba (Medium Transmission)
- Households: 117
- Participants: 454
- Observations: 11,097
- **Microscopy prevalence: 5.97%**
- Median parasite density (when positive): 3,680 parasites/µL
- Date range: 2011-07-29 to 2016-06-30

### Kihihi (Low Transmission)
- Households: 107
- Participants: 478
- Observations: 14,581
- **Microscopy prevalence: 23.03%**
- Median parasite density (when positive): 12,906 parasites/µL
- Date range: 2011-08-14 to 2016-07-06

## Using Site-Specific Data

All analysis and simulation scripts have a `SITE` configuration parameter at the top:

```python
# Site selection (change this to view different sites)
SITE = 'nagongera'  # Options: 'nagongera', 'walukuba', 'kihihi'
```

Simply change this parameter to analyze a different site.

## Current Configuration

The household simulation (`07_household_simulation.py`) is currently configured for:
- **Site**: Nagongera (high transmission)
- **Reference household**: h_125006301 (7 members, 26.2% prevalence)
