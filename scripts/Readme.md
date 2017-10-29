# Data preparation
- All the data has been centered to zero mean and scaled to have unity variance.
- Each feature has been clamped to its 95th percentile to avoid outliers.
- The data has been separated in 6 groups, according to the number of jet and the presence or not of mass:
    1. Group 0: 0 jet, mass
    2. Group 1: 0 jet, no mass
    3. Group 2: 1 jet, mass
    4. Group 3: 1 jet, no mass
    5. Group 4: 2-3 jet, mass
    6. Group 5: 2-3 jet, no mass
- In the 6 groups, all nulls values have been removed (since the columns were null)

# Feature generation
We generated polynomial basis to improve the result of the ridge regression for each group.

# Cross-validation
We used cross-validation to:
- find the best degree of the different polynomial basis, by comparison to no
  polynomial basis.
- find the best lambdas for the ridge regression.
- try different combination of optimization, such as add log-normalization and
  removing features with uniform distribution.
  
# Final model
We tried all combinations of our different optimization and compared the results
with cross-validation to finally only retain
three:
- Group separation
- Percentile clamping
- Polynomial augmentation
