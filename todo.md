- Check (ProppantPerFoot, NormalizedOilEUR, NormalizedGasEUR) why negative
- Try with formationAlias: NIOBRARA and other
- Anomalies from the plot
- knnimputer, iterativeimputer
- remove per foot things so make new features with not per

1. Random sample into train test validation (in preprocess)
2. remove outliers using the quantile, after non positives
3. impute the missing values firstly the distance with a constant value outside or something, all others do iterativeimputer (check the distr and scatters if shit do knn)
4. scale using z-score
5. Make ml model trees and train
6. Calculate reltations
7. Present and clean code

Zaker seeket nlp and problem solving

## Presentation

Show differences in correlation between categorical values like niobar

## easiest

make dist a single value that shows en laa
