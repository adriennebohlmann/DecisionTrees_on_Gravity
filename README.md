# DecisionTrees_on_Gravity

## background
The aim of this excercise: apply decision trees (boosting and ensembles) on a nonlinear regression problem with a threshold in the output variable. 

Specifically: predict bilateral trade between countries with GDP of exporting and importing country, their distance and some binary variables, such as common official language. 
The distribution of the data, theory and numerous empirical studies show that the relation is non-linear, following the law of gravity. 

<img src="https://render.githubusercontent.com/render/math?math=tradeFlow=\frac{GDP_{exp}*GDP_{imp}}{GDP_{world}}*(\frac{tradeCosts}{ML_{exp}*ML_{imp}})^{1-elast}">

There is loads of literature on this, for a practitioner oriented overview highly recommended: Yotov, Yoto; Piermartini, Roberta; Monteiro, José-Antonio; Larch, Mario (2016): An Advanced Guide to Trade Policy Analysis: The Structural Gravity Model. UNCTAD/WTO.

However, there are various problems occuring when applying this equation to data. Above all, the ML-variables cannot be observed. Another point is that trade flow is plagued with Zeros. Intuitively and simplified, they occur as a theshold to be overcome in order to trade at all, because of international trade costs.

There are quite a few solutions to tackle these problems in econometric analysis - see the source cited above!

Here, I offer another perspective and solution on this topic based on decicion trees.

## summary, spoiler alert and why decision trees??

Tradionial gravity analysis highly depends on the transformation of data (usually log) and manipulating or even dismissing zero trade. 

Decision trees do not need transformation of data because they max. entropy / min. variance in a regression problem. Let's try out how they work on this problem!

Decision trees variations are applied to a very simple, traditional gravity analysis. 

The following scikit-learn ensemble methods are applied:
* RandomForestRegressor
* GradientBoostingRegressor
* AdaBoostRegressor

In order to tune and improve results, train-testsplit, grid search, crossvalidation and other scikit-learn methods help a lot too, of course.

Spoiler alert: robustness is a problem. I blame this mainly on the lack of explanatory power of the theshold itself.

# Results

Randomness is in various places - intentionally so!  
Therefore results will vary from running the code, but overall results from cross-validated results should not vary much.

Only one tweak:  
Zero trade is about 25% in the data. In order to smoothe results, statified fold in cross validation ensures that this share is equal in both test and train data.

mean cross validated Test-R2 from 
* RandomForestRegressor(n_estimators=200, max_depth=52): 63% 
* RandomForestRegressor(n_estimators=377, max_depth=13); 62%
* GradientBoostingRegressor(loss = 'huber', max_depth = 8, n_estimators = 200): 69%
* GradientBoostingRegressor(loss = 'huber', max_depth = 8, n_estimators = 200): 68%
* AdaBoostRegressor(base_estimator = DecisionTreeRegressor(criterion = 'friedman_mse', max_depth=34)): 69%

To put this in perspective:  
* Test-R2 are from out-of sample. Training R2 usually overfits with an R2 over 90%.
* There is no transformation of the data (apart from removed missing observations).
* The Zero-trade observations are quite numerous (25%), with no obvious explanatory variable explaining the theshold included. Nevertheless, mean predictive power is comparable to explanatory power of traditional regression analysis.




