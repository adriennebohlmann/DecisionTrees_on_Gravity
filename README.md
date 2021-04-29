# DecisionTrees_on_Gravity

## background
The aim of this excercise: apply decision trees (boosting and ensembles) on a nonlinear regression problem with a threshold in the output variable. 

Specifically: predict bilateral trade between countries with GDP of exporting and importing country, their distance and some binary variables, such as common official language. 
The distribution of the data, theory and numerous empirical studies show that the relation is non-linear, following the law of gravity. 

https://render.githubusercontent.com/render/math?math=trade_flow=\frac{GDP_exp*GDP_imp}{GDP_world}*(\frac{trade_costs}{ML_exp*ML_imp})^elast$$

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



