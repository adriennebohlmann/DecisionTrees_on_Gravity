# DecisionTrees_on_Gravity

draft - comments welcome!

Presentation of key results in the presentation (gravDT.pdf).

## background
The aim of this exercise: apply decision trees (boosting and ensembles) on a nonlinear regression problem with a threshold in the output variable. 

Specifically: predict bilateral trade between countries with GDP of exporting and importing country, their distance and some binary variables, such as common official language. 
The distribution of the data, theory and numerous empirical studies show that the relation is non-linear, following the law of gravity. 

<img src="https://render.githubusercontent.com/render/math?math=tradeFlow=\frac{GDP_{exp}*GDP_{imp}}{GDP_{world}}*(\frac{tradeCosts}{ML_{exp}*ML_{imp}})^{1-elast}">

There is loads of literature on this, for a practitioner oriented overview I highly recommend: Yotov, Yoto; Piermartini, Roberta; Monteiro, José-Antonio; Larch, Mario (2016): An Advanced Guide to Trade Policy Analysis: The Structural Gravity Model. UNCTAD/WTO.

However, there are various problems occurring when applying this equation to data. Above all, the ML-variables cannot be observed. Another point is that trade flow is plagued with Zeros. Intuitively and simplified, they occur as a threshold to be overcome in order to trade at all, because of international trade costs.

There are quite a few solutions to tackle these problems in econometric analysis - see the source cited above!

Here, I offer another possible approach for this topic based on decision trees.

## summary, spoiler alert and why decision trees??

In traditional gravity analysis often the data is (usually log) transformed, Zeros in trade are manipulated or even dismissed. 

Decision trees do not need transformation of data because they max. entropy / min. variance in a regression problem. Let's try out how they perform on this problem!

Decision trees variations are applied to a very simple, traditional gravity analysis with typical explanatory variables for the bilateral trade flow: GDP of exporter and importer, the distance between them, common border, offical language, colonial history, island and landlocked status of a country.

The following scikit-learn ensemble methods are applied:
* RandomForestRegressor
* GradientBoostingRegressor
* Stochastic Gradient Boosting
* AdaBoostRegressor

In order to observe, tune and improve results and check robustness, further methods are applied, such as train-test-split, grid search.

Spoiler alert: robustness is a problem. I blame this mainly on the lack of explanatory power for the threshold.

# Results

Randomness is in various places - intentionally so!  
Therefore results will vary from running the code repeatedly, but mean results from random repetition are reasonably robust.

One tweak:  Zero trade is about 25% in this cross section data. In order to improve results, stratification with a discrete approximation in train-test-split ensures that this share is approx. equal in both test and train data.

Evaluation of results:
* Test-R2 are from out-of sample. Training-R2 is usually over 90% (indicating tree-typical overfitting).
* No transformation of the data (apart from removed missing observations) is necessary - compared to other estimators
* This analysis did not include a variable with substantial explanatory power for the Zeros / the threshold in the output. Test-R2 fluctuates quite strongly, indicating a possible lack of robustness. Nevertheless, mean predictive power is comparable to explanatory power of similar traditional regression analysis.

Comarison to PPML (considered best practice to date):  
Even though robustness is an issue, above methods based on decision trees consistently deliver better out-of-sample predictive performance (test R2) than PPML.

Exemplary test vs predicted trade flow scatter plots from above ensemble methods:


![Figure_y_y](https://user-images.githubusercontent.com/82636544/116657384-99758c80-a98e-11eb-969a-8e6d7ce862dd.png) ![Figure_y_y_loglog](https://user-images.githubusercontent.com/82636544/116657399-9d091380-a98e-11eb-99b3-a65ddee0f53a.png)


