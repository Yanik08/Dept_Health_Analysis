# Title : Dept Health Analysis
# Category : Simulation & Modeling / Data Analysis & Visualization

# Motivations : 
 My interest in public policy, macroeconomics and specifically growth theory give me incentives to direct my project in a direction where it could be applicable in a real life scenarios for policy makers and independent economic analysis firms / organizations. 
 This brings me to the core of the project; The IMF has recently stated that global dept will excede 100% by 2029 (https://www.reuters.com/world/asia-pacific/imf-sounds-alarm-about-high-global-public-debt-urges-countries-build-buffers-2025-10-15/) , and a contrario to what we are used to, the biggest economies in Europe and the World are impcated by growing dept due to uncertainties related to passed crisis (subprimes, corona). The question on how much dept is managable and where we need to draw a limit is therefore crucial.

# Planned approach and technologies: 
 I will build a model that predicts whether a country is in “dept distress” (i.e. is on it’s way of entering a debt crisis) in a given year based on macro variables (Dept to GPD, GDP growth, Interest rate/real interest rate, inflation, primary balance, exchange rate changes, political stability, dummy for crisis years). The target (y) will be binary, either the country is in debt distress or it is not (y = 1: country is in dept distress, y = 0 otherwise). The data used will be the IMF CSV data that is available on their website. I will estimate and compare supervised classification models covered in class, mainly Logistic Regression, Decision Trees, and Random Forest and evaluate them using accuracy, confision matrices, and ROC-AUC.
 To check robustness, I will use the simple dept to GDP forecasting modules, comparing Linear Regression prediciton to the basic macro dept dynamic formula to see whether Machine Learning captures nonlinearities.


# Expected challenges and how I will avoid them :
I will face a few challenges all along the building of this project. First, the most obvious one, I will certainly need some time to be fully comfortable using the dataset from the IMF. Treating and filtering the data for my use will be my first challenge. Then, defining the range of the crisis variable. Forecasting the dept to GDP of next year can be challenging too because the debt dynamics are also influenced by shocks which are not easily captured with Machine Learning. Finally, compairing rhe debt dynamic equation and ML is not easy, one is mechanical and the other is based on different variables. To compaire I will need some assumptions.

# Success criteria:
The model predicts debt crisis and produces economically interpretable results, and debt to gdp regression provides a reasonable robustness check against the standard macro formula.

# Stretch goals:
Connect to the IMF or World Bank APIs so the model updated data automatically instead of using static CSV files.
