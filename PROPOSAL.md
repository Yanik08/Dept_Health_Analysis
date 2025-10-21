# Title : Dept Health Analysis
# Category : Simulation & Modeling / Data Analysis & Visualization

# Motivations : 
 My interest in public policy, macroeconomics and specifically growth theory give me incentives to direct my project in a direction where it could be applicable in a real life scenarios for policy makers and independent economic analysis firms / organizations. 
 This brings me to the core of the project; The IMF has recently stated that global dept will excede 100% by 2029 (https://www.reuters.com/world/asia-pacific/imf-sounds-alarm-about-high-global-public-debt-urges-countries-build-buffers-2025-10-15/) , and a contrario to what we are used to, the biggest economies in Europe and the World are impcated by growing dept due to uncertainties related to passed crisis (subprimes, corona). The question on how much dept is managable and where we need to draw a limit is therefore crucial.

# Planned approach and technologies: 
 The model I am willing to create in my project will help Identifying the level of healthy dept in an economy and therefore give responses to concrete cases. It will be grounded in macroeconoimc theory, using the Dept Dynamics Equation combined with the Dept Sustainability Analysis (DSA) framework to stimulate dept evolution under different economic conditions. 
 The model will be data driven, I will use macroeconomic data (sources: IMF, WEO, World Bank,…) and use them as the input for the different countires that can be analysed by the model. The goal is that policy makers can use the model as a tool to see where their country is at today and what is projected for the future. 
 I will be coding using Python, using different libraries. I will use an interactive Streamlit interface for users to input economic variables (growth, interest rate, exchange rate, inflation). I will do a monte carlo simulation to predict future scenarios to estimate if the dept becomes unhealthy and give policy advice on how to avoid these specific scenarios.

# Expected challenges and how I will avoid them: 
 First of all, as it is the beginning of the semester, I am not even sure if this project even is completely doable as I want it to be, therefore I will have to iterate to adapt the model to a way it would fit. I am optimistic that the course + the help of AI tools will bring me to a point of making it possible. 
 I can imagine that data can be a big problem, it is never clean and you mostly don’t get what you want from untreated data, so I may have to frame the data in a way that it is usable for my purpose.

# Success criteria:
 If the system loads a country, computes indicators, runs Monte Carlo, shows dept health score, and policy advice without errors in a convenable amount of time.

# Stretch goals:
 Connect to the IMF or World Bank APIs so the model updated data automatically instead of using static CSV files.
