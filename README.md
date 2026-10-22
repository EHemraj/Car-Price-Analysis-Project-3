# Car Price Analysis

**Car Price Analysis** is a data project that looks at historical car data to see which features drive car prices up or down. It uses visualisations and a machine learning model to help a business understand the US car market so they can make informed design and pricing decisions.

# ![CI logo](https://codeinstitute.s3.amazonaws.com/fullstack/ci_logo_small.png)

## Summary of Conclusions

Based on the data analysis, several key factors contribute to a car's market price.
Specifically:

- **Performance & Size:** As a car's engine size, horsepower, curb weight, and length increase, its price significantly increases.
- **Drive-Wheel:** Rear-wheel-drive (RWD) vehicles are priced significantly higher than front-wheel-drive (FWD) vehicles.
- **Brand Value:** Luxury manufacturers (such as BMW, Jaguar, Buick, and Porsche) command a statistically significant price premium over mainstream manufacturers.
- **Predictive Modeling:** The initial Extra Trees Regressor model memorised the training data, but has since been refined. By tuning hyperparameters and evaluating different algorithms, the current Version 2 model (a Gradient Boosting Regressor) successfully generalises to unseen data and provides reliable price estimations.

## Dataset Content

This dataset was downloaded from Kaggle. The data was originally compiled from the _1985 Ward's Automotive Yearbook_ and donated to the UCI Machine Learning Repository in 1987. The link to the UCI archive page is found inside the data dictionary and here: https://archive.ics.uci.edu/dataset/10/automobile
Kaggle link here: https://www.kaggle.com/datasets/hellbuoy/car-price-prediction/data

Because the source file did not display unit labels, standard US car measurements were used for this analysis:

- **Price:** US Dollars ($)
- **Horsepower:** hp
- **Curb Weight:** Pounds (lbs)
- **Engine Size:** Cubic inches (cu in)
- **Fuel Economy:** Miles per gallon (mpg) (this was included in the column names ‘city_mpg’ and ‘highway_mpg’)
- **Dimensions (wheelbase, length, width, height):** Inches (in)

The full list of all 26 columns are in [`data/raw data/data_dictionary.csv`]

## Project Setup & Execution

To run this project locally:

1. Clone this repository to your local system.
2. Create a virtual environment (.venv).
3. Install the required dependencies using the terminal command:
   pip3 install -r requirements.txt
4. Open the Jupyter Notebook in the jupyter_notebooks directory and select the Python environment kernel to execute the cells.

## Business Requirements

The goal of this project is to help a new automobile company understand how car prices are determined in the US market so they can successfully compete against existing local and European brands. The main requirements are to explore the data to find out which car features have the greatest effect on price, then build a machine learning model that can predict a car's price based on its specifications.

## Hypothesis & Validation

- **Hypothesis 1 (Engine Specs):** There is a statistically significant correlation between engine size/horsepower and car price.
  - _Validation:_ We visualised this using histograms and scatter plots. Because a Shapiro-Wilk test proved the data did not have a normal distribution, we used a Spearman rank correlation test. This test mathematically proved a significant positive correlation between engine specs and price.
- **Hypothesis 2 (Car Size & Weight):** There is a statistically significant correlation between curb weight/car length and car price.
  - _Validation:_ We plotted these physical features against price using joint plots. After confirming the data was not normally distributed via a Shapiro-Wilk test, we ran a Spearman rank correlation test which proved a significant relationship between vehicle size/weight and price.
- **Hypothesis 3 (Drive-Wheel):** There is a statistically significant difference in price between rear-wheel drive and front-wheel drive cars.
  - _Validation:_ We grouped the data to look at descriptive statistics for each drive-wheel. Because the data (`price`) is not normally distributed, we used the non-parametric Mann-Whitney U test, which confirmed the price difference between RWD and FWD cars is statistically significant.
- **Hypothesis 4 (Brand):** There is a statistically significant difference in price between luxury brands and mainstream brands.
  - _Validation:_ We established a data-informed boundary to split the manufacturers into two groups (luxury vs. mainstream). After calculating the group medians, we ran a Mann-Whitney U test to prove that the price difference between the two brand tiers is statistically significant.

## Project Plan

- **Implementation:**
  - **Planning:** The project was initially mapped out using a Kanban board to track tasks.
  - **ETL (Notebook 1):** We loaded the raw dataset, checked for missing values, cleaned and standardised column names, and exported the clean CSV ready for analysis.
  - **EDA (Notebook 2):** We investigated the clean data using descriptive statistics (calculating mean, median, standard deviation, min/max values, and percentiles) and visualisations to identify outliers, trends, and correlations. We formed hypotheses, explored them visually, and tested them statistically to accept or reject them.
  - **Machine Learning (Notebook 3):** We built a predictive pipeline to estimate car prices. We used regression as we knew the target (`price`) and wanted to predict a continuous numerical value. We evaluated performance using R-squared and Mean Absolute Error (MAE) scores. After the initial hyperparameter tuning resulted in an Extra Trees Regressor that overfitted, a second round of tuning with different parameters was applied. This developed a Gradient Boosting Regressor (Version 2) that generalised to unseen data better than the first model.
  - **Dashboard Creation:** We built an interactive Tableau dashboard, incorporating both the exploratory insights for a non-technical audience and the predictive model's performance metrics.
- **Evaluation:** We evaluated the predictive model using R-squared and Mean Absolute Error (MAE) metrics. We also reviewed the dashboard to make sure the charts clearly explained complex details.
- **Maintenance:** We used Git and GitHub to save and track changes to the project. We wrote Python code to save the machine learning results of the second version of the model into a specific version folder (outputs/v2).
- **Updates:** Future iterations of this project would focus on refining the model by extracting Feature Importances to identify and drop variables that do not actively drive price. Additionally, to make the predictive tool safely viable for a modern business, the existing pipeline would need to be retrained on a contemporary dataset (post-2020) to account for modern inflation, electric vehicle technology, and current emissions standards.

## The Rationale to Map the Business Requirements to the Data Visualisations

| Business Question / Hypothesis                                                                            | Visualisation Used                  | Rationale for Chart Selection                                                                                                                                         |
| :-------------------------------------------------------------------------------------------------------- | :---------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| What does the standard US car market look like compared to luxury outliers?                               | Histograms                          | Shows the spread and distribution of car prices, highlighting where the majority of everyday vehicles sit versus rare, high-end outliers.                             |
| Do core performance specifications (engine size, horsepower) drive up car prices?                         | Scatter Plots                       | Provides visual proof that as engine specs increase, the price increases.                                                                                             |
| Do physical dimensions like curb weight and car length affect the price?                                  | Joint Plots                         | Highlights the trend between vehicle length/weight and price while also showing the distribution shape of those data points.                                          |
| Is there a significant price difference based on the car's drivetrain (rear-wheel vs. front-wheel drive)? | Box Plot and descriptive statistics | Shows a visual comparison of the median price and overall price spread between different drive-wheels, highlighting the higher cost of rear-wheel drive vehicles.     |
| Is there a significant price difference between luxury and mainstream brands?                             | Box Plot                            | Allows easy visual comparison of the median price and overall price spread between different categorical groups, separating the luxury tier from the mainstream tier. |
| Can we reliably estimate a car's market price based on its specific attributes?                           | Actual vs. Predicted Plot           | Visually demonstrates the accuracy of the machine learning model by showing how far predictions are from real prices                                                  |

## Analysis Techniques

- **Descriptive Statistics:** Used to summarise the average (mean, median) and spread (standard deviation, percentiles) of car prices and specs.
- **Hypothesis Testing:** Used the Shapiro-Wilk test to check for normality. Used the Spearman rank correlation test to check the relationship between continuous, non-parametric variables (like engine size and price). Used the Mann-Whitney U test to check for price differences between categorical groups (like drive-wheel and brands).
- **Machine Learning & Tuning:** An Extra Trees Regressor was trained on the data to predict price. As the initial model overfitted (scoring an R-squared of 1.0 on the training data), hyperparameters were changed to restrict tree depth, forcing the model to generalise more to unseen data. This time the best model was a Gradient Boosting Regressor.

## Ethical Considerations, Privacy and Governance

**Data Privacy & Legal (GDPR)**
This dataset uses historical, publicly available data. It contains vehicle specifications and price data. It does not contain any personally identifiable information and is fully GDPR compliant, posing no risk to individual data privacy.

**Bias**
As this dataset dates back to 1985, there are serious concerns around using the predictive model alone as it is today. This model should not be used alone as a predictor of current car prices in the US market. The automotive industry has heavily developed, and the financial situation in the US, buying habits, vehicle regulations and the relationship between the industry and the economy have changed. This model does not account for inflation, modern safety standards, or recent advancements in automotive technology. It can, however, be used to evaluate what car prices were at that time based on features like engine size and horsepower.

**Social Implications**
Using this model as it is today would not be sufficient for a company to enter the current US car market due to the technology and regulatory developments mentioned above. Using it alone may lead to mispricing cars today, which would heavily impact consumer trust.

**Environmental Considerations**
As this dataset is from 1985, it reflects a time before modern emissions standards and the introduction of electric vehicles. The current model does not account for the environmental impact of vehicles, which is a major focus in today's automotive market. Relying on this data today would mean ignoring critical factors like fuel efficiency and emissions compliance, which heavily influence both modern car laws and consumer purchasing choices.

## Dashboard Design

- **Dashboard Link:** https://public.tableau.com/app/profile/e.hemraj/viz/Project3CarPriceAnalysis/Dashboard1

- **Design Strategy:** The dashboard was designed to translate complex Jupyter Notebook outputs into interactive visualisations for a non-technical audience. It allows the audience to filter specific car features and observe how those variables impact car prices within the dataset. It includes narratives alongside technical metrics (R-squared and MAE) to ensure insights regarding model performance and how car specifications affect price are accessible.

## Unfixed Bugs

There are no known bugs in the code.
During the EDA phase, running certain visualisations generated a warnings in the Jupyter Notebook. The warning indicated that some background methods used by our libraries (such as Seaborn or Pandas) will be deprecated in future software versions. As this is a dependency warning about future library updates and does not break the current code or affect the accuracy of the analysis, it was left unresolved.

## Project Retrospective & Practical Challenges

Executing this project presented several practical challenges requiring strategic problem-solving:

- **Statistical & Technical Implementation:** Early challenges included selecting the correct non-parametric statistical tests (Spearman rank correlation and Mann-Whitney U tests) for data that was not normally distributed, as well as understanding and modifying the Python code to build the machine learning pipeline and accurately interpret its results.
- **Dataset Size & Overfitting:** A primary challenge was the limited volume of data (only 205 rows). As the sample was so small, the initially tuned model memorised the training data, achieving an R-squared of 1.0. Resolving this required a second iteration of hyperparameter tuning with stricter parameters to restrict tree depth and force the model to generalise better.
- **Dashboard Interactivity:** Configuring the dashboard filters required significant trial and error. Initially, linking different charts caused misleading visualisations (for example, selecting certain filters incorrectly displayed a specific drive-wheel as making up 100% of mainstream cars). Resolving this required precise filtering to ensure the charts interacted accurately without distorting the data.
- **Contextualising Historical Data:** Adapting an analysis of 1985 historical data to address a modern business requirement meant acknowledging strict limitations. A significant challenge was ensuring the dashboard and documentation clearly stated that while the model proves relationships between car specs and price, the specific financial outputs cannot be safely deployed in today's automotive market.

## Deployment

This dashboard is published and hosted on Tableau Public:

- https://public.tableau.com/app/profile/e.hemraj/viz/Project3CarPriceAnalysis/Dashboard1

- The project code and Jupyter Notebooks are hosted in this GitHub repository.

## Main Data Analysis Libraries

- **Pandas:** Used for core data manipulation, extraction, cleaning, and filtering.
- **Matplotlib & Seaborn:** Used to construct static data visualisations (histograms, box plots, scatter plots, and correlation matrices).
- **Pingouin:** Used to run statistical tests, specifically the Shapiro-Wilk test, Spearman rank correlation, and Mann-Whitney U test.
- **Scikit-Learn:** Used to build, train, and evaluate the Extra Trees Regressor machine learning pipeline.
- **OS:** Used to manage file paths and route ML outputs to version-controlled directories.

## Credits

### Content

- **Code Institute LMS:** Code structures, syntax examples, and data analysis methodologies were used and adapted from the course lessons.
- **Generative AI (Gemini):** Used as a tutoring and linguistic refinement tool. It assisted with writing and adapting code logic, explaining complex statistical concepts, drafting Markdown, formatting this README, checking observations and conclusions, planning the Kanban board tasks, and troubleshooting dashboard logic.
- **GitHub Copilot:** Predictions used for code and comment sections.
- **Previous Coursework:** My complete Project 2 submission was used as the foundational starting point for this project, including the README, data, ETL, EDA, and ML notebooks and Tableau Dashboard. The project was expanded and further developed to meet the Project 3 requirements. Additionally, a code line for imports and the project setup instructions were reused from my Project 1 submission.
- **Jayne Lawley (Project 1):** The code to save the cleaned CSV file was adapted from Jayne Lawley's Project 1 submission.

### Media

- **Code Institute** CI Logo belongs to Code Institute.

## Acknowledgements

I would like to thank my family and my tutors and peers at Code Institute for their encouragement and support.
