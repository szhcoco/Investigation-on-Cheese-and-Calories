# Investigation on the Relationship Between the Existence of Cheese in Ingredients and Calories

**Name(s)**: Huiting Chen, Coco Sun

**Website Link**: [website](https://szhcoco.github.io/Investigation-on-Cheese-and-Calories/)

## Overview
This is a data science project conducted at UCSD, aiming to investigate on the relationship between the presence of cheese in ingredients and the value of calories. We will walk through the process of data cleaning, checking for missingness dependency, hypothesis testing, building and improving regression models for predictions, as well as performing fairness analysis, to explore the level of significance of cheese and other nutritions that contribute to calories.  


## Introduction
 As a dairy made from milk, cheese is rich in nutritions that are essential in maintaining and improving health. The major nutritions include calcium for the growth of bones and teeth, protein that support the development and repair of bone tissues, and vitamins like A and B12 as sources of healthy fat. The nutritions cheese contains determine its importance for people of all ages, and esepcially for children who are in the phase of developing bones and muscles.

In addition to its functionality contributing to health, cheese in food tickles our taste buds and can be added into salads, noodles (Mac & Cheese!), rice, any proteins, pizza, soup... literally anything to make the meal delicious!

The datasets from a [recipe recommendation website](https://www.food.com/), containing recipes and their ratings since 2008. The Food website
offers a massive collecion of recipes categorized by popularity, ingredients, cuisines, and meal type. It serves as a platform where recipes are published with photos where people could comment, reply, and post photos to show their work after trying them. It also contains information about ingredients thus we are able to identify recipes containing cheese. 

Despite its popularity and importance as an ingredient for food, concerns have been raised about the health issues related to cheese. In addition to its nutritions, cheese are considered as a high-calory ingredient, and over consumption of high-calory food may lead to increase in weight and obesity, diabetes, or heart disease that would risk life. 

Our goal in this project is to answer the question of **whether food with cheese tends to have high calories**, as well as to identify nutritions that play an important role in determining the calories of a recipe. By examining the relationship between nutritions and calories, we will provide insights for home cooks to find appealling recipes, enjoying their food while eating healthily, assist creations of balanced meals, and guide food platforms in improving their recommendation system. Based on the diversity and the large volumn of recipes that the website provides, we consider it as a representative dataset for further analysis. 


#### Introduction to the Columns
 - `RAW_recipes` dataset: 83,782 rows and 12 columns, containing information about each recipe. 

 | Column           | Description                                                                                                                                                                                       |
|:-----------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 'name'           | Recipe name                                                                                                                                                                                       |
| 'id'             | Recipe ID                                                                                                                                                                                         |
| 'minutes'        | Minutes to prepare recipe                                                                                                                                                                         |
| 'contributor_id' | User ID who submitted this recipe                                                                                                                                                                 |
| 'submitted'      | Date recipe was submitted                                                                                                                                                                         |
| 'tags'           | Food.com tags for recipe                                                                                                                                                                          |
| 'nutrition'      | Nutrition information in the form [calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)]; PDV stands for “percentage of daily value” |
| 'n_steps'        | Number of steps in recipe                                                                                                                                                                         |
| 'steps'          | Text for recipe steps, in order                                                                                                                                                                   |
| 'description'    | User-provided description                                                                                                                                                                         |

 - `RAW_interactions` dataset: 731,927 rows and 5 columns, recording users' ratings and comments for different recipes.

| Column      | Description         |
|:------------|:--------------------|
| 'user_id'   | User ID             |
| 'recipe_id' | Recipe ID           |
| 'date'      | Date of interaction |
| 'rating'    | Rating given        |
| 'review'    | Review text         |


To address our problem, we need to focus on column `nutrition` and `ingredient`, from which we will extract new values into columns added to the dataframe later in the cleaning process. 

## Data Cleaning and Exploratory Data Analysis

### Data Cleaning
1. Clean the `RAW_recipes` dataframe before merge
- Removing column `Unnamed: 0` since it does not contain any useful information.
2. Examine the types of the `RAW_recipes` column values  

  | column         | type   |
  |:--------------:|:------:|
  | name           | object |
  | id             | int64  |
  | minutes        | int64  |
  | contributor_id | int64  |
  | submitted      | object |
  | tags           | object |
  | nutrition      | object |
  | n_steps        | int64  |
  | steps          | object |
  | description    | object |
  | ingredients    | object |
  | n_ingredients  | int64  |

We notice that the `nutrition` and `ingredients` are `object`, so we want to do transforming of these columns.

3. Convert each nutrition into single column
- For `nutrition`, each row is a lise-list srings storing values in the form `[calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)]`. Since all values are important for later investigation, we want to get access to them easier, so we: 
  - remove the first and last character of the original nutrition and split the rest by ','.
  - use `apply` to get the value for each component and convert them to `float`.
  - add new columns: `calories`, `total_fat`, `sugar`, `sodium`, `protein`, `saturated_fat`, `carbohydrates` into `RAW_recipes`.
  - name a new dataframe after transforming the `RAW_recipes` called `recipes`.

4. Create `cheese` column to `recipes`
- For `ingredients`, we only care about whether cheese is presense in the long string in order to invest the relationship between calories and cheese:
  - use regular expression to find term `cheese` in `ingredients`, and create a new column `cheese` with `boolean` values indicating if there is cheese in ingredients.

5. Merge dataframes into a dataset called `combine`
- We left merge `RAW_interactions` to `recipes` on `id` in `recipes` and `recipe_id` in `RAW_interactions`.

6. Replace all ratings of 0.0 to `nan` in `RAW_interactions`
- Valid ratings only include 1, 2, 3, 4, 5, and ratings of 0.0 indicates that no rating is given in the interaction, which contains no actual meaning.

7. Compute average rating
- We calculate the average rating for each recipe as new column `avg_rating`, so it can better demonstrate the overall ratings of the recipe.
- Merge the `avg_rating` to the `combine`.

After cleaning, there are 234429 rows and 29 columns in `combine`. Here we display the first five rows of the dataframe with 10 columns that will be mainly used for further investigation. 

| name                                 |   calories |   total_fat |   sugar |   sodium |   protein |   saturated_fat |   carbohydrates | cheese   |   rating |
|:------------------------------------:|:----------:|:-----------:|:-------:|:--------:|:---------:|:---------------:|:---------------:|:--------:|:--------:|
| 1 brownies in the world    best ever |      138.4 |          10 |      50 |        3 |         3 |              19 |               6 | False    |        4 |
| 1 in canada chocolate chip cookies   |      595.1 |          46 |     211 |       22 |        13 |              51 |              26 | False    |        5 |
| 412 broccoli casserole               |      194.8 |          20 |       6 |       32 |        22 |              36 |               3 | True     |        5 |
| 412 broccoli casserole               |      194.8 |          20 |       6 |       32 |        22 |              36 |               3 | True     |        5 |
| 412 broccoli casserole               |      194.8 |          20 |       6 |       32 |        22 |              36 |               3 | True     |        5 |


### Univariate Analysis
First, we will examine the distribution of  calories from the `recipes` dataframe. 

<iframe
  src="assets/hist_calories_total.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

But as one can see from the plot, the distribution mostly concentrates on the very left. The outliers at the right tail makes the distribution hard to be examined. 

Then we remove the outliers to focus on values less and equal to 990. 

<iframe
  src="assets/hist_calories.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>


By removing the outliers in the histogram, we can more clearly observe that the distribution is skewed to the left, with the majority of data concentrated between 50 and 400 calories, and a peak at around 180. This is quite surprising, as the distribution suggests that more recipes promoting a healthy diet have been published. After the peak there is a decrease of the count of recipes, corresponding with higher calories.

### Bivariate Analysis
For bivariate analysis, we will examine the distribution of calories with and without cheese, also with outliers removed to better visualization. 

<iframe
  src="assets/fig_with_without_cheese.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

 We can see that both distributions skew to the left, but the peak of distribution of calories without cheese is on the left of the distribution with cheese. We will further investigate in later sections on whether having cheese in recipes tend to have higher calories. 

 [more analysis here??]

### Grouping and Aggregates
An interesting aggregate that we find is shown in the pivot table below. 

|   Year |   ('diff_prop', 0.0) |   ('diff_prop', 1.0) |   ('diff_prop', 2.0) |   ('diff_prop', 3.0) |   ('diff_prop', 4.0) |   ('diff_prop', 5.0) |
|-------:|---------------------:|---------------------:|---------------------:|---------------------:|---------------------:|---------------------:|
|   2008 |          -0.00417087 |           0.0373018  |          0.0314886   |           0.0339742  |           0.0296494  |           0.00595056 |
|   2009 |          -0.00117979 |           0.0121839  |         -0.00573415  |           0.0251196  |           0.00886583 |           0.0102082  |
|   2010 |          -0.00931341 |           0.00106069 |          0.0427559   |          -0.00798016 |          -0.00647318 |          -0.00948729 |
|   2011 |           0.00977973 |          -0.00940128 |          0.00721202  |           0.0137947  |          -0.00071431 |          -0.00714992 |
|   2012 |           0.023112   |          -0.00471617 |          0.016432    |          -0.00377306 |          -0.00190885 |           0.0127113  |
|   2013 |           0.022195   |           0.00508453 |          0.000573415 |          -0.014781   |          -0.00979702 |           0.00557635 |
|   2014 |           0.00269153 |           0.0326774  |          0.00872339  |          -0.0103835  |          -0.00499667 |           0.00335626 |
|   2015 |          -0.0104338  |          -0.0132254  |         -0.0242568   |          -0.0121251  |          -0.00482612 |          -0.00185076 |
|   2016 |          -0.0105039  |          -0.0162877  |         -0.0174133   |          -0.00873181 |          -0.00184123 |          -0.00384644 |
|   2017 |          -0.0131052  |          -0.0247924  |         -0.0407065   |          -0.00617981 |          -0.00443334 |          -0.00802292 |
|   2018 |          -0.00907127 |          -0.0198854  |         -0.0190744   |          -0.00893407 |          -0.00352448 |          -0.0074453  |

The pivot table is aimed to examine the `diff_prop` which is calculated by `prop_cheese_recipes - prop_nocheese_recipes` across years from 2008 to 2018, given the level of rating. Positive values indicate there are more recipes with cheese, and on the other hand negative values indicate more recipes without cheese. 

## Hypothesis Test

We are interested in whether the amount of calories in a recipe is higher when cheese is included as an ingredient. To test this hypothesis, we use a permutation test.

**Null Hypothesis**: the amount of calories in a recipe is not higher when it contains cheese.
**Alternative Hypothesis**: the amount of calories in a recipe is higher when it contains cheese.
**Test Statistic**: mean calories with cheese - mean calories without cheese
**Significance Level**: 0.05

The alternative hypothesis represents our assumption that recipes with cheese have higher calorie content. The null hypothesis is the reverse, stating there is no such relationship between the presence of cheese and the calorie content of a recipe. We use the difference in mean calories (mean calories with cheese - mean calories without cheese) as the test statistic because the mean effectively summarizes the central tendency of the data for each group. A large positive value of this statistic would indicate that recipes with cheese tend to have higher calorie content. We choose a significance level of 0.05 because it is a standard threshold for statistical significance.

In the permutation test, we randomly shuffle the calorie values and assign them to recipes (with or without cheese) for 1,000 iterations. The p-value obtained from this test is 0.0, meaning that none of the randomly generated differences in means is large or equal to the observed difference. This provides strong evidence against the null hypothesis. Therefore, we reject the null hypothesis and conclude that the amount of calories in a recipe is significantly higher when it contains cheese.

## Framing a Prediction

We plan to predict the amount of calories for each recipe using a linear regression model. Since the amount of calories can be any positive value, linear regression is a suitable choice for this continuous target variable.

The target variable for our model is the amount of calories in each recipe. We chose this as our prediction target because calories are an important factor when deciding whether to use a recipe for a meal. Recipes with too many calories may raise health concerns, while recipes with too few calories may not suffice as a complete meal. Additionally, calories are intuitively related to other nutritional components, such as fats, proteins, and carbohydrates. From our preliminary investigation, we also found that the presence of cheese in a recipe is correlated with calorie content. This suggests that we can predict calorie amounts based on the presence of cheese and other nutritional features.

We evaluated our model using Root Mean Squared Error (RMSE). RMSE measures the average magnitude of the prediction errors in the same units as the target variable (calories), making it easy to interpret. We chose RMSE over $R^2$ because RMSE provides a direct measure of prediction error, while $R^2$ only explains the proportion of variance in the target variable. Additionally, we chose RMSE over Mean Absolute Error (MAE) because RMSE penalizes large errors more heavily, which aligns with our goal of avoiding significant inaccuracies in calorie predictions.

Since the nutritions are calculated separately, it is reasonable to use other nutrition and ingredients to predict the amount of calories.

## Baseline Model

In our baseline model, we use linear regression to predict the amount of calories, a quantitative value, in recipes. We chose linear regression because it is simple, interpretable, and suitable for predicting continuous target variables like calories. The data is split into training and testing sets with an 80/20 ratio.

The model uses two quantitative features to predict:
1. total_fat
2. sugar

There are no ordinal or nominal features in this model, so no encoding or transformation is required. The features are used directly in the LinearRegression() function.

The RMSE of train set is 203.8 and RMSE of test set is 196.2. The similar performance on the training and testing sets indicates that there is no significant overfitting. However, the RMSE of approximately 200 calories is relatively high, meaning the model's predictions may not be precise enough for practical use. This is likely because the model uses only two features (total_fat and sugar) without further transformation and does not account for other factors that may influence calorie content. We should be able to improve the model performance by carefully choosing features and try some transformation.

## Final Model

We consider columns 'total_fat', 'sugar', 'sodium', 'protein', 'saturated_fat', 'carbohydrates' as features that may contribute to predicting because calories are derived from macronutrients (fats, proteins, and carbohydrates) and other nutritional components. By including these features, we capture the primary sources of calories in recipes, making them highly relevant for the prediction task. Besides them, we also include 'cheese' because it is a calorie-dense ingredient, and its presence in a recipe is likely to significantly impact the total calorie count. Including this feature allows the model to account for the additional calories contributed by cheese.

We use Linear regression to make the prediction. To optimize the model, we performed hyperparameter tuning using GridSearchCV and find that the best hyperparameter is True for fit_intercept and False for positive.

To improve the model's performance, we experimented with different combinations of features and transformations:\
We tried combinations of degree one and found 'total_fat', 'protein', 'carbohydrates' and 'cheese' performed best.\
We tried applying QuantileTransform to different combination of features, and not surprisingly it doesn't produce a better result.\
We wondered would larger degree give better result and tried adding different combinations of features of degree 2 to 'total_fat', 'protein', 'carbohydrates' and 'cheese'. [carbohydrates with degree 2, carbohydrates, total_fat, protein, cheese] achieved smallest RMSE.\
We further tried different combinations of features of degree 2 and 3 and add them to 'total_fat', 'protein', 'carbohydrates' and 'cheese' of degree 1. We didn't achieve better result than ['total_fat', 'protein', 'carbohydrates', 'cheese'].\
We didn't try degree 4 because as none in degree 3 outperforming orginal combination, it is unlikely we will find a better combination in degree 4. We can also see that when we try different d for PolynomialFeatures(d), 1 achieves smallest RMSE for test set.

So our final model is a linear regression using [carbohydrates with degree 2, carbohydrates, total_fat, protein, cheese], in which [carbohydrates, total_fat, protein] are quantitative values that don't need further transformation. cheese is booleans but since the program takes True as 1 and False as 0, it also don't need further transformation. carbohydrates with degree 2 is got by apply make_column_transformer to carbohydrates.

The final model performs better than baseline model. It achieves RMSE of 48.40 for train set and of 43.58 for test set.

## Fairness Analysis

For fairness analysis, we split the data into two groups: recipes with low sugar (less or equal to 23) and recipes with high sugar (more than 23). We pick 23 as the split point because it is the median of sugar in dataset. Since we uses linear regression model, we decide to rate our test using rooted mean squared error.\
To check the p value, we use permutation testing here. We shuffle whether each recipe has high or low sugar and calculate simulated result based on shuffled category.

**Null hypothesis**: Our model is fair. It predicts calories for recipes with low sugar and high sugar with similar RMSE.\
**Alternative hypothesis**: our model is biased. It predicts calories for recipes with high sugar with higher RMSE than recipes with low sugar.\
**Test statistic**: RMSE of recipes with high sugar-RMSE of recipes with low sugar\
**Significance level**: 0.05

<iframe
  src="assets/sugar_fair.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>

We also want to see if our prediction is fair for cheese. More precisely, if our prediction is fair for recipes with cheese and without cheese. Similarly, we use RMSE as evaluation metric and permutation test to simulate the difference in random state.

**Null hypothesis**: Our model is fair. It predicts calories for recipes with and without cheese with similar RMSE.\
**Alternative hypothesis**: our model is biased. It predicts calories for recipes without cheese with higher RMSE than recipes with cheese.\
**Test statistic**: RMSE of recipes with cheese - RMSE of recipes without cheese\
**Significance level**: 0.05

<iframe
  src="assets/cheese_fair.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>
Since we get a p value of 0.0, we reject null hypothesis. Our model is biased because it predicts calories for recipes with cheese better comparing to recipes without cheese.