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
 - `RAW_recipes` dataset: 83,782 rows and 10 columns, containing information about each recipe. 

 | Column           | Description                                                                                                                                                                                       |
|:----------------:|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
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
|:-----------:|:--------------------|
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
- For `nutrition`, each row is a list-like srings storing values in the form `[calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)]`. Since all values are important for later investigation, we want to get access to them easier, so we: 
  - remove the first and last character of the original nutrition and split the rest by ','.
  - use `apply` to get the value for each component and convert them to `float`.
  - add new columns: `calories`, `total_fat`, `sugar`, `sodium`, `protein`, `saturated_fat`, `carbohydrates` into `RAW_recipes`.
  - name a new dataframe after transforming the `RAW_recipes` called `recipes`.

4. Create `cheese` column to `recipes`
- For `ingredients`, we only care about whether cheese is presense in the long string in order to invest the relationship between calories and cheese:
  - use regular expression to find term `cheese` in `ingredients`, and create a new column `cheese` with `boolean` values indicating if there is cheese in ingredients.

5. Merge dataframes into a dataset called `combine`
- We left merge `RAW_interactions` to `recipes` on `id` in `recipes` and `recipe_id` in `RAW_interactions`.

6. Replace all ratings of 0.0 to `nan` in `combine`
- Valid ratings only include 1, 2, 3, 4, 5, and ratings of 0.0 indicates that no rating is given in the interaction, which contains no actual meaning.

7. Compute average rating
- We calculate the average rating for each recipe as new column `avg_rating`, so it can better demonstrate the overall ratings of the recipe.
- Merge the `avg_rating` to the `combine`.

After cleaning, there are 234429 rows and 26 columns in `combine`. Here we display the first five rows of the dataframe with 10 columns that will be mainly used for further investigation. 

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
  height="450"
  frameborder="0"
></iframe>

But as one can see from the plot, the distribution mostly concentrates on the very left. The outliers at the right tail makes the distribution hard to be examined. 

Then we remove the outliers to focus on values less and equal to 990. 

<iframe
  src="assets/hist_calories.html"
  width="800"
  height="450"
  frameborder="0"
></iframe>


By removing the outliers in the histogram, we can more clearly observe that the distribution is skewed to the left, with the majority of data concentrated between 50 and 400 calories, and a peak at around 180. This is quite surprising, as the distribution suggests that more recipes promoting a healthy diet have been published. After the peak there is a decrease of the count of recipes, corresponding with higher calories.

### Bivariate Analysis
For bivariate analysis, we will examine the distribution of calories with and without cheese, also with outliers removed to better visualization. 

<iframe
  src="assets/fig_with_without_cheese.html"
  width="900"
  height="500"
  frameborder="0"
></iframe>

 We can see that both distributions skew to the left, but the peak of distribution of calories without cheese is on the left of the distribution with cheese, indicating that calories of recipes with cheese may be higher than calories of those without cheese. We will further investigate in later sections on whether having cheese in recipes tend to have higher calories.

<iframe
  src="assets/fig_scatter_subplots.html"
  width="900"
  height="500"
  frameborder="0"
></iframe> 

Then we want to examine the relationship between calories and nutritions. The above scatter plot is consisted of six subplots, each showing the relationship betwee calories and one nutrition `[total_fat, sugar, sodium, protein, saturated_fat, carbonhydrates]`. Except for the scatter plot with sodium, all the other five display a positive correlated relationship. It implies the role that the nutritions might play in prediction calories. 

### Grouping and Aggregates
An interesting aggregate that we find is shown in the pivot table below. 

|   Year |   ('diff_prop', 1.0) |   ('diff_prop', 2.0) |   ('diff_prop', 3.0) |   ('diff_prop', 4.0) |   ('diff_prop', 5.0) |
|:------:|:--------------------:|:--------------------:|:--------------------:|:--------------------:|:--------------------:|
|   2008 |                 0.04 |                 0.03 |                 0.03 |                 0.03 |                 0.01 |
|   2009 |                 0.01 |                -0.01 |                 0.03 |                 0.01 |                 0.01 |
|   2010 |                 0    |                 0.04 |                -0.01 |                -0.01 |                -0.01 |
|   2011 |                -0.01 |                 0.01 |                 0.01 |                -0    |                -0.01 |
|   2012 |                -0    |                 0.02 |                -0    |                -0    |                 0.01 |
|   2013 |                 0.01 |                 0    |                -0.01 |                -0.01 |                 0.01 |
|   2014 |                 0.03 |                 0.01 |                -0.01 |                -0    |                 0    |
|   2015 |                -0.01 |                -0.02 |                -0.01 |                -0    |                -0    |
|   2016 |                -0.02 |                -0.02 |                -0.01 |                -0    |                -0    |
|   2017 |                -0.02 |                -0.04 |                -0.01 |                -0    |                -0.01 |
|   2018 |                -0.02 |                -0.02 |                -0.01 |                -0    |                -0.01 |

<br>
The pivot table is aimed to examine the `diff_prop` which is calculated by `prop_cheese_recipes - prop_nocheese_recipes` across years from 2008 to 2018, given the level of rating. Positive values indicate there are more recipes with cheese, and on the other hand negative values indicate more recipes without cheese. 


## Assessment of Missingness

### NMAR Analysis
We believe the missingness in the `review column` is NMAR. There are a few possible reasons for people not to fill in the review section. For example, people tend not to post reviews if they feel neutural or nothing special about the recipes, and thus choose not to take extra steps wrting the reviews and publish on the site. Also, it is less likely for people to write reviews if they have negative experience with the recipes, either because they want to avoid conflicts against those who like the food, or it is worth wasting their time doing so. On the other hand, people who like the recipes or have anything interesting to share tend to leave comments. Thus the missingess depends on the values of the `review` column, making it NMAR. 

We could obtain additional data to to make the missingness MAR. One possible approach is to send out polls through personal emails provided during registration process, or communicate through messages on the Food website, to ask if they could share their experience with the recipes and reason why they don't want to write reviews. 


### Missingness Dependency

In this part, we will examine the missingness dependency between `rating` and the other two columns, using permutation tests. 

The first two columns that we want to focus on are `calories` and `rating`, and we want to determine whether the missingness in `rating` is dependent on distribution of `calories`. 

**Null Hypothesis**: the distribution of  `calories` is the same when `rating` is missing and not missing.


**Alternative Hypothesis**: the distribution of `calories` is not the same when `rating` is missing and not missing.

Before performing the permutation test, we want to visualize the distribution of calories for recipes missing and not missing.


<iframe
  src="assets/rating_calories_total_2.html"
  width="1000"
  height="500"
  frameborder="0"
></iframe>

But because there are a lot of outliers in `calories`, the values are compressed to the left and make the distributions hard to see. So we remove the outliers and generate another plot as shown below. 


<iframe
  src="assets/rating_calories_no_outleirs.html"
  width="1000"
  height="500"
  frameborder="0"
></iframe>

The shape of two distributions look similar, so we will use **permutation test** with **absolute value of difference in means** as our test statistics, and the rejection threshold is **0.05**.

In the permutation test, we shuffle the `calories` column and calculate the test statstics for 1000 times. Then we plot the distribution of the calculated statistics and the observed one. 

<iframe
  src="assets/permu_test_1.html"
  width="900"
  height="500"
  frameborder="0"
></iframe>

Since the p value is less than 0.05 and as small as 0, we have strong evidence to reject the null hypothesis and say that the distribution for `calories` is different for missing and not missing `rating`. Thus missingness in `rating` is MAR, dependent on `calories`.


<br>
Then we want to see if the missingness in rating is dependent on the time spent on cooking, or `minutes` in the `combine` dataframe.

**Null hypothesis**: the distribution of `minutes` is the same when `rating` is missing and not missing.

**Alternative hypothesis**: the distribution of `minutes` is not the same when `rating` is missing and not missing.

We visualize the two distributions to determine type of test to perform. We also exclude outliers in `minutes` to make it clearer to see.  

<iframe
  src="assets/rating_minutes_no_outliers.html"
  width="900"
  height="500"
  frameborder="0"
></iframe>

From the histogram we notice that two distributions have similar shape. Thus we would use **permutation test** with **absolute values in means** as test statistics, or the mean of `minutes` when `rating` is missing and not missing. The significance level is **0.05**. 

<iframe
  src="assets/permu_test_2.html"
  width="900"
  height="450"
  frameborder="0"
></iframe>

The p_value from the permutation test is 0.14, which is greater than the threshold of 0.05. We fail to reject the null hypothesis to say that the two distributions of `minutes` are different when rating is missing and not missing. Thus the missingness of `rating` is not dependent on `minutes`. 




## Hypothesis Testing

Recalling the goal of our project, we will investigate whether the amount of calories in a recipe is larger when there is cheese in the ingredients. The columns we will use are `calories` and `cheese`. We decided to test our hypothesis using a permutation test.

**Null Hypothesis**: the amount of calories for a recipe with cheese is the same as the amount of calories without cheese.

**Alternative Hypothesis**: the amount of calories for a recipe with cheese is the greater than the amount of calories without cheese.

**Test Statistic**: mean calories with cheese - mean calories without cheese.

**Significance Level**: 0.05

For test statistics, a large value in (mean calories with cheese - mean calories without cheese) will imply that more calories in a recipe  with cheese than recipes without cheese. 

In our permutation test we randomly shuffle calories and assign them to recipes for 1000 times to generate an empirical distribution of the test statistics. We find p value is 0.0, which means none of our randomly generated difference is larger than the observed statistic. Therefore, we have strong evidence to reject the null hypothesis and suggest that recipes with cheese would have higher calories. 

<iframe
  src="assets/hypothesis.html"
  width="800"
  height="450"
  frameborder="0"
></iframe>

## Framing a Prediction

We plan to predict the amount of calories for each recipe using a linear regression model as `calories` are continous variable. It is hard to predict the amount of calories as a quantitative value by classification.

The reason we choose to use `calories `as our prediction target is that it is an important factor in deciding whether to use a recipe for a meal. Recipes with too many calories may raise health concerns, while recipes with too few calories may not be able to maintain our daily activities. From our earlier investigation, we  found that the presence of cheese in a recipe correlates with calory values. As an ingredient rich in nutrients like calcium, protein, and fat, we wonder what the key nutrients play a significant role in determining calories. Thus, we want to predict calorie amounts based on the presence of cheese and other nutritional features.

We evaluated our model using Root Mean Squared Error (`RMSE`). It is a more direct value to assess the errors in the test set, making it easy to interpret. Also, we can compare `RMSE` of testing and training set to examine if the model is overfitting, resulting in a dataframe that can generalize better. It is very important for our model because the values of our features, the nutritions, can vary a lot in real world. Compared to `RMSE`, R<sup>2</sup> only tells us how the linear model fit the existed dataset, with less power to generalize.   

At the point of prediction, we information about values of different nutritions used in recipes from the `nutritions` column, and we also know the prescence of from the `cheese` column created from `ingredients`. We will use them as features in our regression model. 

## Baseline Model

In our baseline model, we want to use linear regression to predict the amount of calories, a quantitative value. We use features in `recipes` dataframe, including `total_fat` and `sugar`. All features here (`calories`, `total_fat`, `sugar`) are quantitative and continuous, so we don’t need to do any transformation here and can use them directly in the `LinearRegression()` function. The data is split into training and testing sets with ratios of 0.8 and 0.2.

The `RMSE` of the train set is 203.8 and the `RMSE` of the test set is 196.2. The similar performance on the training and testing sets indicates that there is no significant overfitting. However, the `RMSE` of approximately 200 calories is relatively high, so we still need to improve our model. Besides, the model uses only two features (`total_fat` and `sugar`), all directly put into the `LinearRegression()` function. So we should be able to improve the model performance by carefully choosing features and trying some transformation.

## Final Model

We consider all columns of nutritions (`total_fat`, `sugar`, `sodium`, `protein`, `saturated_fat`, `carbohydrates`) as features that may contribute to improvement of our prediction model, because calories scientifically are derived from macronutrients (fats, proteins, and carbohydrates) and other nutritional components. By including these features, we are able to capture more primary sources of calories in recipes, making features more relevant for our target value. In addition, we also include `cheese` because it is a calorie-dense ingredient, and its presence in a recipe is likely to impact the total calorie count, supported by evidence from our hypothesis test. Including this feature allows the model to account for the additional calories contributed by cheese.

We use Linear regression to make the prediction. To optimize the model, we performed hyperparameter tuning using `GridSearchCV`. The only two hyperparameters in `LinearRegression()` are `fit_intercept` and `positive`, and we find that the best hyperparameter is `True` for `fit_intercept` and `False` for `positive`. 

To improve the model's performance, we experimented with different combinations of features and transformations:
- We tried combinations of degree one of features, and found `total_fat`, `protein`, `carbohydrates` and `cheese` performed best.
- We tried applying `QuantileTransform` to different combination of features, and it largely increases our `RMSE`, from 196 for the baseline model to 486.72 on test set. Thus `QuantileTransform`is not a good transformer for our model. 

Then we wondered if higher degrees in our regression model would give a better result, so we tried adding different combinations of features of degree 2 to `total_fat`, `protein`, `carbohydrates` and `cheese`. The result we got is the list of features `curr_best_features = [carbohydrates^2, carbohydrates, total_fat, protein, cheese]` that achieved the smallest `RMSE`.

We further tried different combinations of features of degree 2 and 3 and add them to `total_fat`, `protein`, `carbohydrates` and `cheese` of degree
- We failed to achieve better result than `curr_best_features`.
- We stopped at degree 3 because as none in degree 3 outperforming orginal combination, it is unlikely we will find a better combination in degree 4. We can also see that when we try different `d` (in `range(4)`) for `PolynomialFeatures(d)`, degree 1 achieves smallest RMSE for test set.

So our final model is a linear regression using `curr_best_features = [carbohydrates^2, carbohydrates, total_fat, protein, cheese]`, in which `carbohydrates`, `total_fat`, `protein` are quantitative values that don't need further transformation. `cheese` is booleans but since the program takes `True` as 1 and `False` as 0, it also don't need further transformation. `carbohydrates` with degree 2 is achieved by applying `make_column_transformer`.

The final model performs better than baseline model. It achieves `RMSE` of 48.40 for train set and of 43.58 for test set.

## Fairness Analysis

For fairness analysis, we split the data into two groups: recipes with low sugar (less or equal to 23) and recipes with high sugar (more than 23). We pick 23 as the split point because it is the median of sugar in the `recipes` dataset. Since we use a linear regression model, we decide to use `RMSE` as our evaluation metric. It is easier to understand compared to R<sup>2</sup> and it is better to keep a consistency with the elavuation metric we used during model prediction.

To check the p value, we use permutation testing here. For each recipe we list their actual calories and predicted calories. Then we can calculate squared error for each recipe. By calculating the square root of mean value for each group (low sugar and high sugar), we first created a column called `X_test_high` using `Binarizer` transformer to determine if the value of sugar in the test set is above 23. Then for a 1000 times, we calculated two `RMSE` values and so their difference. We shuffled `X_test_high` and re-calculate the `RMSE` and their differences, storing in a `stats` list. 

**Null hypothesis**: Our model is fair. It predicts calories for recipes with low sugar and high sugar with similar RMSE.

**Alternative hypothesis**: our model is biased. It predicts calories for recipes with high sugar with higher RMSE than recipes with low sugar.

**Test statistic**: RMSE of recipes with high sugar-RMSE of recipes with low sugar

**Significance level**: 0.05

<iframe
  src="assets/sugar_fair.html"
  width="800"
  height="450"
  frameborder="0"
></iframe>

As a result, we get a p value of 0.201, which is larger than the significance level. We don't have enough evidence to reject the null hypothesis. It implies that our model results in similar performance in predicting calories when the amount of sugar is low and high. 


We also want to see if our model is fair in predicting cheese. More specifically, if our prediction is fair for recipes with cheese and without cheese. Similarly, we use `RMSE` as an evaluation metric and permutation test to simulate the difference in random state.\
We run a permutation test in a similar way as we did in fairness analysis for sugar. The only difference is that instead of grouping recipes based on sugar, we group them based on the existence of cheese.


**Null hypothesis**: Our model is fair. It predicts calories for recipes with and without cheese with similar RMSE.

**Alternative hypothesis**: our model is biased. It predicts calories for recipes without cheese with higher RMSE than recipes with cheese.

**Test statistic**: RMSE of recipes with cheese - RMSE of recipes without cheese

**Significance level**: 0.05

<iframe
  src="assets/cheese_fair.html"
  width="800"
  height="500"
  frameborder="0"
></iframe>

As a result, we get a p value of 0.0, which is much lower than the significance level, so we reject the null hypothesis. The result implies that our model is unfair. It predicts calories for recipes without cheese with higher RMSE than recipes with cheese. 
