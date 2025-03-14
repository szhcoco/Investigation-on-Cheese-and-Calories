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

 <table border="1">
    <thead>
    <tr style="text-align: middle;">
      <th>Column</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>'name'</td>
      <td>Recipe name</td>
    </tr>
    <tr>
      <td>'id'</td>
      <td>Recipe ID</td>
    </tr>
    <tr>
      <td>'minutes'</td>
      <td>Minutes to prepare recipe</td>
    </tr>
    <tr>
      <td>'contributor_id'</td>
      <td>User ID who submitted this recipe</td>
    </tr>
    <tr>
      <td>'submitted'</td>
      <td>Date recipe was submitted</td>
    </tr>
    <tr>
        <td>'tags'</td>
        <td>Food.com tags for recipe</td>
    </tr>
    <tr>
      <td>'nutrition'</td>
      <td>Nutrition information in the form [calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)]; PDV stands for “percentage of daily value”</td>
    </tr>
    <tr>
      <td>'n_steps'</td>
      <td>DNumber of steps in recipe</td>
    </tr>
    <tr>
      <td>'steps'</td>
      <td>Text for recipe steps, in order</td>
    </tr>
    <tr>
      <td>''description'</td>
      <td>User-provided description</td>
    </tr>
  </tbody>
 </table>

 - `RAW_interactions` dataset: 731,927 rows and 5 columns, recording users' ratings and comments for different recipes.

 <table border="1">
    <thead>
    <tr style="text-align: right;">
      <th>Column</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>'user_id'</td>
      <td>User ID</td>
    </tr>
    <tr>
      <td>'recipe_id'</td>
      <td>Recipe ID</td>
    </tr>
    <tr>
      <td>'date'</td>
      <td>Date of interaction</td>
    </tr>
    <tr>
      <td>'rating'</td>
      <td>Rating given</td>
    </tr>
    <tr>
      <td>'review'</td>
      <td>Review text</td>
    </tr>
  </tbody>
 </table>

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