# Investigation on the Relationship Between the Existence of Cheese in Ingredients and Calories

**Name(s)**: Huiting Chen, Coco Sun

**Website Link**: [website](https://szhcoco.github.io/Investigation-on-Cheese-and-Calories/)

## Overview
This is a data science project, conducted by UCSD, aiming to investigate on the relationship between the existence of cheese in ingredients and calories

## Instruction
Cheese is a pupolar ingredient for food. But it also raise concerns about healthy problem because people tend to take cheese as having high calories. Does recipes with cheese really have high calories comparing to recipes without cheese? To answer the question, we investigate on recipes.

To explore this question, we analyze two datasets from a [recipe recommendation website](https://www.food.com/), containing recipes and their ratings since 2008:
 - Recipe dataset: 83,782 rows and 12 columns, containing information about each recipe
 
 <table border="1">
    <thead>
    <tr style="text-align: right;">
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
 - Ratings dataset: 731,927 rows and 5 columns, recording users' ratings for different recipes.

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

To address the problem, we need to focus on column 'nutrition' and 'ingredient'. More accurately, we will focus on column 'calories' and 'cheese' we got from 'nutrition' and 'ingredient'.

This insight could help home cooks find more appealing recipes, assist recipe creators in tailoring their content, and guide food platforms in improving recommendations.

## Data Cleaning and Exploratory Data Analysis
1. Download the datasets.
2. Remove column 'Unnamed: 0' from recipes\
Column doesn't contain any useful information, so we drop the column.
3. Split column 'nutrition' in recipes into columns 'calories', 'total_fat', 'sugar', 'sodium', 'protein', 'saturated_fat', 'carbohydrates'.\
Since column 'nutrition' in recipes contains list-like strings storing [calories (#), total fat (PDV), sugar (PDV), sodium (PDV), protein (PDV), saturated fat (PDV), carbohydrates (PDV)], we split the column and store informations as floats in separate columns for each kind of nutrition so we can access them easier in the future.\
To do so, we remove the first and last character of the original nutrition and split the rest by ','. Then we use apply to get the value for each component and convert them to float at last.
4. Add column 'cheese'\
Since we are going to invest the relationship between calories and cheese, we need to decide if ingredient for each recipe contains cheese. To do so, we add a new column 'cheese' contains boolean values indicating if there is cheese in ingredient.\
We use re libarary to check if there is strings of form r' cheese |\'cheese |\'cheese\'| cheese\'' in ingredient.
5. Merge dataframes\
We left interactions to recipes on recipe id.
6. Change rating of 0.0 to np.nan\
We change rating of 0.0 in combined dataset to np.nan because valid ratings only include 1, 2, 3, 4, 5. A rating of 0.0 actually indicats that no rating is given in the interaction, so we change it to np.nan
7. Compute average rating\
We calculate the average rating for each recipe as new column 'avg_rating' so it can better demonstrate the overall ratings of the recipe.
### Univariate Analysis
We examine the distribution of calories for calories smaller than 1000 for this analysis in recipes. From the plot we can see that the distribution skew to the right, indicating that most proportion of recipes have a low calories and center around 175. After that there is a decrease of the proportion of recipes and the amount of calories increase.
<iframe
  src="assets/hist_calories_total.html"
  width="800"
  height="600"
  frameborder="0"
></iframe>