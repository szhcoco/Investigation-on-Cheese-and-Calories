# Investigation on the Relationship Between the Existence of Cheese in Ingredients and Calories

**Name(s)**: Huiting Chen, Coco Sun

**Website Link**: (your website link)

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

To address the problem, we need to focus on column 'nutrition' and 'ingredient'. More accurately, we will focus on column 'calories' and 'cheese' we got from 'nutrition' and 'ingredient'.

This insight could help home cooks find more appealing recipes, assist recipe creators in tailoring their content, and guide food platforms in improving recommendations.
