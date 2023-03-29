#!/usr/bin/env python
# coding: utf-8

# In[1]:
import pandas as pd
from surprise.model_selection import train_test_split
from mlxtend.frequent_patterns import apriori, association_rules
import numpy as np
import ast
import pickle


# In[2]:
## Loading data
books = pd.read_csv('data/books.csv', sep = ";", 
                    error_bad_lines=False, encoding='latin-1')


# In[4]:
## reduciing the column to the relevant
books = books[['ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication',
             'Publisher','Image-URL-L']]


# In[5]:
## Renaming the columns
books.rename(columns={
    "Book-Title": "Title",
    "Book-Author": "Author",
    "Year-Of-Publication": "Year",
    "Image-URL-L": "image-url"}, inplace = True)


# In[6]:
## users data and rating analysis
users = pd.read_csv("data/users.csv", sep = ";", 
                    error_bad_lines = False, encoding = 'latin-1')


# In[8]:
ratings = pd.read_csv("data/ratings.csv", sep = ";", 
                      error_bad_lines = False, encoding = 'latin-1')


# In[9]:
## renaming
ratings.rename(columns={
"User-ID": "User_id",
"Book-Rating": "Rating"}, inplace = True)


# In[10]:
users.rename(columns={
    "User-ID": "User_id"}, inplace = True)


# In[11]:
x = ratings['User_id'].value_counts()>200
y = x[x].index
ratings = ratings[ratings["User_id"].isin(y)]


# In[12]:
rating_with_books = ratings.merge(books, on = "ISBN")
num_rating = rating_with_books.groupby("Title")["Rating"].count().reset_index()


# In[13]:
num_rating.rename(columns={"Rating":"No_of_rating"}, inplace=True)

    ## merging
final_rating = rating_with_books.merge(num_rating, on = "Title")
    ## cleaning
final_rating = final_rating[final_rating["No_of_rating"] >=50]
final_rating.drop_duplicates(["User_id", "Title"], inplace=True)
final_rating


# In[14]:
pivot = final_rating.pivot_table(index='User_id', columns='Title', values='Rating')


# In[15]:
basket = pivot.notnull().astype('int')


# In[16]:
def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
#apply the function to data using applymap
basket_sets = basket.applymap(encode_units)
basket_sets.shape


# In[17]:
# Split your data into training and testing sets
train_data = basket_sets[:800]
test_data = basket_sets[800:]


# In[18]:
#Generating itemsets using apriori
frequent_itemsets = apriori(basket_sets, min_support = 0.07, use_colnames = True)


# In[19]:
rules = association_rules(frequent_itemsets, metric = 'lift', min_threshold = 1)


# In[20]:


rules[ (rules['lift'] >= 6) &
      (rules['confidence'] >= 0.8)]


# In[22]:
book_name = basket_sets.columns


# In[23]:
#lists for string the antecedents and consequents
rules_list = []
for i in rules['antecedents']:
    string_set = str(i)
    list_id = '[' + string_set[11:-2] + ']'
    list_id = ast.literal_eval(list_id)
    rules_list.append(list_id)
    
book_strings = [book[0] for book in rules_list]
unique_books = (set(book_strings))


# In[29]:


# Define the book of your choosing
my_book = 'Congo'

# Create an empty set to store the recommendations
my_recommendations = set([my_book])

# Loop over each rule and check if it contains the chosen book in the antecedents
for _, row in rules.iterrows():
    if my_book in row['antecedents']:
        # If the book is in the antecedents, add the consequents to the recommendations set
        my_recommendations.update(row['consequents'])

# Print the set of recommendations for the chosen book
print(f"Recommendations for '{my_book}':")
for book in my_recommendations:
    print(book)


# In[30]:


import seaborn as sns
sns.set_style('white')
sns.set_style('ticks')

sns.regplot(x='lift', y='confidence', data=rules)


# In[31]:


sns.lmplot(x='lift', y='confidence', hue='support', data=rules)


# In[33]:


# Find users who bought "The Firm" in the testing set
test_users = set(final_rating[final_rating['Title'] == my_book]['User_id'])
# Evaluate the accuracy of the recommendations
total_recommendations = 0
correct_recommendations = 0
for user in test_users:
    user_bought_items = set(final_rating[final_rating['User_id'] == user]['Title'])
    recommendations = set()
    for _, row in rules.iterrows():
        if 'The Firm' in row['antecedents'] and set(row['antecedents']).issubset(user_bought_items):
            recommendations.update(row['consequents'])
    total_recommendations += len(recommendations)
    correct_recommendations += len(recommendations.intersection(user_bought_items))
accuracy = correct_recommendations / total_recommendations
print(f"Accuracy: {accuracy}")


# In[34]:


pickle.dump(rules, open('artifacts/rules.pkl', 'wb'))
pickle.dump(unique_books, open('artifacts/unique_books.pkl', 'wb'))
pickle.dump(final_rating, open('artifacts/final_rating.pkl', 'wb'))
pickle.dump(basket_sets, open('artifacts/basket_sets.pkl', 'wb'))


# In[ ]:




