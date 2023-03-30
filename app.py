import pickle
import streamlit as st
import numpy as np

# Launching the header of the locallhost
st.header("Book Recommender System Using Association Rule Mining")

# importing asll the pickle model files
rules = pickle.load(open('rules.pkl', 'rb'))
book_name = pickle.load(open('unique_books.pkl', 'rb'))
final_rating = pickle.load(open('final_rating.pkl', 'rb'))
basket_sets = pickle.load(open('basket_sets.pkl', 'rb'))

def fetch_poster(my_recommendations):
    books_name = []
    poster_url = []
    ids_index = []

    for book in my_recommendations:
        books_name.append(book)

    for name in books_name:
        ids = np.where(final_rating['Title'] == name)[0][0]
        ids_index.append(ids)

    for ids in ids_index:
        url = final_rating.iloc[ids]['image-url']
        poster_url.append(url)

    return poster_url

def recommend_book(book_name):
    # Create an empty set to store the recommendations
    my_recommendations = set([book_name])

    # Loop over each rule and check if it contains the chosen book in the antecedents
    for _, row in rules.iterrows():
        if book_name in row['antecedents']:
            # If the book is in the antecedents, add the consequents to the recommendations set
            my_recommendations.update(row['consequents'])

    poster_url = fetch_poster(my_recommendations)

    my_recommendations = list(my_recommendations)
    return my_recommendations, poster_url


selected_books = st.selectbox(
    "Type or Select a book",
    book_name
)
rules.plot.scatter(x='lift', y='confidence')
if st.button("Show Recommendation"):
    
    books, poster_url = recommend_book(selected_books)

    num_books = min(len(books), 5)
    cols = st.columns(num_books)
    
    for i in range(num_books):
        with cols[i]:
            st.text(books[i])
            st.image(poster_url[i])

