import streamlit as st
import pickle
import numpy as np
import sqlite3
import seaborn as sns
import requests

# Create a connection to the SQLite database
conn = sqlite3.connect("users.db")
c = conn.cursor()

# Create the users table if it does not exist
c.execute("CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, password TEXT)")

# Create a function to add a new user to the database
def add_user(username, password):
    c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
    conn.commit()

# Create a function to authenticate the user
def authenticate(username, password):
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    if c.fetchone() is not None:
        return True
    else:
        return False

# Create a sidebar with login form
with st.sidebar:
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if authenticate(username, password):
            st.success("Logged in!")
            # Redirect the user to the main app
            # Here, you can use Streamlit's built-in `st.experimental_set_query_params()` function to pass the user's login information to the main app
        else:
            st.error("Invalid username or password.")
        
    st.title("Register")
    new_username = st.text_input("New username")
    new_password = st.text_input("New password", type="password")
    if st.button("Register"):
        add_user(new_username, new_password)
        st.success("Registered successfully!")


# Redirect the user to the main app after successful login
if authenticate(username, password):
    query_params = {"username": username}
    #st.success("Logged in!")
    # Redirect the user to the main app
    # Launching the header of the locallhost
    st.header("Book Recommender System Using Association Rule Mining")
    model = pickle.load(open('artifacts/model.pkl', 'rb'))
    book_name = pickle.load(open('artifacts/book_name.pkl', 'rb'))
    final_rating = pickle.load(open('artifacts/final_rating.pkl', 'rb'))
    book_pivot = pickle.load(open('artifacts/book_pivot.pkl', 'rb'))

    rules = pickle.load(open('artifacts/rules.pkl', 'rb'))


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

    def recommend_books(book_name):
        # Get the association rule recommendations
        my_recommendations = set([book_name])

        for _, row in rules.iterrows():
            if book_name in row['antecedents']:
                my_recommendations.update(row['consequents'])

        # Get the collaborative filtering recommendations
        book_id = np.where(book_pivot.index == book_name)[0][0]
        distance, suggestion = model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1), n_neighbors=6)
        cf_recommendations = []
        for i in range(len(suggestion)):
            books = book_pivot.index[suggestion[i]]
            for j in books:
                cf_recommendations.append(j)
        cf_recommendations = list(set(cf_recommendations))

        # Combine the two recommendation sets
        all_recommendations = list(my_recommendations.union(cf_recommendations))

        #book_links = fetch_links(all_recommendations, api_key)
        poster_url = fetch_poster(all_recommendations)

        return all_recommendations, poster_url#, book_links

    
    def fetch_links(all_recommendations, api_key):
        book_links = []
        
        for book in all_recommendations:
            response = requests.get(f"https://www.googleapis.com/books/v1/volumes?q={book}&key={api_key}")
            data = response.json()
            # Check if any items were returned
            if "items" in data:
                link = data["items"][0]["volumeInfo"]["previewLink"]
                book_links.append(link)
            else:
                book_links.append("Not found")
        return book_links

    selected_books = st.selectbox(
        "Type or select a book",
        book_name
    )

    if st.button("Show Recommendation"):
        
        books, poster_url = recommend_books(selected_books, api_key)
        book_links = fetch_links(books, api_key)

        num_books = min(len(books), 5)
        cols = st.columns(num_books)
        
        for i in range(num_books):
            with cols[i]:
                st.text(books[i])
                st.image(poster_url[i])
                st.write(f"[Preview]({book_links[i]})")

    sns.set_style('white')
    sns.set_style('ticks')

    # Create the plot
    reg_plot = sns.regplot(x='lift', y='confidence', data=rules)

    # Display the plot in Streamlit
    st.pyplot(reg_plot.figure)
    
else:
    st.header("Welcome, Login To Enjoy Book Recommender System!")
