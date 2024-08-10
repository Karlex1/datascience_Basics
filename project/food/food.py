import pandas as pd 
import numpy as np 
import streamlit as st
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import seaborn as sns

# Load your datasets
food = pd.read_csv('./1662574418893344.csv')
ratings = pd.read_csv('./ratings.csv')

# Display the first few rows of the food data


# Create the ratings matrix (Food_ID x User_ID)
dataset = ratings.pivot_table(index='Food_ID', columns='User_ID', values='Rating')
dataset.fillna(0, inplace=True)


# Prepare the NearestNeighbors model
csr_dataset = csr_matrix(dataset.values)
model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
model.fit(csr_dataset)

def food_rec(Food_Name):
    n = 5
    Food_List = food[food['Name'].str.contains(Food_Name, case=False)]  # Case-insensitive search
    
    if not Food_List.empty:
        Foodid = Food_List.iloc[0]['Food_ID']
        
        # Get the index of the Foodid in the dataset's index
        try:
            food_index = dataset.index.get_loc(Foodid)
        except KeyError:
            return ["NO SIMILAR FOOD"]
        
        distances, indices = model.kneighbors(csr_dataset[food_index], n_neighbors=n+1)
        Food_indices = sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])
        
        recom = []
        for val in Food_indices[1:]:  # Skip the first one since it's the query food itself
            Foodid = dataset.index[val[0]]
            i = food[food['Food_ID'] == Foodid].index[0]
            recom.append({'Name': food.iloc[i]['Name'], 'Distance': val[1]})
        
        df = pd.DataFrame(recom)
        return df['Name'].tolist()  # Return as a list of names instead of a Series
    else:
        return ["NO SIMILAR FOOD"]

# Streamlit app interface
st.title('Food Recommendation System')

# Cuisine selection
cuisines = food['C_Type'].unique().tolist()
selected_cuisine = st.selectbox('Select a cuisine:', cuisines)

# Food name dropdown filtered by selected cuisine
filtered_food = food[food['C_Type'] == selected_cuisine]
food_name_list = filtered_food['Name'].tolist()
food_name_input = st.selectbox('Select a food name:', food_name_list)

if food_name_input:
    # Display the recommendations
    recommendations = food_rec(food_name_input)
    
    if "NO SIMILAR FOOD" in recommendations:
        st.write("No similar food found.")
    else:
        st.write("Here are some foods you might like:")
        for food_name in recommendations:
            st.write(food_name)