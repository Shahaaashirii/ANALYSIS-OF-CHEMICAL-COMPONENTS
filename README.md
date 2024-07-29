# ANALYSIS-OF-CHEMICAL-COMPONENTS
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from bokeh.plotting import figure, show
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.io import output_notebook

# Enable output in Jupyter Notebook
output_notebook()

# Step 1: Load the Dataset
df = pd.read_csv(r"C:\Users\shaha\cosmetics.csv")

# Step 2: Filter the Dataset for a Specific Category and Skin Type
moisturizers_dry = df[(df['Label'] == 'Moisturizer') & (df['Dry'] == 1)]

# Step 3: Tokenize the Ingredients
def tokenize(ingredients):
    return ingredients.split(", ")

moisturizers_dry.loc[:, 'Tokens'] = moisturizers_dry['Ingredients'].apply(tokenize)

# Step 4: Initialize a Document-Term Matrix (DTM)
M = moisturizers_dry.shape[0]
corpus = moisturizers_dry['Tokens']
all_ingredients = [ingredient for sublist in moisturizers_dry['Tokens'] for ingredient in sublist]
unique_ingredients = list(set(all_ingredients))
N = len(unique_ingredients)
ingredient_idx = {ingredient: i for i, ingredient in enumerate(unique_ingredients)}

# Initialize the matrix
A = np.zeros((M, N))

# Step 5: Create a Counter Function
def oh_encoder(tokens):
    x = np.zeros(N)
    indices = [ingredient_idx[token] for token in tokens if token in ingredient_idx]
    x[indices] = 1
    return x

# Step 6: Get the Binary Value of the Tokens for Each Row of the Matrix
for i, tokens in enumerate(corpus):
    A[i, :] = oh_encoder(tokens)

# Step 7: Reduce the Dimensions of the Matrix Using t-SNE
model = TSNE(n_components=2, learning_rate=200, random_state=42)
tsne_features = model.fit_transform(A)

# Assign the t-SNE features to the DataFrame
moisturizers_dry.loc[:, 'X'] = tsne_features[:, 0]
moisturizers_dry.loc[:, 'Y'] = tsne_features[:, 1]

# Step 8: Plot a Scatter Plot with the Vectorized Items
source = ColumnDataSource(moisturizers_dry)

# Create the plot
plot = figure(title="Cosmetic Ingredient Similarity Map", 
              x_axis_label='T-SNE 1', y_axis_label='T-SNE 2',
              width=800, height=600)

# Add circle renderer
plot.circle(x='X', y='Y', size=10, source=source, color='navy', alpha=0.6)

# Step 9: Add a Hover Tool
hover = HoverTool()
hover.tooltips = [
    ('Item', '@Name'),
    ('Brand', '@Brand'),
    ('Price', '$@Price'),
    ('Rank', '@Rank')
]

plot.add_tools(hover)

# Step 10: Display the Plot
show(plot)

# Step 11: Print Ingredients for Two Similar Products
similar_products = moisturizers_dry.iloc[[0, 1]]
print("Product 1:", similar_products.iloc[0]['Name'], "\nIngredients:", similar_products.iloc[0]['Ingredients'])
print("\nProduct 2:", similar_products.iloc[1]['Name'], "\nIngredients:", similar_products.iloc[1]['Ingredients'])
