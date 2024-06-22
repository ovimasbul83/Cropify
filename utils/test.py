import pandas as pd

# Load the CSV file
file_path = 'Crop_Recommendation.csv'
data = pd.read_csv(file_path)

# Define lists of crops and fruits (this should be tailored to your specific dataset)
crops = ["Rice", "Maize", "ChickPea", "KidneyBeans", "PigeonPeas", "MothBeans", "MungBean", "Blackgram", "Lentil", "Jute", "Coffee", "Cotton"]
fruits = ["Pomegranate", "Banana", "Mango", "Grapes", "Watermelon", "Muskmelon", "Apple", "Orange", "Papaya", "Coconut"]
others = []  # Example list of fruits

# Create empty DataFrames for crops and fruits test sets
crops_test_set_list = []
fruits_test_set_list = []

# Sample one row from each category
for crop in crops:
    sample = data[data['Crop'] == crop]
    crops_test_set_list.append(sample)

for fruit in fruits:
    sample = data[data['Crop'] == fruit]
    fruits_test_set_list.append(sample)

# Concatenate all the samples into single DataFrames
crops_test_set = pd.concat(crops_test_set_list, ignore_index=True)
fruits_test_set = pd.concat(fruits_test_set_list, ignore_index=True)

# Save the test sets to new CSV files
crops_test_set_file_path = 'Crops_Recommendation_train_set.csv'
fruits_test_set_file_path = 'Fruits_Recommendation_train_set.csv'

crops_test_set.to_csv(crops_test_set_file_path, index=False)
fruits_test_set.to_csv(fruits_test_set_file_path, index=False)

# Display the test sets
print("Crops Test Set:")
print(crops_test_set)

print("\nFruits Test Set:")
print(fruits_test_set)
