import pandas as pd

# Load the test set
test_set_file_path = '/media/mash/77f10723-17ac-4e3a-b138-4565dae2f76c/crop recommneder/Crops_Recommendation_train_set.csv'
test_set = pd.read_csv(test_set_file_path)

# Define lists of crops, fruits, and other types
crops = ["Rice", "Maize", "ChickPea", "KidneyBeans", "PigeonPeas", "MothBeans", "MungBean", "Blackgram", "Lentil", "Jute", "Coffee", "Cotton"]
fruits = ["Pomegranate", "Banana", "Mango", "Grapes", "Watermelon", "Muskmelon", "Apple", "Orange", "Papaya", "Coconut"]
others = []

# Function to classify and generate the prompt
def generate_prompt(row):
    crop_name = row['Crop']
    classification = "something else"
    if crop_name in crops:
        classification = "crop"
    elif crop_name in fruits:
        classification = "fruit"
    
    user_question = (
        f"If Nitrogen level is {row['Nitrogen']}, "
        f"Phosphorus level is {row['Phosphorus']}, "
        f"Potassium level is {row['Potassium']}, "
        f"Temperature is {row['Temperature']}°C, "
        f"Humidity is {row['Humidity']}%, "
        f"pH value is {row['pH_Value']}, "
        f"and Rainfall is {row['Rainfall']} mm. "
        f"The {classification} is {crop_name}."
    )
    return user_question

# Generate prompts for each row in the test set
prompts = test_set.apply(generate_prompt, axis=1)

# Save the prompts to a new CSV file
prompts_df = pd.DataFrame(prompts, columns=["Prompt"])
prompts_file_path = 'Crops_Recommendation_Prompts.csv'
prompts_df.to_csv(prompts_file_path, index=False)

print("Prompts have been saved to:", prompts_file_path)
