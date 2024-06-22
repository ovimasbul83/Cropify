import pandas as pd
df= pd.read_csv('/media/mash/77f10723-17ac-4e3a-b138-4565dae2f76c/crop recommneder/Crops_Recommendation_train_set.csv')
filtered_df=df[df['Response'].notnull()]
filtered_df.to_csv("crop_test_with_response.csv")