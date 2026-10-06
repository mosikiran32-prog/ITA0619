
# FIND-S Algorithm

import pandas as pd

# Training dataset
data = [
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
]

columns = ['Sky', 'Temperature', 'Humidity',
           'Wind', 'Water', 'Forecast', 'EnjoySport']

df = pd.DataFrame(data, columns=columns)
print("Training Data:")
print(df)

# Initialize hypothesis
hypothesis = ['0'] * (len(columns) - 1)

# FIND-S
for row in data:
    if row[-1] == 'Yes':
        for i in range(len(hypothesis)):
            if hypothesis[i] == '0':
                hypothesis[i] = row[i]
            elif hypothesis[i] != row[i]:
                hypothesis[i] = '?'

print("\nFinal Hypothesis:")
print(hypothesis)
