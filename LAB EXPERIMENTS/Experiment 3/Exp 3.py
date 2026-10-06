import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text

# Training dataset
data = {
    'Outlook': [
        'Sunny', 'Sunny', 'Overcast', 'Rain', 'Rain',
        'Rain', 'Overcast', 'Sunny', 'Sunny', 'Rain',
        'Sunny', 'Overcast', 'Overcast', 'Rain'
    ],
    'Temperature': [
        'Hot', 'Hot', 'Hot', 'Mild', 'Cool',
        'Cool', 'Cool', 'Mild', 'Cool', 'Mild',
        'Mild', 'Mild', 'Hot', 'Mild'
    ],
    'Humidity': [
        'High', 'High', 'High', 'High', 'Normal',
        'Normal', 'Normal', 'High', 'Normal', 'Normal',
        'Normal', 'High', 'Normal', 'High'
    ],
    'Wind': [
        'Weak', 'Strong', 'Weak', 'Weak', 'Weak',
        'Strong', 'Strong', 'Weak', 'Weak', 'Weak',
        'Strong', 'Strong', 'Weak', 'Strong'
    ],
    'PlayTennis': [
        'No', 'No', 'Yes', 'Yes', 'Yes',
        'No', 'Yes', 'No', 'Yes', 'Yes',
        'Yes', 'Yes', 'Yes', 'No'
    ]
}

df = pd.DataFrame(data)

# Convert categorical values to numerical values
X = pd.get_dummies(df.drop('PlayTennis', axis=1))
y = df['PlayTennis']

# Build ID3 decision tree
model = DecisionTreeClassifier(
    criterion='entropy',
    random_state=0
)

model.fit(X, y)

# Display tree
print("Decision Tree:")
print(export_text(model, feature_names=list(X.columns)))

# New sample
new_sample = pd.DataFrame([{
    'Outlook': 'Sunny',
    'Temperature': 'Cool',
    'Humidity': 'High',
    'Wind': 'Strong'
}])

new_sample = pd.get_dummies(new_sample)
new_sample = new_sample.reindex(columns=X.columns, fill_value=0)

# Classification
prediction = model.predict(new_sample)

print("\nNew Sample:")
print("Outlook = Sunny, Temperature = Cool, Humidity = High, Wind = Strong")

print("\nClassification Result:")
print("PlayTennis =", prediction[0])
