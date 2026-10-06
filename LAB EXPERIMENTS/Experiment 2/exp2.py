import pandas as pd
import os

# Get the folder where this Python file is saved
folder = os.path.dirname(os.path.abspath(__file__))

# CSV file path
csv_path = os.path.join(folder, "training_data.csv")

# Create training data
data = {
    "Sky": ["Sunny", "Sunny", "Rainy", "Sunny"],
    "Temperature": ["Warm", "Warm", "Cold", "Warm"],
    "Humidity": ["Normal", "High", "High", "High"],
    "Wind": ["Strong", "Strong", "Strong", "Strong"],
    "Water": ["Warm", "Warm", "Warm", "Cool"],
    "Forecast": ["Same", "Same", "Change", "Change"],
    "EnjoySport": ["Yes", "Yes", "No", "Yes"]
}

# Create and save CSV file
df = pd.DataFrame(data)
df.to_csv(csv_path, index=False)

print("CSV file created at:")
print(csv_path)

# Read CSV file
data = pd.read_csv(csv_path)

print("\nTraining Data:")
print(data)

# Candidate Elimination
attributes = list(data.columns[:-1])
n = len(attributes)

# Initial Specific and General boundaries
S = ["0"] * n
G = [["?"] * n]

for index, row in data.iterrows():

    example = list(row.iloc[:-1])
    target = row.iloc[-1]

    if target == "Yes":

        # Generalize S
        for i in range(n):
            if S[i] == "0":
                S[i] = example[i]
            elif S[i] != example[i]:
                S[i] = "?"

        # Remove G hypotheses that don't cover the positive example
        G = [
            g for g in G
            if all(g[i] == "?" or g[i] == example[i] for i in range(n))
        ]

    else:

        # Specialize G
        new_G = []

        for g in G:
            for i in range(n):
                if g[i] == "?":
                    for value in data[attributes[i]].unique():
                        if value != example[i]:
                            h = g.copy()
                            h[i] = value

                            # Check whether h is more general than S
                            valid = True
                            for j in range(n):
                                if S[j] != "0" and h[j] != "?" and h[j] != S[j]:
                                    valid = False
                                    break

                            if valid:
                                new_G.append(h)

        G = new_G

print("\nFinal Specific Boundary:")
print(S)

print("\nFinal General Boundary:")
for g in G:
    print(g)
