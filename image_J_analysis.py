import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Read the CSV file ---
try:
    df = pd.read_csv('points.csv')
except FileNotFoundError:
    print("Error: 'points.csv' not found. Please create this file with x1,y1,x2,y2 columns.")
    exit()

# --- 2. Define a function to calculate the angle ---
def calculate_angle(row):
    """
    Calculates the angle between two vectors defined by the row's x and y coordinates.
    """
    # Create numpy arrays for the two vectors
    vector1 = np.array([row['X1'], row['Y1']])
    vector2 = np.array([row['X2'], row['Y2']])

    # Calculate the dot product
    dot_product = np.dot(vector1, vector2)

    # Calculate the magnitudes of the vectors
    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    # Handle the case of zero-length vectors to avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return np.nan # Not a Number, as the angle is undefined

    # Calculate the cosine of the angle, ensuring the value is within the valid range for arccos [-1, 1]
    # np.clip is used to handle potential floating-point inaccuracies [9]
    cosine_angle = np.clip(dot_product / (magnitude1 * magnitude2), -1.0, 1.0)

    # Calculate the angle in radians
    angle_radians = np.arccos(cosine_angle)

    # Convert the angle to degrees
    angle_degrees = np.degrees(angle_radians)

    return angle_radians, angle_degrees

# --- 3. Apply the function to each row of the DataFrame ---
# The .apply() method iterates over each row of the DataFrame and applies the function
results = df.apply(calculate_angle, axis=1)

# --- 4. Store and display the results ---
df[['angle_radians', 'angle_degrees']] = pd.DataFrame(results.tolist(), index=df.index)

print("\nCalculated angles:")
print(df)