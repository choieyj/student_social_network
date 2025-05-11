# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load the dataset
df = pd.read_csv('students_social_network.csv')

# Clean 'age' and 'NumberOffriends' to ensure they are numeric
# If parsing fails (because of '18. Jul' etc.), set as NaN
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['NumberOffriends'] = pd.to_numeric(df['NumberOffriends'], errors='coerce')

# Drop rows where 'age' or 'NumberOffriends' couldn't be converted
df.dropna(subset=['age', 'NumberOffriends'], inplace=True)


# Preview
print(df.head())

# Drop missing values if any
df.dropna(inplace=True)

# Select features for clustering
# Exclude gradyear, gender (if you want, can keep gender if needed), and NumberOffriends separately
features = [
    'age', 'NumberOffriends', 'basketball', 'football', 'soccer', 'softball',
    'volleyball', 'swimming', 'cheerleading', 'baseball', 'tennis', 'sports',
    'cute', 'sex', 'sexy', 'hot', 'kissed', 'dance', 'band', 'marching', 'music', 'rock',
    'god', 'church', 'jesus', 'bible', 'hair', 'dress', 'blonde', 'mall',
    'shopping', 'clothes', 'hollister', 'abercrombie', 'die', 'death', 'drunk', 'drugs'
]

X = df[features]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Find optimal k using Elbow Method
inertias = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Plot Elbow curve
plt.figure(figsize=(8,5))
plt.plot(k_values, inertias, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method to Determine Optimal k')
plt.grid(True)
plt.show()

# From elbow plot, assume k=3 (adjust if elbow is somewhere else)
k_optimal = 3
kmeans = KMeans(n_clusters=k_optimal, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Analyze cluster centers (mean values)
cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=features)
print("\nCluster Centers:")
print(cluster_centers)

# Optional: visualize clusters in 2D using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=df['Cluster'], cmap='viridis', s=50)
plt.title('Clusters Visualized with PCA')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.grid(True)
plt.show()

# Save clustered data (optional)
df.to_csv('students_social_network_clustered.csv', index=False)

# View sample students from each cluster
for cluster in range(k_optimal):
    print(f"\nSample students from Cluster {cluster}:")
    display(df[df['Cluster'] == cluster].sample(2, random_state=cluster))
