import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import knn_algo
from mlxtend.plotting import plot_decision_regions

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#importing the dataset
names = ['Sepal_length',
         'Sepal_width',
         'Petal_length',
         'Petal_width',
         'class']
flowers = pd.read_csv('iris.csv', names = names, header = None)

#Visiualizing the dataset
colors = ['red', 'green', 'purple']

plt.figure(figsize = (12, 8))

plt.subplot(121)
sns.scatterplot(
    data = flowers,
    x = 'Sepal_length',
    y = 'Sepal_width',
    hue = 'class',
    style = 'class',
    palette = colors
    )
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.title("Sepal")


plt.subplot(122)
sns.scatterplot(
    data = flowers,
    x = 'Petal_length',
    y = 'Petal_width',
    hue = 'class',
    style = 'class',
    palette = colors
    )

plt.xlabel("Petal Length (cm)")
plt.ylabel("Petal Width (cm)")
plt.title("Petal")
plt.show()


#Building out the data for prediction
X = np.array(flowers.drop(columns = ['class']))
y = np.array(flowers['class'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)

#Starting the prediction model
Knn = knn_algo.KnnClassifier(1)

Knn.fit(X_train, y_train)

pred = Knn.predict(X_test)

print("accuracy: {}".format(accuracy_score(y_test, pred)))



# ─── Decision boundary visualization (on petal features only) ──────
# Select only the two most discriminative features
X_2d = flowers[['Petal_length', 'Petal_width']].values

# Scale the 2D data (very important for KNN)
scaler = StandardScaler()
X_2d_scaled = scaler.fit_transform(X_2d)

# Train a new model just for visualization (on scaled 2D data)
knn_vis = knn_algo.KnnClassifier(k=5)
knn_vis.fit(X_2d_scaled, y)   # using full data for clean boundary

# Plot decision regions
plt.figure(figsize=(9, 7))
plot_decision_regions(
    X_2d_scaled,
    y,
    clf=knn_vis,
    legend=2,
    colors='red,green,purple',
    hide_spines=False
)

plt.xlabel('Petal Length (standardized)')
plt.ylabel('Petal Width (standardized)')
plt.title('KNN Decision Boundary (k=5) – Petal Features')
plt.grid(True, alpha=0.3)
plt.show()
