import matplotlib.pyplot as plt
import pandas
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import LabelEncoder

df = pandas.read_csv("drawn_points.csv")

# Prepare the data
X = df[['x', 'y']].values
y = LabelEncoder().fit_transform(df['feature'])  # Encode the 'feature' column into numerical labels

# Train a Logistic Regression classifier
# classifier = LogisticRegression().fit(X, y)
classifier = DecisionTreeClassifier().fit(X, y)

# Plot decision boundary
disp = DecisionBoundaryDisplay.from_estimator(
    classifier, X, response_method="predict",
    xlabel="x", ylabel="y",
    alpha=0.5,
)

# Scatter plot with colors
for label, color in zip(df['feature'].unique(), df['color'].unique()):
    mask = df['feature'] == label
    disp.ax_.scatter(
        df.loc[mask, 'x'], df.loc[mask, 'y'],
        c=color, label=label, edgecolor="k"
    )

# Add legend and show the plot
plt.legend()
plt.show()
