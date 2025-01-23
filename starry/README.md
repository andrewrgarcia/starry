
# starry/script

```bash
starry/
├── starry/
│   ├── README
│   └── script.py
├── README.md
└── requirements.in
```

The `script.py` allows users to:

- Visualize **decision boundaries** for two classifiers: Logistic Regression and Decision Tree.
- Accept datasets exported from the Starry Data App (CSV format).
- Save decision boundary visualizations as PDF files.

## Usage

### How to Run the Script

1. Export your dataset from the Starry Data App as a CSV file.
2. Run the script using the following command:

```bash
python decision_boundary_plot.py -c logreg
```

#### Command-line Arguments

- `-f`, `--file` (optional): Path to the exported CSV file from the Starry Data App. Defaults to `drawn_points.csv`.
- `-c`, `--classifier` (optional): Choose the classifier type:
  - `logreg`: Logistic Regression (default)
  - `dectree`: Decision Tree
- `-s`, `--savefile` (optional): Path to save the decision boundary visualization as a PDF file. If not specified, the plot will be displayed interactively.

### Source Code

```python
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import LabelEncoder
import argparse

def plot_decision_boundary(filename, classifier_type="logreg", savefile=None):
    """
    Plots the decision boundary of a classifier (Logistic Regression or Decision Tree) on 2D data.

    Parameters:
        filename (str): Path to the CSV file containing the data.
        classifier_type (str): Type of classifier to use, either 'logreg' or 'dectree'.
        savefile (str, optional): Path to save the plot as a PDF file. If None, the plot is displayed interactively.

    Returns:
        None
    """
    # Load the data
    df = pd.read_csv(filename)

    # Prepare the data
    X = df[['x', 'y']].values
    y = LabelEncoder().fit_transform(df['feature'])  # Encode the 'feature' column into numerical labels

    # Choose classifier based on input
    if classifier_type == "logreg":
        classifier = LogisticRegression().fit(X, y)
    elif classifier_type == "dectree":
        classifier = DecisionTreeClassifier().fit(X, y)
    else:
        raise ValueError("Invalid classifier_type. Choose either 'logreg' or 'dectree'.")

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

    # Add legend
    plt.legend()

    # Save or show the plot
    if savefile:
        plt.savefig(savefile, format="pdf")
        print(f"Plot saved to {savefile}")
    else:
        plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot decision boundary using a classifier.")
    parser.add_argument("-f", "--file", type=str, default="drawn_points.csv", help="Path to the CSV file containing the data.")
    parser.add_argument("-c", "--classifier", type=str, default="logreg", choices=["logreg", "dectree"], help="Type of classifier to use: 'logreg' for Logistic Regression or 'dectree' for Decision Tree.")
    parser.add_argument("-s", "--savefile", type=str, help="Path to save the plot as a PDF file.")

    args = parser.parse_args()
    plot_decision_boundary(args.file, args.classifier, args.savefile)
```

## Development Notes

This script complements the [Starry Data App](https://starrydata.vercel.app) by offering additional analysis capabilities for the generated datasets. It was designed to provide seamless integration with the app's exported data format and enable quick visualization of classification results.

