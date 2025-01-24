import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import LabelEncoder
import argparse

def prepare_for_classification(df, columns):
    _, x_col, y_col, feature_col, _ = columns
    
    X = df[[x_col, y_col]].values
    y = LabelEncoder().fit_transform(df[feature_col])  # Encode the 'feature' column into numerical labels
    return X, y

def classify(X, y, classifier_type):
    if classifier_type == "logreg":
        classifier = LogisticRegression().fit(X, y)
    elif classifier_type == "dectree":
        classifier = DecisionTreeClassifier().fit(X, y)
    else:
        raise ValueError("Invalid classifier_type. Choose either 'logreg' or 'dectree'.")
    return classifier

def plot_decision_boundary(df, X, classifier, columns, savefile):
    _, x_col, y_col, feature_col, color_col = columns

    disp = DecisionBoundaryDisplay.from_estimator(
        classifier, X, response_method="predict",
        xlabel=x_col, ylabel=y_col,
        alpha=0.5,
    )

    for label in df[feature_col].unique():
        mask = df[feature_col] == label     # Selective plot depending on featuretype
        color = df.loc[mask, color_col].iloc[0] if color_col else "blue"
        disp.ax_.scatter(
            df.loc[mask, x_col], df.loc[mask, y_col],
            c=color, label=label, edgecolor="k"
        )

    plt.legend()

    if savefile:
        plt.savefig(savefile, format="pdf")
        print(f"Plot saved to {savefile}")
    else:
        plt.show()


def classify_and_plot(filename, classifier_type="logreg", savefile=None):
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
    columns = list(df.columns)
    print("Columns are:", columns)
    
    X, y = prepare_for_classification(df, columns)
    classifier = classify(X, y, classifier_type)
    plot_decision_boundary(df, X, classifier, columns, savefile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot decision boundary using a classifier.")
    parser.add_argument(
        "-f", "--file",
        type=str,
        default="drawn_points.csv",
        help="Path to the CSV file containing the data (default: 'drawn_points.csv')."
    )
    parser.add_argument(
        "-c", "--classifier",
        type=str,
        default="logreg",
        choices=["logreg", "dectree"],
        help="Type of classifier to use: 'logreg' for Logistic Regression or 'dectree' for Decision Tree. Default is 'logreg'."
    )
    parser.add_argument(
        "-s", "--savefile",
        type=str,
        help="Path to save the plot as a PDF file. If not provided, the plot will be displayed interactively."
    )

    # Print help message if -h or --help is used
    args = parser.parse_args()

    classify_and_plot(args.file, args.classifier, args.savefile)
