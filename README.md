# starry-script

This repository contains Python code designed to complement the [Starry Data App](https://starrydata.vercel.app), a tool for interactively generating and visualizing datasets through a drawing interface. The script extends the functionality of the app by providing a way to visualize decision boundaries of classifiers using the generated datasets.

## Features of the Companion Script

This script allows users to:

- Visualize **decision boundaries** for two classifiers: Logistic Regression and Decision Tree.
- Accept datasets exported from the Starry Data App (CSV format).
- Save decision boundary visualizations as PDF files.

## Usage

### How to Run the Script

1. Export your dataset from the Starry Data App as a CSV file.
2. Run the script using the following command:

```bash
python decision_boundary_plot.py -c <classifier_type> -s <output_pdf_file>
```

#### Command-line Arguments

- `-f`, `--file` (optional): Path to the exported CSV file from the Starry Data App. Defaults to `drawn_points.csv`.
- `-c`, `--classifier` (optional): Choose the classifier type:
  - `logreg`: Logistic Regression (default)
  - `dectree`: Decision Tree
- `-s`, `--savefile` (optional): Path to save the decision boundary visualization as a PDF file. If not specified, the plot will be displayed interactively.


## Development Notes

This script complements the [Starry Data App](https://starrydata.vercel.app) by offering additional analysis capabilities for the generated datasets. It was designed to provide seamless integration with the app's exported data format and enable quick visualization of classification results.

---

For more information about the Starry Data App, visit the main project repository or contact Andrew R. Garcia.

