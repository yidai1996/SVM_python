import pandas as pd
from sklearn.model_selection import train_test_split

def load_dataset(path, target_col, test_size=0.2, random_state=42):
    df = pd.read_csv(path)
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return {'X_train': X_train, 'y_train': y_train,
            'X_test':  X_test,  'y_test':  y_test}

file_paths = ['data1.csv', 'data2.csv']
datasets = [load_dataset(fp, 'label') for fp in file_paths]

from sklearn.svm import SVC

kernels = ['linear', 'poly', 'sigmoid', 'rbf']
models = [
    [SVC(kernel=k).fit(ds['X_train'], ds['y_train']) for k in kernels]
    for ds in datasets
]

# Test SVM

from IPython.display import Markdown as md

table = []
for i, ds in enumerate(datasets):
    row = []
    for j, ker in enumerate(kernels):
        accuracy = models[i][j].score(ds['X_test'], ds['y_test'])
        row.append(accuracy)
    table.append(row)

header = '|  |' + ' | '.join(k for k in kernels) + ' |\n'
header += '|' + '--|'*(len(kernels)+1) + '\n'
rows = '\n'.join(
    f'| dataset {i} | ' + ' | '.join(f'{table[i][j]*100:.2f}' for j in range(len(kernels))) + ' |'
    for i in range(len(datasets))
)
md('## Sklearn SVM performance\n' + header + rows)

# Visualization

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.inspection import DecisionBoundaryDisplay

cm = plt.cm.RdBu
cm_bright = ListedColormap(["#FF0000", "#0000FF"])

def vis_decision_boundaries(datasets, kernels, models, show_support=False):
    figure = plt.figure(figsize=(len(datasets)*5, (len(kernels)+1)*3))
    for i, ds in enumerate(datasets):
        # Plot the datasets first
        ax = plt.subplot(len(datasets), len(kernels)+1, i*(len(kernels)+1)+1)
        if i == 0:
            ax.set_title('Input data')
        # Plot the training points
        ax.scatter(ds['X_train'][:, 0], ds['X_train'][:, 1], c=ds['y_train'], cmap=cm_bright, edgecolors="k")
        # Plot the testing points
        ax.scatter(
            ds['X_test'][:, 0], ds['X_test'][:, 1], c=ds['y_test'], cmap=cm_bright, alpha=0.5, edgecolors="k"
        )
        x_min, x_max = ds['X'][:, 0].min() - 0.5, ds['X'][:, 0].max() + 0.5
        y_min, y_max = ds['X'][:, 1].min() - 0.5, ds['X'][:, 1].max() + 0.5
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xticks(())
        ax.set_yticks(())

        for j, ker in enumerate(kernels):
            base_alpha = 0.2 if show_support else 1.0
            ax = plt.subplot(len(datasets), len(kernels)+1, i*(len(kernels)+1)+j+2)
            score = models[i][j].score(ds['X_test'], ds['y_test'])
            DecisionBoundaryDisplay.from_estimator(
                models[i][j], ds['X'], cmap=cm, alpha=0.8, ax=ax, eps=0.5
            )

            # Plot the training points
            ax.scatter(ds['X_train'][:, 0], ds['X_train'][:, 1], c=ds['y_train'], cmap=cm_bright, alpha=base_alpha, edgecolors="k")
            # Plot the testing points
            ax.scatter(
                ds['X_test'][:, 0], ds['X_test'][:, 1], c=ds['y_test'], cmap=cm_bright, alpha=0.5*base_alpha, edgecolors="k"
            )

            if show_support:
                # Support vectors
                support_idxs = models[i][j].support_
                support, support_y = ds['X_train'][support_idxs], ds['y_train'][support_idxs]
                ax.scatter(
                    support[:, 0], support[:, 1], c=support_y, cmap=cm_bright, edgecolors="k"
                )
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xticks(())
            ax.set_yticks(())

            if i == 0:
                ax.set_title(ker + ' supports' if show_support else ker)
            if not show_support:
                ax.text(
                    x_max - 0.3,
                    y_min + 0.3,
                    f'{score*100:.1f}%',
                    size=15,
                    horizontalalignment="right",
                    weight="bold",
                )