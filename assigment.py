import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_dataset():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
    data = pd.read_csv(url, names=column_names)
    return data


df = load_dataset()


def basic_statistics(df):
    print("Descriptive statistics:\n", df.describe())

    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    correlation_matrix = numeric_df.corr()

    print("\nCorrelation matrix:\n", correlation_matrix)
    return df.describe(), correlation_matrix


stats, corr = basic_statistics(df)


def plot_bar_chart(df):
    plt.figure(figsize=(8, 5))
    sns.countplot(x='species', data=df)
    plt.title("Count of Each Species in the Dataset")
    plt.xlabel("Species")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("species_count_bar_chart.png")
    plt.show()


def plot_scatter(df):
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='petal_length', y='petal_width', hue='species', data=df)
    plt.title("Scatter Plot of Petal Length vs Petal Width by Species")
    plt.xlabel("Petal Length")
    plt.ylabel("Petal Width")
    plt.legend(title="Species")
    plt.tight_layout()
    plt.savefig("petal_scatter_plot.png")
    plt.show()


def plot_heatmap(corr):
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Heatmap of Feature Correlations")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png")
    plt.show()


plot_bar_chart(df)
plot_scatter(df)
plot_heatmap(corr)
