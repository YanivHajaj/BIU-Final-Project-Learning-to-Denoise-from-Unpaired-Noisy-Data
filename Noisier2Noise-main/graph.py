import pandas as pd
import matplotlib.pyplot as plt

def plot_csv_graph(file_path, x_column, y_column):
    # Read the CSV file into a pandas DataFrame
    data = pd.read_csv(file_path)

    # Plot the graph
    plt.figure(figsize=(10, 6))
    plt.plot(data[x_column], data[y_column], marker='o', linestyle='-', color='b')

    # Add titles and labels
    plt.title(f'{y_column} vs {x_column}')
    plt.xlabel(x_column)
    plt.ylabel(y_column)

    # Display the graph
    plt.show()

if __name__ == "__main__":
    # Example usage
    file_path = 'SSIM_all_images_average.csv'  # replace with your CSV file path
    x_column = 'k'  # replace with your x-axis column name
    y_column = 'overlap_mean'  # replace with your y-axis column name
    plot_csv_graph(file_path, x_column, y_column)


