import pandas as pd
import matplotlib.pyplot as plt

def plot_csv_graph(file_path, x_column, y_columns, name):
    # Read the CSV file into a pandas DataFrame
    data = pd.read_csv(file_path)

    # Plot the graph
    plt.figure(figsize=(10, 6))
    
    for y_column in y_columns:
        plt.plot(data[x_column], data[y_column], marker='o', linestyle='-', label=y_column)

    # Add titles and labels
    plt.title(f'{name} Comparison of {y_columns} vs {x_column} for all images')
    plt.xlabel(x_column)
    plt.ylabel(name)

    # Add a legend to differentiate the graphs
    plt.legend()

    # Display the graph
    plt.show()

if __name__ == "__main__":
    # Example usage
    file_path = './csvs/SSIM_all_images_average.csv'  # replace with your CSV file path
    x_column = 'k'  # replace with your x-axis column name
    y_columns = ['overlap_mean', 'overlap_median']  # replace with your y-axis column names
    plot_csv_graph(file_path, x_column, y_columns, 'SSIM')

    file_path = './csvs/PSNR_all_images_average.csv'  # replace with your CSV file path
    x_column = 'k'  # replace with your x-axis column name
    y_columns = ['overlap_mean', 'overlap_median']  # replace with your y-axis column names
    plot_csv_graph(file_path, x_column, y_columns, 'PSNR')

