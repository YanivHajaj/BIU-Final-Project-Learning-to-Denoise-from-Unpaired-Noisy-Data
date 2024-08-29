import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

parser = argparse.ArgumentParser(description='Graph exp Nr2N')
parser.add_argument('--exp', default=1, type=str)
parser.add_argument('--set', default=20, type=str)

opt = parser.parse_args()

experiment_num = opt.exp
dataset = opt.set

# Ensure the directory exists
graph_path = os.path.join('./results', dataset, f'graphs/exp{experiment_num}')
os.makedirs(graph_path, exist_ok=True)  # Create directory if it doesn't exist

def plot_csv_graph(file_path, x_column, y_columns, name, save_path=None):
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

    # Save the graph if a save path is provided
    if save_path:
        plt.savefig(save_path)

    # Display the graph
    plt.show()

if __name__ == "__main__":
    # Example usage
    file_path = os.path.join('./results', dataset, f'csvs/exp{experiment_num}/SSIM_all_images_average.csv')
    x_column = 'k'  # replace with your x-axis column name
    y_columns = ['overlap_mean', 'overlap_median', 'overlap_trimmed_mean', 'prediction']  # replace with your y-axis column names
    plot_csv_graph(file_path, x_column, y_columns, 'SSIM', os.path.join(graph_path, 'average_SSIM.png'))

    file_path = os.path.join('./results', dataset, f'csvs/exp{experiment_num}/PSNR_all_images_average.csv')
    plot_csv_graph(file_path, x_column, y_columns, 'PSNR', os.path.join(graph_path, 'average_PSNR.png'))
