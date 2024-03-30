from handle_data import load_dataset, split_dataset
from plot import plot_characteristics_separability

def main():
    setosa_matrix, versicolor_matrix, virginica_matrix = load_dataset()
    plot_characteristics_separability(setosa_matrix, versicolor_matrix, virginica_matrix)



if __name__ == '__main__':
    main()

