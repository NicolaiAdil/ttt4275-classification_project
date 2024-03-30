import os
import numpy as np

def load_dataset(relative_path: str = 'iris_data/iris_dataset.csv'):
    # Get the directory path of the current Python script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path_to_data = os.path.join(current_dir, relative_path)

    data = np.genfromtxt(path_to_data, delimiter=',', dtype=[('sepal_length', float),
                                                             ('sepal_width', float),
                                                             ('petal_length', float),
                                                             ('petal_width', float),
                                                             ('class', 'U20')])
    # Initialize matrices for each class
    setosa_matrix = np.zeros((50, 4))
    versicolor_matrix = np.zeros((50, 4))
    virginica_matrix = np.zeros((50, 4))

    # Populate matrices
    
    for i in range(50):
        setosa_matrix[i] = (data[i][0], data[i][1], data[i][2], data[i][3])
        versicolor_matrix[i] = (data[i+50][0], data[i+50][1], data[i+50][2], data[i+50][3])
        virginica_matrix[i] = (data[i+100][0], data[i+100][1], data[i+100][2], data[i+100][3])
            

    # Display matrices
    print("Setosa Matrix:")
    print(setosa_matrix)
    print("\n Length of Setosa Matrix: ", len(setosa_matrix))

    print("\nVersicolor Matrix:")
    print(versicolor_matrix)
    print("\n Length of Versicolor Matrix: ", len(setosa_matrix))

    print("\nVirginica Matrix:")
    print(virginica_matrix)
    print("\n Length of Virginica Matrix: ", len(setosa_matrix))

load_dataset()