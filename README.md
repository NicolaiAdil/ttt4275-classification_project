### Introduction

Classification project in the course [TTT4275 - Estimation, Detection and Classification](https://www.ntnu.edu/studies/courses/TTT4275#tab=omEmnet). The project is divided into two parts, "The Iris task", and "Classification of handwritten numbers 0-9" (MNIST). See the project description for more information.

### Installation

Create a python virtual environment, where `VENV_NAME` is a variable name of your choice
```bash 
python3 -m venv VENV_NAME
```

Source it (this is specific for Linux, if you have another OS look [here](https://docs.python.org/3/library/venv.html#how-venvs-work))
```bash 
source VENV_NAME/bin/activate
```

And install the necessary packages
```bash 
pip3 install -r requirements.txt   
```

### Running the Programs

When running the program a menu will show up. Choose the alternative you want to test.

#### Part 1: The Iris Task

```bash 
python3 iris_task/main.py
```

#### Part 2: Classification of Handwritten Numbers (MNIST)

```bash 
python3 mnist_task/main.py
```

### Configuration

To properly test the efficiency of training in batches vs having everything in memory (Which is alternative 3 in the MNIST task), you need to change `TEST_SAMPLE_SIZE` to a larger number.