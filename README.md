# Charity Funding Predictor

## Instructions

### Step 1: Preprocess the Data

Using Pandas and scikit-learn’s `StandardScaler()`, we preprocessed the dataset. This step prepares the data Step 2, where we compiled, train, and evaluate the neural network model.

1. Read in the `charity_data.csv` to a Pandas DataFrame, and identify the following in your dataset:
  * What variable(s) are the target(s) for the model?
  * What variable(s) are the feature(s) for the model?

2. Drop the `EIN` and `NAME` columns.

3. Determine the number of unique values for each column.

4. For columns that have more than 10 unique values, determine the number of data points for each unique value.

5. Use the number of data points for each unique value to pick a cutoff point to bin "rare" categorical variables together in a new value, `Other`, and then check if the binning was successful.

6. Use `pd.get_dummies()` to encode categorical variables.

### Step 2: Compile, Train, and Evaluate the Model

Using TensorFlow, we’ll design a neural network, or deep learning model, to create a binary classification model that can predict if a funded organization will be successful based on the features in the dataset. Taking into account how many inputs there we will be determining the number of neurons and layers for our model. Then, we’ll compile, train, and evaluate our binary classification model to calculate the model’s loss and accuracy.

1. Create a neural network model by assigning the number of input features and nodes for each layer using TensorFlow and Keras.

2. Create the first hidden layer and choose an appropriate activation function.

3. Add a second hidden layer with an appropriate activation function.

4. Create an output layer with an appropriate activation function.

5. Check the structure of the model.

6. Compile and train the model.

7. Evaluate the model using the test data to determine the loss and accuracy.

9. Save and export your results to an HDF5 file. Name the file `AlphabetSoupCharity.h5`.

### Report
The goal of this neural network is to predict whether applicants will be successful if funded by Alphabet Soup.

#### Data Preprocessing
* `IS_SUCCESSFUL` column is used as our target variable `y`
* `APPLICATION_TYPE`, `AFFILIATION`, `CLASSIFICATION`, `USE_CASE`, `ORGANIZATION`, `STATUS`, `INCOME_AMT`, `SPECIAL_CONSIDERATIONS`, `ASK_AMT` is used as our features `X` after encoding catagorical variables using **pd.getdummies()**
* `EIN`, `NAME` columns are dropped from the data set because they are neither targets nor features

#### Compiling, Training, and Evaluating the Model
* 2 hidden layer are choosen for our neural network model, 1st layer consists of 80 neural nodes with 42 input features and for 2nd layer 30 nodes are choosen.
* Model performance on test data set - Loss: 0.57, Accuracy: 0.73
* Increasing the number of Epoch for training the model didn't have much impact on the performance of the model. Choosing the right loss function for compiling the model helped train the model better.

#### Summary
Given the complexity of the problem at hand this neural network model perfomed pretty good on the testing data set. But its always hard to explain the outcome with the neural network network model. K-NN can also be used to solve this problem, as the applicants can be classified based on similarity. K-NN model during training phase can store the data and with the new data point will lie in which of these categories


