# TEST_COMPLETION_PROJECT

This source code's aim is to assist in completing sentences; it utilizes the Neural Network model. sklearn "sklearn.preprocessing.HotEncoding()" class has two major drawbacks viz: 1) creating an unusual large dataframe when encoding each column of strings; as a result of creating additional columns, and 2) the  new columns are not be utilized in training the machine learing model.
To overcome these drawbacks, I created a new derived class: Encoding; "Encoding()" has three methods and they are discussed below. The latest structure of the class is found in encoding.py

method 1 :  encoder

        Since a typical Machine learing model requires that its trainging data are numeric, this class encodes the training data so that the words in the original data are represented by integers.
        It acheives its objectives through the steps:
        
        step 1 => iterate through the columns in the original data

        step 2 => for a given column "i", fill empty spaces with zeros

        step 3 => get the values of each cells in the column

        step 4 => append each cell's value to the list "self.__all_words_in_dataframe"

        step5 => on completely getting the words in the original data, utilize the set() function to prevent redundacy in elements, then convert it to a list and assign it to "self.__all_words_in_dataframe"

        step 6 => re-iterate through the columns in the original dataset

        step 7 = > for each column get the words i.e values in each cell

        step 8 => iterate through the words  obtained in step 7, get the index location of such words in "self.__all_words_in_dataframe"

        step 9 => append the resulting list of integers obtained in step 8 to "self.__indexes_lists"

        step 10 => since the encodes for a given column in the dataset is recorded as row in "self.__indexes_list", transpose "self.__indexes_lists" so that the resulting 2d-array has same structure and perfectly represent the data.


method 2 :   train_model

        This method trains the Neural network model. It uses the encoded dataframe "self.__indexes_lists" in training the ML  model. It acheives its aim using the following steps.

        step 1 => create the features by selecting the first column to a penultimate column and assigning the resulting to "self.__features"

       step 2 =>  create the label by selecting the last column in "self.__indexes_list" and assign the resulting to "self.__label"

       step 3 =>  with the aid of sklearn's "sklearn.model_selection.train_test_split()" function, split the 2D-arrays obtained from step 1 and step 2  into training_set and test_set.

        step 4 =>  Due to significant difference in the elements in each column, we have to standardize the data using "sklearn.preprocessign.StandardScaler()" class. 
                This necessary so enable the numbers are placed on same scale and all have equal influence on the prediction.
        step 5 => the training_set data is used to train the Neural Network in batch propagation mode. The Neural network used has the properties:
                a) two hidden layers
                b) utilizes the Relu  activation fucntion
                c)uses the adam optimizer
                d) uses an epoch of 400 i.e the entire data is used to train the model 400 times.
method 3        Complete_the_sentence

                This method accepts the sentence needed to be completed, and completes the sentence with the aid of the trained Neural Network model.
        It acheives its objective throught the following steps:

        step 1 => the incomplete sentence is splitted into a list of words

        step 2 => Check if the size of the list obtained in step1 equals the dimensionality of the training data, If no, append zero to the list until its size equals the dimensionality of the training data.

        step 3 => encode the  words present in the list obtained in step 2. Each  word is searched in "self.__all_words_in_dataframe", if present, then they are encoded will the number used in encoding same word in the training data.
        step 4 => convert the encoded list obtained in step 3 to a 2D-array of size 1

        step 5 => feature-scale the 2D-array obtained in step 4; the arrays are scaled using the same standard deviation that was used in scaling the training features.

        step 6  = > predict the encode for the label using the neural network object "self.__ann".

        step 7 =>  using the encode gotten from step 7, traceback the word that has such code in "self.__all_words_in_data_frame"


