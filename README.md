# TEST_COMPLETION_PROJECT

This source code's aim is to assist in completing sentences; it utilizes the Neural Network model. This source code circumvents the two challenges encountered in using the sklearn "sklearn.preprocessing.HotEncoding()" class viz: 1) creating an unusual large dataframe when encoding each column of strings; as a result of creating additional columns, and 2) the  new columns are not be utilized in training the machine learing model.
To overcome these drawbacks, I created a new derived class: Encoding; "Encodign()" has three methods and they are discussed below.

method 1 :  encoder

        Since a typical Machine learing model requires that its trainging data are numeric, this class encodes the training data so that the words in the original data are represented by integers. It acheives its objectives through the steps:
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

        step 1 => create the features by selecting the first row to a penultimate row and assigning the resulting to "self.__features"

       step 2 =>  create the label by selecting the last column in "self.__indexes_list" and assign the resulting to "self.__label"

       step 3 =>  with the aid of sklearn's "sklearn.model_selection.train_test_split()" function, split the 2D-arrays obtained from step 1 and step 2  into training_set and test_set.

        step 4 =>  Due to significant difference in the elements in each column, we have to standard the data using "sklearn.preprocessign.StandardScaler()" class  so we can place all elements of the data on the same scale
