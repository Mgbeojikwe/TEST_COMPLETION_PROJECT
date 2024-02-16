#This source code assumes that:
#1) The data is clean and elements in each column of the pandas dataframe are all strings
#2) The label is made the last column in the dataset

import pandas as pd
import xlrd
import numpy as np
import tensorflow as tf



def load_data(file_name):
    if file_name.endswith(".csv"):
        return pd.read_csv(file_name)

    elif file_name.endswith("xlsx"):
      return pd.read_excel(file_name)

    elif file_name.endswith("txt"):
        return  pd.read_csv(file_name,sep=" ",header=0)
    else:
      pass


class Encoding:


  def __init__(self,file_name):
    #let us instead consider having the file name as instance attribute
    file_name = str(file_name)

    self.file_name=file_name


    if self.file_name.endswith("csv") or self.file_name.endswith("xlsx") or self.file_name.endswith("txt"):
          self.__dataframe = load_data(self.file_name)

    elif self.file_name.startswith("http"):

          url=self.file_name
          filename="filename"+url[-4:]

          import requests

          response = requests.get(url, allow_redirects=True)
          with open(filename, mode="wb") as fh:
            fh.write(response.content)

          self.__dataframe = load_data(filename)


    else:
      print("could not recognize the extension of the file")



    self.__columns=list(self.__dataframe.columns.values)                        #getting the names of the columns in the dataset

    self.__words_lists = [ [] for column in self.__columns]                     #a 2-D list where by each list will later contain the unique
                                                                                #words in their respective coulmns. the number of empty lists
                                                                                #is equal to the number of columns in the dataframe. The last column
                                                                                #is taken to be the vector of labels


    #self.__indexes_lists= [ [0  for i in range(len(words_in_first_column))] for index in self.__columns]  #this is also a 2D list of whereby each list will be used to store
    self.__indexes_lists =[]                                              #the indexes of each unique words obtained from the respective columns


    self.__sc_x=0
    self.__sc_y=0
    (self.__x_test, self.__x_train, self.__y_test, self.__y_train)=(0,0,0,0)
    self.__ann=0
    self.__features=[]
    self.__label=[]

    self.__length_of_longest_list=0


  def encoder(self):
    """
    This method is used in encoding the entire pandas dataframe i.e (features and label).


    this method does the following:
    a) iterate through the columns in the dataset
    b) fill empty cells in column "i" with zeros
    c) get the words(including zeros) in columns "i"
    d) assign the result of step c (the words in column_i) to the list at same index in in "self.__words_lists".
    e) With the help of the number of unique elements in step c, create a range of indexes and , after converting the results to array, assign them to, at a given index, in "self.__indexes_lists"
    f) convert "self.__indexes_list" to an array

    This method encodes the words in the dataset so as to enable usage of the dataset in the Machine learning model

    """

    for index, column_name in enumerate(self.__columns):


                  self.__dataframe[column_name] = self.__dataframe[column_name].fillna(0)                 #filling up empty cells with zero

                  words_list=list(self.__dataframe[column_name])
                  self.__words_lists[index]= words_list                                                   #an recording them in row "i" in self.__words_lists

                  number_of_words_in_column_i = len(words_list)                                    #encoding  the words in column "i" and assiging it to row "i" in self.__indexes_lists
                  self.__indexes_lists. append(list(range(1,number_of_words_in_column_i+1)))      # the words are encoded starting from 1.




    self.__indexes_lists = np.array(self.__indexes_lists)                       #converting the lists to array make suitable for ML models
    self.__words_lists = np.array(self.__words_lists)


    #since the words, and hence the resulting encodings, of a column were transformed to rows,
    #we need to reverse "self.__indexes_lists"  and "self.__words_lists" to a structure
    #that is same as the original dataframe. This is achieved by transposing
    self.__words_lists = self.__words_lists.transpose()
    self.__indexes_lists = self.__indexes_lists.transpose()


  def train_model(self):

      """
          This method trains the NN model. The model has two hidden layers with each using the RelU activation function, and it carries out its operation through the steps:

          1)


          The method trains then Neural network model. The model has two hidden layers with each utilzing the Relu activation ficntion.
      """

      #use Neural Network in training the model, compare result obtained when feature scaling is applied to only features and (features and label)
      #also compare the result obtained when you didn't standardize



      self.encoder()



      self.__features = self.__indexes_lists[:,:-1]
      self.__label = self.__indexes_lists[:,-1].reshape(-1,1)


      from sklearn.model_selection import train_test_split

      self.__x_train, self.__x_test, self.__y_train, self.__y_test = train_test_split(self.__features, self.__label,test_size=0.2, random_state=1)

      from sklearn. preprocessing import StandardScaler

      self.__sc_x =  StandardScaler()
      self.__x_train = self.__sc_x.fit_transform(self.__x_train)
      self.__x_test = self.__sc_x.transform(self.__x_test)

      self.__sc_y = StandardScaler()
      self.__y_train=self.__sc_y.fit_transform(self.__y_train)


      #creating the neurons, layers and optimizer
      num_of_features = len(self.__columns)-1
      num_of_neurons = int((num_of_features + 1)/2)

      self.__ann = tf.keras.models.Sequential()
      self.__ann.add(tf.keras.layers.Dense(units=num_of_neurons,activation="relu"))            #adding the first hidden layer
      self.__ann.add(tf.keras.layers.Dense(units=num_of_neurons,activation="relu"))            #adding the second hidden layer
      self.__ann.add(tf.keras.layers.Dense(units=1,activation="linear"))                       #adding the outer layer needed for a regression analysis
      self.__ann.compile(optimizer= "adam", loss="mse" ,metrics=["accuracy"] )

      #step 2: Training the model
      size_of_data_per_training= len(self.__indexes_lists)//4

      self.__ann.fit(self.__x_train,self.__y_train, batch_size= size_of_data_per_training,epochs=200)


  def predict(self):

      y_test_predicted = self.__ann.predict(self.__x_test)

      #on getting the predicted labels, we have to inverse-transform them

      y_test_predicted = self.__sc_y.inverse_transform(y_test_predicted)

      predicted_words=[]

      # loop through the predicted y values, then find their index position in "self.__label"
      # then utilize that index position in getting the predicted words in "self.__words_lists"

      for y_predicted in y_test_predicted:

              y_predicted_index = self.__label.index(y_predicted)

              last_column_in_words_lists = self.__words_lists[:,-1]
              predicted_word = last_column_in_words_lists[y_predicted_index]

              predicted_words.append(predicted_word)

      print(predicted_words)


  def complete_the_sentence(self, sentence):
    """
        This method completes our sentence. The input sentence is used as feature in predicting
        the target.
    """
    self.train_model()
    encode_for_entire_sentence_1_D = []
    encode_for_entire_sentence_2_D = []


    sentence=str(sentence).lower()
    list_of_words = sentence.split()

    #we have to ensure that the number of words in "sentence" is never less than the dimensionality of the features
    num_of_features_columns = len(self.__columns)-1

    if num_of_features_columns >  len(list_of_words):
          diff = num_of_features_columns - len(list_of_words)

          for i in range(diff):
                list_of_words.append("0")



    # getting the index position of each word.
    # Assumption: thw words are searched index-wise in the fields in "self.__dataset".i.e the first word is search in first column,
    #second word searched in second column and so on.

    for index,word in enumerate(list_of_words):

          column_i_where_word_is_to_be_searched = self.__words_lists[:,index].tolist()
          encode_list_for_column_i = self.__indexes_lists[:,index].tolist()

          if word in column_i_where_word_is_to_be_searched:

                   word_index = column_i_where_word_is_to_be_searched.index(word)

                   encode_value_for_word = encode_list_for_column_i[word_index]                               #search for the encode value allocated to that word in column "i"
                   encode_for_entire_sentence_1_D.append(encode_value_for_word)

          else:
                encode_for_entire_sentence_1_D.append(0)                                                    #as the size of the data increase, the probability of
                                                                                                            # appending zero , as a result of absence of the word in the searched column, will decrease

      #after working on all words in the sentence, we need to utilize the encode as a 2D-array
    encode_for_entire_sentence_1_D = np.array(encode_for_entire_sentence_1_D)

    encode_for_entire_sentence_2_D .append(encode_for_entire_sentence_1_D)

    scaled_encoded_sentence = self.__sc_x.transform(encode_for_entire_sentence_2_D)                   #apply feature scaling using the standard deviation used on the training data

    predicted_completing_label = self.__ann.predict(scaled_encoded_sentence)            #where predicted completing sentence is the predicted label for the incomplete sentence


    predicted_completing_label = predicted_completing_label.reshape(-1,1)

    predicted_completing_label = self.__sc_y.inverse_transform(predicted_completing_label)

    predicted_encode = predicted_completing_label[0][0]
    predicted_encode = round(predicted_encode)                                  #round-up the predicted encode to the nearest whole number
    #find the index position of the predicted label in "self.__label"
    label = self.__label.tolist()
    predicted_completing_label_index = label.index(predicted_encode)


    target_column = self.__words_lists[:,-1]

    predicted_completing_word= target_column[predicted_completing_label_index]

    print(f"{sentence} {predicted_completing_word}")




