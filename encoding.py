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
      exit()



    self.__columns=list(self.__dataframe.columns.values)                        #getting the names of the columns in the dataset

    self.__words_lists=[]                                                       #a 2D-list whose elements are 1D-list representing the words in each column

    self.__indexes_lists =[]                                                     # a 2D-list  whose elements are 1D-list representing the encode for the words in each repective column
    self.__all_words_in_dataframe=[]

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


    #taking a record of all words in the dataframe
    for index, column_name in enumerate(self.__columns):

          self.__dataframe[column_name] = self.__dataframe[column_name].fillna(0)                     #fill empty spaces with zero

          column_i_words_list = self.__dataframe[column_name]                                         #get the words in each column
          #column_i_words_list = [ word.lower() for word in column_i_words_list]                       #converting the words to lower case

          self.__words_lists.append(column_i_words_list)                                              # take a record of such list of words

          var =[ self.__all_words_in_dataframe.append(i) for i in column_i_words_list]                       #append each words in column_i to "all_words_in_dataframe"
          del var

    self.__all_words_in_dataframe = list(set(self.__all_words_in_dataframe))                                          #ensuring that their is no word repetition

    print(self.__all_words_in_dataframe)


    #encoding the words in the datafarame. A word is made to have a particular code number irrespective of its location in the dataframe
    for column_name in  self.__columns:

            column_i_words_list = self.__dataframe[column_name]

            words_encode = list(map(lambda x: self.__all_words_in_dataframe.index(x)   , column_i_words_list))               #find the index of each word in "column_i_words_list" and result is a list

            self.__indexes_lists.append(words_encode)



    self.__indexes_lists = np.array(self.__indexes_lists)                       #converting the lists to array make suitable for ML models
    
    self.__indexes_lists=self.__indexes_lists.transpose()
    self.__words_lists = np.array(self.__words_lists)
   

    #since the words, and hence the resulting encodings, of a column were transformed to rows,
    #we need to reverse "self.__indexes_lists"  and "self.__words_lists" to a structure
    #that is same as the original dataframe. This is achieved by transposing
    self.__words_lists = self.__words_lists.transpose()
  

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

      self.__x_train, self.__x_test, self.__y_train, self.__y_test = train_test_split(self.__features, self.__label,test_size=0.8, random_state=1)

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

      self.__ann.fit(self.__x_train,self.__y_train, batch_size= size_of_data_per_training,epochs=400)



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



   #getting the encode value of the words.  If the word isn't present in the dataframe, it is assigned a value of zero

    for word in list_of_words:

          if word in self.__all_words_in_dataframe:

                 word_index = self.__all_words_in_dataframe.index(word)
                 encode_for_entire_sentence_1_D.append(word_index)
          else:
              encode_for_entire_sentence_1_D.append(0)
                                                                                                            # appending zero , as a result of absence of the word in the searched column, will decrease

      #after working on all words in the sentence, we need to utilize the encode as a 2D-array
    encode_for_entire_sentence_1_D = np.array(encode_for_entire_sentence_1_D)

    encode_for_entire_sentence_2_D .append(encode_for_entire_sentence_1_D)

    scaled_encoded_sentence = self.__sc_x.transform(encode_for_entire_sentence_2_D)                   #apply feature scaling using the standard deviation used on the training data

    predicted_completing_label = self.__ann.predict(scaled_encoded_sentence)            #where predicted completing sentence is the predicted label for the incomplete sentence


  # predicted_completing_label = predicted_completing_label.reshape(-1,1)

    predicted_completing_label = self.__sc_y.inverse_transform(predicted_completing_label)

    predicted_encode = predicted_completing_label[0][0]
    
    predicted_encode = round(predicted_encode)                                  #round-up the predicted encode to the nearest whole number

    label =self.__indexes_lists[:,-1]


    if predicted_encode in label:
          predicted_word = self.__all_words_in_dataframe[predicted_encode]
    else:

        closest_diff = min([ abs(predicted_encode - index)  for index in label])        # getting the difference between the predicted encode and all index in label,
                                                                                        #getting the minimum differnence
        predicted_encode += closest_diff                                                 #altering the value of the predicted encode

        predicted_word = self.__all_words_in_dataframe[predicted_encode]    
    
    #predicting the complete sentence

    print(f"{sentence} {predicted_word}")



