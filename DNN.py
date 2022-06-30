#Installed python modules are pandas, numpy, matplotlib.pyplot, os, sklearn and tensorflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.utils import shuffle

#Initialize the neural network, otherwise it will overlap with the previous one
tf.compat.v1.reset_default_graph()
#set random seed
tf.random.set_seed(1234)

#Read genes that require deep learning as a starting features
gene_last = pd.read_table('mRNA_last.txt', sep = "\t", header = None)
gene_last = gene_last.iloc[:,0]
gene_last
#Read in the expression spectrum that was divided into two parts, tumor and normal
df_normal =pd.read_table('mRNA_CPTAC_tumor.txt', sep = "\t")
df_tumor=pd.read_table('mRNA_CPTAC_normal.txt', sep = "\t")

#Get the normla expression profile of the starting feature, and label the sample annotations to scramble
df1 = df_normal.loc[gene_last]
df1 = pd.DataFrame(df1.values.T, index=df1.columns, columns=df1.index)
df1['Exited'] = 0 #0 means normal label
#Get the tumor expression profile of the starting feature, and label the sample annotations to scramble
df2 = df_tumor.loc[gene_last]
df2 = pd.DataFrame(df2.values.T, index=df2.columns, columns=df2.index)
df2['Exited'] = 1 #1 means disease label

#The normal expression profile and disease expression profile are divided into training set and test set
train_normal, test_normal = train_test_split(df1, test_size=0.2, random_state=2)
train_tumor, test_tumor = train_test_split(df2, test_size=0.2, random_state=2)
#Combine training set and test set
train = pd.concat([train_normal, train_tumor])
test = pd.concat([test_normal, test_tumor])
#Shuffle the training set and test set
train = shuffle(train)    
test = shuffle(test)

#Get the scrambled training set labels and test set labels
label_train = train[['Exited']]
label_test = test[['Exited']]
#Remove labels from training set and test set
train = train[set(train.columns) ^ set(['Exited'])]
test = test[set(test.columns) ^ set(['Exited'])]

#Model initial neuron structure settings
model = tf.keras.Sequential()
#Add the first neuron layer, 400 neurons, the input layer dimension is the number of features
model.add(layers.Dense(400,input_dim=len(train.columns), activation = 'relu',))
#Here is the usage of regularization, read the paper to know the usage
model.add(layers.BatchNormalization())
#Similarly, add a second layer of neurons
model.add(layers.Dense(100, activation = 'relu'))
#model.add(layers.BatchNormalization())
#Similarly, add a third layer of neurons
model.add(layers.Dense(40, activation = 'relu'))
#Output layer, dimension is 1, activation function must be sigmod
model.add(layers.Dense(1,activation ='sigmoid'))

#Set the parameters of deep learning
#Refer to the paper for specific parameter explanations and settings
model.compile(loss = 'binary_crossentropy',
              optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001),
              metrics = [tf.keras.metrics.BinaryAccuracy()]
             )

#model fit
history = model.fit(
          train,
          label_train,
          validation_data = (test, label_test),
          epochs = 1000, 
          batch_size = 12)
result = pd.DataFrame(history.history)

#Draw the accuracy curve and the loss rate curve
#Refer to the evaluation criteria of deep learning in the discussion section of the paper, you need to get a good evaluation and then explain
#fixed parameters
plt.rcParams.update(plt.rcParamsDefault)
plt.figure(dpi = 80)
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.plot(history.history['binary_accuracy'])
plt.plot(history.history['val_binary_accuracy'])
print(np.mean(history.history['val_binary_accuracy']))
plt.legend(['train accuracy', 'valid accuracy'], loc='lower right')
plt.show()
plt.figure(dpi = 80)
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train loss', 'valid loss'], loc='upper right')
plt.show()

#The SHAP framework interprets the deep learning model, note that the computer needs better performance
import shap
shap.initjs()

#Build an interpreter with the training set
explainer = shap.KernelExplainer(model, train)
#Get SHAP value with test set
shap_values = explainer.shap_values(test)
#See the effect of a single gene on a sample
shap.force_plot(explainer.expected_value, shap_values[0], train)
#View the contribution rate of the overall gene max_display indicates the number of displays
shap.summary_plot(shap_values, train, max_display = 20)

#Algorithm for calculating contribution rate
which = lambda lst:list(np.where(lst)[0])
weight_df = pd.DataFrame(shap_values[0])
weight_df.columns = train.columns
for col in weight_df.columns:
    weight_df.loc[:,col] = weight_df[col].abs()
weight_df['Index'] = 0
weight_df = weight_df.groupby('Index').mean()
weight_df = pd.DataFrame(weight_df.values.T, index = weight_df.columns, columns = weight_df.index)
weight_df.columns = ['weight']
weight_values = weight_df['weight'].sort_values(ascending = False)
weight_genes  = weight_values.index
weight_values.to_csv('weight_values_CPTAC_mRNA.txt',sep='\t')
