
import tensorflow as tf
import os
from tensorflow.contrib.learn.python.learn.datasets.mnist import extract_images, extract_labels
import time
import matplotlib.pyplot as plt
from google.colab import drive
import pandas as pd
import numpy as np


#### The following lines are used in google colab to save the data directly in google drive####
# drive.mount('/content/gdrive', force_remount=True)
# root_dir = "/content/gdrive/My Drive/"
# base_dir = root_dir + '542HW4/'
# path = '/content/gdrive/My Drive/542HW4/'
def load_mnist_data():
    path = '/content/gdrive/My Drive/542HW4/'
    with open((path+"train-images-idx3-ubyte.gz"), "rb") as f:
        train_allimages = extract_images(f)

    with open((path+"train-labels-idx1-ubyte.gz"), "rb") as f:
        train_alllabels = extract_labels(f)
    valid_set_size = 10000
    split = len(train_allimages) - valid_set_size
    valid_images = train_allimages[split:]  
    valid_labels = train_alllabels[split:]  
    train_images = train_allimages[:split]  
    train_labels = train_alllabels[:split]  

    with open((path+ "t10k-images-idx3-ubyte.gz"), "rb") as f:
        test_images = extract_images(f)

    with open((path+"t10k-labels-idx1-ubyte.gz"), "rb") as f:
        test_labels = extract_labels(f)

    return (train_images, train_labels), (valid_images, valid_labels),(test_images, test_labels)

(training_images, training_labels),(valid_images,valid_labels), (test_images, test_labels) = load_mnist_data()   

#print(len(training_images)) 
training_images=training_images.reshape(len(training_images), 28, 28, 1)
training_images=training_images / 255.0
valid_images = valid_images.reshape(len(valid_images), 28, 28, 1)
valid_images=valid_images/255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0

def train_mnist_conv(dense=128,dropout=0.6,activation='relu',eta=0.001,maps=64,ksize=7,batch=64):
    
    model = tf.keras.models.Sequential([
            
            tf.keras.layers.Conv2D(maps, (ksize), activation=activation, input_shape=(28, 28, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(maps, (ksize), activation=activation),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(maps, (ksize), activation=activation),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(maps, (ksize), activation=activation),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(dense, activation=activation),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout),

            tf.keras.layers.Dense(10, activation='softmax')
    ])
    opt = tf.keras.optimizers.Adam(learning_rate=eta)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    history = model.fit(
        training_images, training_labels, epochs=30,batch_size = batch, #callbacks=[callbacks]
        validation_data=(valid_images,valid_labels)
    )
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    prediction = model.predict(test_images)
    
    ####Following code is commented out. They are used for storing values in a text file and storing the prediction ####

    for i, d in enumerate(prediction):
      prediction[i] = prediction[i] // max(d)

      prediction[i] = np.array(y_pred, dtype=np.uint8)

      save = pd.DataFrame(y_pred, columns=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
      save.to_csv(path+'MNIST.csv', index=False, header=False)
    # f.write("\n --------------------new--------------------")
    # f.write("\n Denselayer: ")
    # f.write(str(dense))
    # f.write("\n Dropout: ")
    # f.write(str(dropout))
    # f.write("\n Activation: ")
    # f.write(str(activation))
    # f.write("\n Learning Rate: ")
    # f.write(str(eta))
    # f.write("\n Maps: ")
    # f.write(str(maps))
    # f.write("\n kernel size: ")
    # f.write(str(ksize))
    # f.write("\n Batch size: ")
    # f.write(str(Batch))
    
    # f.write("\n Training_Loss: ")
    # f.write(str(history.history['loss'][-1]))
    # f.write("\n Training_acc: ")
    # f.write(str(history.history['acc'][-1]))
    # f.write("\n Valid_Loss: ")
    # f.write(str(history.history['val_loss'][-1]))
    # f.write("\n Valid_acc: ")
    # f.write(str(history.history['val_acc'][-1]))

    # f.write("\n Training_Loss Complete: ")
    # f.write(str(history.history['loss']))
    # f.write("\n Training_acc Complete: ")
    # f.write(str(history.history['acc']))
    # f.write("\n Valid_Loss Complete: ")
    # f.write(str(history.history['val_loss']))
    # f.write("\n Valid_acc Complete: ")
    # f.write(str(history.history['val_acc']))
    
    # f.write("\n test_loss: ")
    # f.write(str(test_loss))
    # f.write("\n test_acc: ")
    # f.write(str(test_acc))

    return history.epoch, history.history['acc'], history.history['val_acc']

if __name__ == "__main__":
  train_mnist_conv(dense=128,dropout=0.6,activation='relu',eta=0.001,maps=64,ksize=7)


  
  #### The following code which has been commented out is used for plotting and saving data ####
  # filename = time.strftime("%Y%m%d-%H%M%S")
  # f = open(path+'datafor'+filename,"w+")  
  
  # DenseList = [16,32,64,128,256,512]
  # ActivationList = ['relu','sigmoid','tanh']
  # etaList = [0.01,0.001,0.003,0.007,0.0001,0.00001]
  # mapsList = [8,16,24,32,48,64,96]
  # ksizeList = [2,3,4,5,6,7]
  # DropoutList = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

  #plt = plt.figure(1)
  #plt = plt.figure(2)

  # for i in range(len(DenseList)):
  #     n,t,v = train_mnist_conv(dense=DenseList[i])
  #     #print('n',n,'t',t,'v',v)
  #     plt.figure(1)
  #     plt.plot(list(range(0,len(t))),t ,label=str(DenseList[i]))#, label=str(DenseList[i]))
  #     plt.figure(2)
  #     plt.plot(list(range(0,len(t))),v, label=str(DenseList[i]))#
  # plt.figure(1)

  # legend = plt.legend(loc='lower right', shadow=True)
  # plt.title("Training Accuracy: Dense Layer")
  # plt.xlabel('Epoch')
  # plt.ylabel('Accuracy')
  # plt.savefig(path+'Trainingaccuracydenselayer')
  # plt.clf()

  # plt.figure(2)

  # legend = plt.legend(loc='lower right', shadow=True)
  # plt.title("Validation Accuracy: Dense Layer")
  # plt.xlabel('Epoch')
  # plt.ylabel('Accuracy')
  # plt.savefig(path+'Validationaccuracydenselayer')
  # plt.clf()

  # for i in range(len(etaList)):
  #     n,t,v = train_mnist_conv(eta=etaList[i])
  #     #print(i,n,t,v)
  #     plt.figure(1)
  #     plt.plot(list(range(0,len(t))),t, label=str(etaList[i]))
  #     plt.figure(2)
  #     plt.plot(list(range(0,len(v))),v, label=str(etaList[i]))
  # plt.figure(1)
  # legend = plt.legend(loc='lower right', shadow=True)
  # plt.title("Training Accuracy: Learning Rate")
  # plt.xlabel('Epoch')
  # plt.ylabel('Accuracy')
  # plt.savefig(path+'TrainingaccuracyLearningRate')
  # plt.clf()
  # plt.figure(2)
  
  # legend = plt.legend(loc='lower right', shadow=True)
  # plt.title("Validation Accuracy: Learning Rate")
  # plt.xlabel('Epoch')
  # plt.ylabel('Accuracy')
  # plt.savefig(path+'Validationaccuracylearningrate')
  # plt.clf()

  # for i in range(len(mapsList)):
  #     n,t,v = train_mnist_conv(maps=mapsList[i])
  #     plt.figure(1)
  #     plt.plot(list(range(0,len(t))),t, label=str(mapsList[i]))
  #     plt.figure(2)

  #     plt.plot(list(range(0,len(v))),v, label=str(mapsList[i]))

  # plt.figure(1)
  # legend = plt.legend(loc='lower right', shadow=True)
  # plt.title("Training Accuracy: Feature Maps")
  # plt.xlabel('Epoch')
  # plt.ylabel('Accuracy')
  # plt.savefig(path+'Trainingaccuracymaps')
  # plt.clf()
  # plt.figure(2)
  # legend = plt.legend(loc='lower right', shadow=True)
  # plt.title("Validation Accuracy: Feature Maps")
  # plt.xlabel('Epoch')
  # plt.ylabel('Accuracy')
  # plt.savefig(path+'Validationaccuracymaps')
  # plt.clf()


  # for i in range(len(ksizeList)):
  #     n,t,v = train_mnist_conv(ksize=ksizeList[i])
  #     plt.figure(1)
  #     plt.plot(list(range(0,len(t))),t, label=str(ksizeList[i]))
  #     plt.figure(2)
  #     plt.plot(list(range(0,len(v))),v, label=str(ksizeList[i]))
  # plt.figure(1)
  # legend = plt.legend(loc='lower right', shadow=True)
  # plt.title("Training Accuracy: Kernel Size")
  # plt.xlabel('Epoch')
  # plt.ylabel('Accuracy')
  # plt.savefig(path+'Trainingaccuracyksize')
  # plt.clf()
  # plt.figure(2)
  # legend = plt.legend(loc='lower right', shadow=True)
  # plt.title("Validation Accuracy: Kernel Size")
  # plt.xlabel('Epoch')
  # plt.ylabel('Accuracy')
  # plt.savefig(path+'Validationaccuracyksize')
  # plt.clf()

  # for i in range(len(DropoutList)):
  #     n,t,v = train_mnist_conv(dropout=DropoutList[i])
  #     plt.figure(1)
  #     plt.plot(list(range(0,len(t))),t, label=str(DropoutList[i]))
  #     plt.figure(2)
  #     plt.plot(list(range(0,len(v))),v, label=str(DropoutList[i]))
  # plt.figure(1)
  # legend = plt.legend(loc='lower right', shadow=True)
  # plt.title("Training Accuracy: Dropout")
  # plt.xlabel('Epoch')
  # plt.ylabel('Accuracy')
  # plt.savefig(path+'Trainingaccuracydropout')
  # plt.clf()
  # plt.figure(2)
  # legend = plt.legend(loc='lower right', shadow=True)
  # plt.title("Validation Accuracy: Dropout")
  # plt.xlabel('Epoch')
  # plt.ylabel('Accuracy')
  # plt.savefig(path+'Validationaccuracydropout')
  # plt.clf()
  #LayerList = [train_mnist_conv1,train_mnist_conv2,train_mnist_conv3,train_mnist_conv4,train_mnist_conv5,train_mnist_conv6]
  # LayerList = [train_mnist_conv3]
  # #train_mnist_conv4()
  # ActivationList = ['relu','sigmoid','tanh']
  # for i in range(len(ActivationList)):
  #   n,t,v = train_mnist_conv4(activation=ActivationList[i])
  #   plt.figure(1)
  #   plt.plot(list(range(0,len(t))),t, label=str(ActivationList[i]))
  #   plt.figure(2)
  #   plt.plot(list(range(0,len(v))),v, label=str(ActivationList[i]))
  # plt.figure(1)
  # legend = plt.legend(loc='lower right', shadow=True)
  # plt.title("Training Accuracy: Activation Function")
  # plt.xlabel('Epoch')
  # plt.ylabel('Accuracy')
  # plt.savefig(path+'Trainingaccuracyactivation')
  # plt.clf()
  # plt.figure(2)
  # legend = plt.legend(loc='lower right', shadow=True)
  # plt.title("Validation Accuracy: Activation Function")
  # plt.xlabel('Epoch')
  # plt.ylabel('Accuracy')
  # plt.savefig(path+'Validationaccuracyactivation')
  # plt.clf()

  #train_mnist_conv4(activation='sigmoid')
  #train_mnist_conv4(activation='tanh')
  
#   for i in range(len(LayerList)):
#     fa = LayerList[i]
#     n,t,v = fa()
#     plt.figure(1)
#     plt.plot(list(range(0,len(t))),t, label=str(i+1))
#     plt.figure(2)
#     plt.plot(list(range(0,len(v))),v, label=str(i+1))
#   plt.figure(1)
#   legend = plt.legend(loc='lower right', shadow=True)
#   plt.title("Training Accuracy: CNN Layers")
#   plt.xlabel('Epoch')
#   plt.ylabel('Accuracy')
#   plt.savefig(path+'TrainingaccuracyLayers')
#   plt.clf()
#   plt.figure(2)
#   legend = plt.legend(loc='lower right', shadow=True)
#   plt.title("Validation Accuracy: CNN Layers")
#   plt.xlabel('Epoch')
#   plt.ylabel('Accuracy')
#   plt.savefig(path+'ValidationaccuracyLayers')
#   plt.clf()
#   f.close()