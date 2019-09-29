from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras import optimizers
import h5py
import os

def make_model():
        files = os.listdir()
        if "Trained_model.h5" in files:
                return
        classifier = Sequential()
        classifier.add(Convolution2D(32, 3,  3, input_shape = (64, 64, 3), activation = 'relu'))

        classifier.add(MaxPooling2D(pool_size =(2,2)))

        classifier.add(Convolution2D(64, 3,  3, activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size =(2,2)))

        classifier.add(Flatten())

        classifier.add(Dense(256, activation = 'relu'))
        classifier.add(Dropout(0.5))
        classifier.add(Dense(7, activation = 'softmax'))

        classifier.compile(
                optimizer = optimizers.SGD(lr = 0.01),
                loss = 'categorical_crossentropy',
                metrics = ['accuracy'])

        from keras.preprocessing.image import ImageDataGenerator
        train_datagen = ImageDataGenerator(rescale=1./255)

        test_datagen = ImageDataGenerator(rescale=1./255)

        training_set = train_datagen.flow_from_directory(
                'data/training_set',
                target_size=(64, 64),
                batch_size=32,
                class_mode='categorical')

        test_set = test_datagen.flow_from_directory(
                'data/test_set',
                target_size=(64, 64),
                batch_size=32,
                class_mode='categorical')

        model = classifier.fit_generator(
                training_set,
                steps_per_epoch=100,
                epochs=5,
                validation_data = test_set,
                validation_steps = 100
        )
        classifier.save('Trained_model.h5')

make_model()