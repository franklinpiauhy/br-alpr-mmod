# -*- coding: utf-8 -*-

from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import utils as utils

TRAIN_DIR = 'D:\\UFPR-ALPR dataset\\training\\'
VALIDATION_DIR = 'D:\\UFPR-ALPR dataset\\validation\\'
TEST_DIR = 'D:\\UFPR-ALPR dataset\\testing\\'


print("reading and processing training data...")
X_let_train, y_let_train, X_dig_train, y_dig_train = utils.load_data(TRAIN_DIR)
print("...done!")
print("reading and processing validation data...")
X_let_val, y_let_val, X_dig_val, y_dig_val = utils.load_data(VALIDATION_DIR)
print("...done!")
print("reading and processing testing data...")
X_let_test, y_let_test, X_dig_test, y_dig_test = utils.load_data(TEST_DIR)
print("...done!")
print("")

# Digit training
digit_model = utils.LeNet5(10)

datagen = ImageDataGenerator()
   
datagen.fit(X_dig_train)

batch_size = 64
epochs = 1000

tbCallBack = TensorBoard(log_dir='./new_log_digit_otsu', histogram_freq=0, write_graph=True, write_images=True)
checkpoint = ModelCheckpoint("new_digit_otsu.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto')
early = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, verbose=1, mode='auto')

hist_digit = digit_model.fit_generator(datagen.flow(X_dig_train, y_dig_train, batch_size=batch_size), steps_per_epoch=len(X_dig_train) // batch_size, epochs=epochs, validation_data=(X_dig_val, y_dig_val), callbacks=[tbCallBack, checkpoint, reduce, early])
loss, acc = digit_model.evaluate(X_dig_test, y_dig_test, batch_size=batch_size)

print()
print('Test Loss = {}\nTest Accuracy = {}'.format(loss, acc))
#digit_model.save('digit.hdf5')

# Letter training
letter_model = utils.LeNet5(26)

batch_size = 64
epochs = 1000

tbCallBack = TensorBoard(log_dir='./new_log_letter_otsu', histogram_freq=0, write_graph=True, write_images=True)
checkpoint = ModelCheckpoint("new_letter_otsu.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, mode='auto')
early = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, verbose=1, mode='auto')

hist_letter = letter_model.fit(X_let_train, y_let_train, batch_size=batch_size, epochs=epochs, validation_data=(X_let_val, y_let_val), callbacks=[tbCallBack, checkpoint, reduce, early])
loss, acc = letter_model.evaluate(X_let_test, y_let_test, batch_size=batch_size)

print()
print('Test Loss = {}\nTest Accuracy = {}'.format(loss, acc))

#letter_model.save('letter.hdf5')

