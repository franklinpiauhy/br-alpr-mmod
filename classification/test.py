# -*- coding: utf-8 -*-
import numpy as np
from keras.models import load_model
from sklearn.metrics import classification_report
import utils
import string

TRAIN_DIR = 'D:\\UFPR-ALPR dataset\\training\\'
VALIDATION_DIR = 'D:\\UFPR-ALPR dataset\\validation\\'
TEST_DIR = 'D:\\UFPR-ALPR dataset\\testing\\'

print("loading digit model...")
digit = load_model('new_digit_otsu.61-0.11.hdf5')
print("done!")


print("loading digit model...")
letter = load_model('new_letter_otsu.28-0.75.hdf5')
print("done!")

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

#########################################################
####################### DÍGITOS #########################
#########################################################

print("Digit results\n")

####################### TREINO ##########################

pred_train = digit.predict(X_dig_train, batch_size = 128, verbose = 0)

pred_train = np.argmax(pred_train, axis=1)
y_train = np.argmax(y_dig_train, axis=1)
loss, acc = digit.evaluate(X_dig_train, y_dig_train, batch_size=128, verbose = 0)

print(classification_report(y_train, pred_train, target_names=list(string.digits)))
print()
print('Train Loss = {}\nTrain Accuracy = {}'.format(loss, acc))

######################## VALIDAÇÃO #########################

pred_val = digit.predict(X_dig_val, batch_size = 128, verbose = 0)

pred_val = np.argmax(pred_val, axis=1)
y_val = np.argmax(y_dig_val, axis=1)
loss, acc = digit.evaluate(X_dig_val, y_dig_val, batch_size=128, verbose = 0)

print(classification_report(y_val, pred_val, target_names=list(string.digits)))
print()
print('Validation Loss = {}\nValidation Accuracy = {}'.format(loss, acc))

######################## TESTE #############################

pred_test = digit.predict(X_dig_test, batch_size = 128, verbose = 0)

pred_test = np.argmax(pred_test, axis=1)
y_test = np.argmax(y_dig_test, axis=1)
loss, acc = digit.evaluate(X_dig_test, y_dig_test, batch_size=128, verbose = 0)

print(classification_report(y_test, pred_test, target_names=list(string.digits)))
print()
print('Test Loss = {}\nTest Accuracy = {}'.format(loss, acc))

print()
print()
##############################################################
######################## LETRAS ##############################
##############################################################

print("Letter results \n")

####################### TREINO ##########################

pred_train = letter.predict(X_let_train, batch_size = 128, verbose = 0)

pred_train = np.argmax(pred_train, axis=1)
y_train = np.argmax(y_let_train, axis=1)
loss, acc = letter.evaluate(X_let_train, y_let_train, batch_size=128, verbose = 0)


print(classification_report(y_train, pred_train, target_names=list(string.ascii_uppercase)))
print()
print('Training Loss = {}\nTraining Accuracy = {}'.format(loss, acc))

######################## VALIDAÇÃO #########################

pred_val = letter.predict(X_let_val, batch_size = 128, verbose = 0)

pred_val = np.argmax(pred_val, axis=1)
y_val = np.argmax(y_let_val, axis=1)
loss, acc = letter.evaluate(X_let_val, y_let_val, batch_size=128, verbose = 0)

print(classification_report(y_val, pred_val, target_names=list(string.ascii_uppercase)))
print()
print('Validation Loss = {}\nValidation Accuracy = {}'.format(loss, acc))

######################## TESTE #############################

pred_test = letter.predict(X_let_test, batch_size = 128, verbose = 0)

pred_test = np.argmax(pred_test, axis=1)
y_test = np.argmax(y_let_test, axis=1)
loss, acc = letter.evaluate(X_let_test, y_let_test, batch_size=128, verbose = 0)

print(classification_report(y_test, pred_test, target_names=list(string.ascii_uppercase)))
print()
print('Test Loss = {}\nTest Accuracy = {}'.format(loss, acc))