import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import EfficientNetB0, DenseNet121
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
base_path = '' #file directory
train_dir = os.path.join(base_path, 'train')
test_path = os.path.join(base_path, 'test')
IMAGE_SIZE = [224, 224]
BATCH_SIZE = 32
INITIAL_EPOCHS = 5
FINE_TUNE_EPOCHS = 10
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS
def create_data_generators(preprocessing_function):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_function,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )
    training_set = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='training'
    )
    validation_set = train_datagen.flow_from_directory(
        train_dir,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        subset='validation'
    )
    return training_set, validation_set
def train_model(model_name, base_model, training_set, validation_set, class_weights_dict):
    print(f"\n--- Building Model: {model_name} ---")
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    prediction = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=base_model.input, outputs=prediction)
    '''Initial Training'''
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
    print(f"\n--- Starting Initial Training for {model_name} ---")
    history = model.fit(
        training_set, validation_data=validation_set, epochs=INITIAL_EPOCHS,
        class_weight=class_weights_dict, steps_per_epoch=len(training_set), validation_steps=len(validation_set)
    )
    '''Fine-Tuning'''
    base_model.trainable = True
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.00001), metrics=['accuracy'])
    print(f"\n--- Starting Fine-Tuning for {model_name} ---")
    model.fit(
        training_set, validation_data=validation_set, epochs=TOTAL_EPOCHS, initial_epoch=history.epoch[-1],
        class_weight=class_weights_dict, steps_per_epoch=len(training_set), validation_steps=len(validation_set)
    )
    model_save_path = f'pneumonia_model_{model_name}.h5'
    model.save(model_save_path)
    print(f"--- {model_name} Model Training Complete. Saved to {model_save_path} ---\n")
    return model_save_path


'''PART A: TRAIN EFFICIENTNETB0'''
print("="*50)
print("PART A: TRAINING EFFICIENTNETB0 MODEL")
print("="*50)
effnet_train_gen, effnet_val_gen = create_data_generators(efficientnet_preprocess)
class_labels = np.unique(effnet_train_gen.classes)
class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=effnet_train_gen.classes)
class_weights_dict = dict(zip(class_labels, class_weights))
effnet_base = EfficientNetB0(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
effnet_model_path = train_model('EfficientNetB0', effnet_base, effnet_train_gen, effnet_val_gen, class_weights_dict)

'''PART B: TRAIN DENSENET121'''
print("="*50)
print("PART B: TRAINING DENSENET121 MODEL")
print("="*50)
densenet_train_gen, densenet_val_gen = create_data_generators(densenet_preprocess)
densenet_base = DenseNet121(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
densenet_model_path = train_model('DenseNet121', densenet_base, densenet_train_gen, densenet_val_gen, class_weights_dict)

'''PART C: EVALUATE ENSEMBLE MODEL'''
print("="*50)
print("PART C: EVALUATING ENSEMBLE MODEL")
print("="*50)
print("Loading trained models")
model1 = load_model(effnet_model_path)
model2 = load_model(densenet_model_path)

'''Creating 2 separate test generators with the correct preprocessing for each model'''
test_datagen1 = ImageDataGenerator(preprocessing_function=efficientnet_preprocess)
test_set1 = test_datagen1.flow_from_directory(
    test_path, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=False)
test_datagen2 = ImageDataGenerator(preprocessing_function=densenet_preprocess)
test_set2 = test_datagen2.flow_from_directory(
    test_path, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='binary', shuffle=False)
'''Get predictions from both models'''
print("Making predictions with Model 1 (EfficientNetB0)")
preds1 = model1.predict(test_set1, steps=len(test_set1))
print("Making predictions with Model 2 (DenseNet121)")
preds2 = model2.predict(test_set2, steps=len(test_set2))
print("Averaging predictions for ensemble result")
ensemble_preds = (preds1 + preds2) / 2
ensemble_preds_binary = np.round(ensemble_preds)
true_labels = test_set1.classes
from sklearn.metrics import accuracy_score
ensemble_accuracy = accuracy_score(true_labels, ensemble_preds_binary)
print("\n FINAL EVALUATION")
print(f"Final Ensemble Test Accuracy: {ensemble_accuracy * 100:.2f}%")