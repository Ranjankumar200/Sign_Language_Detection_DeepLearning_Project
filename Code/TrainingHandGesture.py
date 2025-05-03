from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
import os

# Model Creation
model = Sequential()

# Convolutional Layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten and Fully Connected Layers
model.add(Flatten())
model.add(Dense(units=150, activation='relu', kernel_regularizer=l2(0.001)))
model.add(Dropout(0.25))
model.add(Dense(units=6, activation='softmax'))  # 6 classes: NONE, ONE, TWO, THREE, FOUR, FIVE

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data Augmentation for Training Data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=12,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.15,
    horizontal_flip=True
)

# Data Generator for Validation Data
val_datagen = ImageDataGenerator(rescale=1./255)

# Directory paths
train_dir = r'C:\Users\sachi\OneDrive\Desktop\AICTE_Microsoft_Internship3_Project\Sign_Language_Detection_DeepLearning_Project\HandGestureDataset\train'
val_dir = r'C:\Users\sachi\OneDrive\Desktop\AICTE_Microsoft_Internship3_Project\Sign_Language_Detection_DeepLearning_Project\HandGestureDataset\test'

# Training Data
training_set = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    color_mode='grayscale',
    batch_size=8,
    classes=['NONE', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE'],
    class_mode='categorical'
)

# Validation Data
val_set = val_datagen.flow_from_directory(
    val_dir,
    target_size=(256, 256),
    color_mode='grayscale',
    batch_size=8,
    classes=['NONE', 'ONE', 'TWO', 'THREE', 'FOUR', 'FIVE'],
    class_mode='categorical'
)

# Automatically calculate steps
steps_per_epoch = training_set.samples // training_set.batch_size
validation_steps = val_set.samples // val_set.batch_size

# Callbacks for EarlyStopping and Model Checkpoint
callback_list = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ModelCheckpoint(filepath="model.keras", monitor='val_loss', save_best_only=True, verbose=1)
]

# Fit the Model
model.fit(
    training_set,
    steps_per_epoch=steps_per_epoch,
    epochs=25,
    validation_data=val_set,
    validation_steps=validation_steps,
    callbacks=callback_list
)
