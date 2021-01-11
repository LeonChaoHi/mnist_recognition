from keras import Model, Sequential, optimizers
from keras.layers import Input
from keras.layers import Conv2D, MaxPool2D, MaxPooling2D, Dense, Flatten, Dropout, Lambda

def 


def VGG16(input_shape=(224, 224, 3), nclass=8):
    input_ = Input(shape=input_shape)
    # block1
    x = Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv1")(input_)
    x = Conv2D(64, (3, 3), activation="relu", padding="same", name="block1_conv2")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block1_pool")(x)
    # block2
    x = Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv1")(x)
    x = Conv2D(128, (3, 3), activation="relu", padding="same", name="block2_conv2")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block2_pool")(x)
    # block3
    x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv1")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv2")(x)
    x = Conv2D(256, (3, 3), activation="relu", padding="same", name="block3_conv3")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block3_pool")(x)
    # block4
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv1")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv2")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block4_conv3")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block4_pool")(x)
    # block5
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv1")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv2")(x)
    x = Conv2D(512, (3, 3), activation="relu", padding="same", name="block5_conv3")(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name="block5_pool")(x)
    # Classification block
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dropout(0.5)(x)
    output_ = Dense(nclass, activation='softmax', name='fc3')(x)
    model = Model(inputs=input_, outputs=output_)
    model.summary()
    opti_sgd = optimizers.sgd(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=opti_sgd, metrics=['accuracy'])
    return model
