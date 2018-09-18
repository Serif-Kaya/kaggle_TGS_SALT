version = 20

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.layers import Input, Lambda, Conv2D, SpatialDropout2D, BatchNormalization,Activation
from keras.layers import MaxPooling2D, Conv2DTranspose, concatenate
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.models import Model, load_model, model_from_json
from keras.optimizers import Adam, SGD
import keras.backend as K
from keras import losses
import tensorflow as tf
from skimage.transform import resize
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from sklearn.model_selection import train_test_split
import gc


# Alternative resizing method by padding to (128,128,1) and reflecting

def pad_reflect(img, north=0, south=0, west=0, east=0):
    h = img.shape[0]
    w = img.shape[1]
    new_image = np.zeros((north+h+south,west+w+east))
    
    # Place the image in new_image
    new_image[north:north+h, west:west+w] = img
    
    new_image[north:north+h,0:west] = np.fliplr(img[:,:west])
    new_image[north:north+h,west+w:] = np.fliplr(img[:,w-east:])
    
    new_image[0:north,:] = np.flipud(new_image[north:2*north,:])
    new_image[north+h:,:] = np.flipud(new_image[north+h-south:north+h,:]) 
    
    return new_image.reshape(128,128,1)

def unpad_reflect(img, north=0, west=0, height=0, width=0):
    return img[north:north+height,west:west+width,:]
def castF(x):
    return K.cast(x, K.floatx())

def castB(x):
    return K.cast(x, bool)

def iou_loss_core(y_true, y_pred, smooth=1):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    union = K.sum(y_true,-1) + K.sum(y_pred,-1) - intersection
    iou = (intersection + smooth) / ( union + smooth)
    return iou


def iou_loss(y_true, y_pred):
    return 1 - iou_loss_core(y_true, y_pred)

def iou_bce_loss(y_true, y_pred):
    return losses.binary_crossentropy(y_true, y_pred) + 5 * iou_loss(y_true, y_pred)

def competition_metric(true, pred): #any shape can go

    tresholds = [0.5 + (i*.05)  for i in range(10)]

    #flattened images (batch, pixels)
    true = K.batch_flatten(true)
    pred = K.batch_flatten(pred)
    pred = castF(K.greater(pred, 0.5))

    #total white pixels - (batch,)
    trueSum = K.sum(true, axis=-1)
    predSum = K.sum(pred, axis=-1)

    #has mask or not per image - (batch,)
    true1 = castF(K.greater(trueSum, 1))    
    pred1 = castF(K.greater(predSum, 1))

    #to get images that have mask in both true and pred
    truePositiveMask = castB(true1 * pred1)

    #separating only the possible true positives to check iou
    testTrue = tf.boolean_mask(true, truePositiveMask)
    testPred = tf.boolean_mask(pred, truePositiveMask)

    #getting iou and threshold comparisons
    iou = iou_loss_core(testTrue,testPred) 
    truePositives = [castF(K.greater(iou, tres)) for tres in tresholds]

    #mean of thressholds for true positives and total sum
    truePositives = K.mean(K.stack(truePositives, axis=-1), axis=-1)
    truePositives = K.sum(truePositives)

    #to get images that don't have mask in both true and pred
    trueNegatives = (1-true1) * (1 - pred1) # = 1 -true1 - pred1 + true1*pred1
    trueNegatives = K.sum(trueNegatives) 

    return (truePositives + trueNegatives) / castF(K.shape(true)[0])


# src: https://www.kaggle.com/aglotero/another-iou-metric
def iou_metric(y_true_in, y_pred_in, print_table=False):
    labels = y_true_in
    y_pred = y_pred_in
    
    true_objects = 2
    pred_objects = 2

    intersection = np.histogram2d(labels.flatten(), y_pred.flatten(), bins=(true_objects, pred_objects))[0]

    # Compute areas (needed for finding the union between all objects)
    area_true = np.histogram(labels, bins = true_objects)[0]
    area_pred = np.histogram(y_pred, bins = pred_objects)[0]
    area_true = np.expand_dims(area_true, -1)
    area_pred = np.expand_dims(area_pred, 0)

    # Compute union
    union = area_true + area_pred - intersection

    # Exclude background from the analysis
    intersection = intersection[1:,1:]
    union = union[1:,1:]
    union[union == 0] = 1e-9

    # Compute the intersection over union
    iou = intersection / union

    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    if print_table:
        print("Thresh\tTP\tFP\tFN\tPrec.")
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        if print_table:
            print("{:1.3f}\t{}\t{}\t{}\t{:1.3f}".format(t, tp, fp, fn, p))
        prec.append(p)
    
    if print_table:
        print("AP\t-\t-\t-\t{:1.3f}".format(np.mean(prec)))
    return np.mean(prec)


    # Precision helper function
    def precision_at(threshold, iou):
        matches = iou > threshold
        true_positives = np.sum(matches, axis=1) == 1   # Correct objects
        false_positives = np.sum(matches, axis=0) == 0  # Missed objects
        false_negatives = np.sum(matches, axis=1) == 0  # Extra objects
        tp, fp, fn = np.sum(true_positives), np.sum(false_positives), np.sum(false_negatives)
        return tp, fp, fn

    # Loop over IoU thresholds
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        tp, fp, fn = precision_at(t, iou)
        if (tp + fp + fn) > 0:
            p = tp / (tp + fp + fn)
        else:
            p = 0
        prec.append(p)
    return np.mean(prec)

def iou_metric_batch(y_true_in, y_pred_in):
    batch_size = y_true_in.shape[0]
    metric = []
    for batch in range(batch_size):
        value = iou_metric(y_true_in[batch], y_pred_in[batch])
        metric.append(value)
    return np.mean(metric)



#NETWORK ARCHITECTURE
def conv_block(neurons, block_input, bn=False, dropout=None):
    conv1 = Conv2D(neurons, (3,3), padding='same', kernel_initializer='he_normal')(block_input)
    if bn:
        conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    if dropout is not None:
        conv1 = SpatialDropout2D(dropout)(conv1)
    conv2 = Conv2D(neurons, (3,3), padding='same', kernel_initializer='he_normal')(conv1)
    if bn:
        conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    if dropout is not None:
        conv2 = SpatialDropout2D(dropout)(conv2)
    pool = MaxPooling2D((2,2))(conv2)
    return pool, conv2  # returns the block output and the shortcut to use in the uppooling blocks

def middle_block(neurons, block_input, bn=False, dropout=None):
    conv1 = Conv2D(neurons, (3,3), padding='same', kernel_initializer='he_normal')(block_input)
    if bn:
        conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)
    if dropout is not None:
        conv1 = SpatialDropout2D(dropout)(conv1)
    conv2 = Conv2D(neurons, (3,3), padding='same', kernel_initializer='he_normal')(conv1)
    if bn:
        conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)
    if dropout is not None:
        conv2 = SpatialDropout2D(dropout)(conv2)
    
    return conv2

def deconv_block(neurons, block_input, shortcut, bn=False, dropout=None):
    deconv = Conv2DTranspose(neurons, (3, 3), strides=(2, 2), padding="same")(block_input)
    uconv = concatenate([deconv, shortcut])
    uconv = Conv2D(neurons, (3, 3), padding="same", kernel_initializer='he_normal')(uconv)
    if bn:
        uconv = BatchNormalization()(uconv)
    uconv = Activation('relu')(uconv)
    if dropout is not None:
        uconv = SpatialDropout2D(dropout)(uconv)
    uconv = Conv2D(neurons, (3, 3), padding="same", kernel_initializer='he_normal')(uconv)
    if bn:
        uconv = BatchNormalization()(uconv)
    uconv = Activation('relu')(uconv)
    if dropout is not None:
        uconv = SpatialDropout2D(dropout)(uconv)
        
    return uconv
    
def build_model(start_neurons, bn=False, dropout=None):
    
    input_layer = Input((128, 128, 1))
    
    # 128 -> 64
    conv1, shortcut1 = conv_block(start_neurons, input_layer, bn, dropout)

    # 64 -> 32
    conv2, shortcut2 = conv_block(start_neurons * 2, conv1, bn, dropout)
    
    # 32 -> 16
    conv3, shortcut3 = conv_block(start_neurons * 4, conv2, bn, dropout)
    
    # 16 -> 8
    conv4, shortcut4 = conv_block(start_neurons * 8, conv3, bn, dropout)
    
    # 8 -> 4
    conv5, shortcut5 = conv_block(start_neurons * 16, conv4, bn, dropout)
    
    # Middle
    convm = middle_block(start_neurons * 32, conv5, bn, dropout)
    
    # 4 -> 8
    deconv5 = deconv_block(start_neurons * 16, convm, shortcut5, bn, dropout=False)
    
    # 8 -> 16
    deconv4 = deconv_block(start_neurons * 8, deconv5, shortcut4, bn, dropout=False)
    
    # 16 -> 32
    deconv3 = deconv_block(start_neurons * 4, deconv4, shortcut3, bn, dropout=False)
    
    # 32 -> 64
    deconv2 = deconv_block(start_neurons * 2, deconv3, shortcut2, bn, dropout=False)
    
    # 64 -> 128
    deconv1 = deconv_block(start_neurons, deconv2, shortcut1, bn, dropout=False)
    
    #uconv1 = Dropout(0.5)(uconv1)
    output_layer = Conv2D(1, (1,1), padding="same", activation="sigmoid")(deconv1)
    
    model = Model(input_layer, output_layer)
    return model

#Data loading and preprocessing

%%time
DATA_DIR = '../input/'
train = pd.read_hdf(DATA_DIR + 'tgs_salt.h5', key='train')
train.head()

#Resizing the images
#For this I pad the images with 14 or 13 pixels, and reflect the appropriate part of the image to fill these pads.

images_resized = [pad_reflect(img, 14, 13, 14, 13) for img in train.images]
masks_resized = [pad_reflect(mask, 14, 13, 14, 13) for mask in train.masks]
X = np.stack(images_resized) / 255
y = np.stack(masks_resized) / 255
X.shape


#Basic data augmentation
#I'll try basic data augmentation by just flipping the images horizontally.

X_aug = np.concatenate((X, [np.fliplr(img) for img in X]), axis=0)
y_aug = np.concatenate((y, [np.fliplr(img) for img in y]), axis=0)


# Split the train data into actual train data and validation data
# train_test_split already shuffles data by default, so no need to do it

X_train, X_val, y_train, y_val = train_test_split(X_aug, y_aug, test_size=0.25, random_state=42)


# Build and save model in JSON format

json_filename = 'unet_salt_{}.json'.format(version)

model = build_model(start_neurons=8, bn=True, dropout=0.2)
model_json = model.to_json()
with open(json_filename, 'w') as json_file:
    json_file.write(model_json)
    

#Loading previous weights (optional)
#If we want our model to load weights from a previous training (best model), 
#set load_previous_weights = True and execute the next cell.
    
load_previous_weights = False
if (load_previous_weights):
    # Restore the model 
    weights_filename = 'unet_salt_weights_{}.h5'.format(version)
    model.load_weights(weights_filename)

early_stopping = EarlyStopping(patience=10, monitor='val_loss', mode='min', verbose=1)
weights_filename = 'unet_salt_weights_{}.h5'.format(version)
checkpoint = ModelCheckpoint(weights_filename, monitor='val_competition_metric', verbose=0, save_best_only=True, save_weights_only=True, mode='max')
optimizer = SGD(lr=0.1, momentum=0.8, decay=0.001, nesterov=False)


model.compile(optimizer=optimizer, loss=iou_bce_loss, metrics=['accuracy', competition_metric])
history = model.fit(X_train, y_train, batch_size=16, validation_data = [X_val, y_val], 
                    epochs=100, callbacks=[checkpoint])


# Restore the best model's weight
weights_filename = 'unet_salt_weights_{}.h5'.format(version)
model.load_weights(weights_filename)


# Let's see how the model performs (last model after training, not the saved best one)

# On the train set
print('*** Last model on train set ***')
model.evaluate(X_train, y_train)

# On the validation set
print('*** Last model on val set ***')  
model.evaluate(X_val, y_val)


print(model.metrics_names)

hist = history.history

figure, ax = plt.subplots(1,3, figsize=(18,6))
print(ax.shape)

def plot_history(history, metric, title, ax):
    ax.plot(history[metric])
    ax.plot(history['val_' + metric])
    ax.grid(True)
    ax.set_title(title)
    ax.set_ylabel(metric)
    ax.set_xlabel('epoch')
    ax.legend(['train', 'validation'], loc='upper left')                
    
plot_history(hist, 'loss', 'LOSS', ax[0])
plot_history(hist, 'acc', 'ACCURACY', ax[1])
plot_history(hist, 'competition_metric', 'COMPETITION METRIC', ax[2])


# Save image for reports
plt.savefig('history_unet_salt_{}.png'.format(version))



#Finding the optimal threshold for predictions
thresholds = np.linspace(0, 1, 50)
y_val_pred = model.predict(X_val)
ious = np.array([iou_metric_batch(y_val, np.uint8(y_val_pred > threshold)) for threshold in tqdm_notebook(thresholds)])


best_th_index = ious.argmax()
best_th = thresholds[best_th_index]
best_iou = ious[best_th_index]


plt.plot(thresholds, ious)
plt.plot(best_th, best_iou, "xr", label="Best threshold")
plt.xlabel("Threshold")
plt.ylabel("IoU")
plt.title("Threshold vs IoU ({}, {})".format(best_th, best_iou))
plt.legend()
# Save image for reports
plt.savefig('threshold_selection_{}.png'.format(version))

