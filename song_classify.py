import librosa
import librosa.feature
import librosa.display
import glob
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Ham trich xuat MFCC cua bai hat
def get_mfcc(f):
    y, _ = librosa.load(f)

    # Trich xua MFCC
    my_features = librosa.feature.mfcc(y)
    # Normalize gia tri MFCC
    my_features /= np.amax(np.absolute(my_features))

    # Flattern MFCC
    return np.ndarray.flatten(my_features)[:25000]

# Load du lieu
def load_data():
    mfcc_list = []
    label_list = []
    # Dinh nghia cac class
    song_type = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

    # Load cac file tu folder
    for curr_type in song_type:
        sound_files = glob.glob('data/'+curr_type+'/*.wav')
        for f in sound_files:
            # Trich xuat MFCC
            features = get_mfcc(f)
            mfcc_list.append(features)
            # Them label
            label_list.append(curr_type)

    # convert labels to one-hot encoding
    label_uniq_ids, label_row_ids = np.unique(label_list, return_inverse=True)
    label_row_ids = label_row_ids.astype(np.int32, copy=False)
    onehot_labels = to_categorical(label_row_ids, len(label_uniq_ids))
    return np.stack(mfcc_list), onehot_labels

# Load du lieu
features, labels = load_data()

# Tham so chia train/test
training_split = 0.8

# Noi labels vao feature
alldata = np.column_stack((features, labels))

# Random xao tron data
np.random.shuffle(alldata)

# Thuc hien chia train/test
splitidx = int(len(alldata) * training_split)
train, test = alldata[:splitidx,:], alldata[splitidx:,:]


train_input = train[:,:-10]
train_labels = train[:,-10:]

test_input = test[:,:-10]
test_labels = test[:,-10:]


#Dinh nghia model
model = Sequential([
    Dense(1024, input_dim=np.shape(train_input)[1]),
    Dense(1024, input_dim=1024),
    Dropout(0.5),
    Activation('relu'),
    Dense(10),
    Activation('softmax'),
    ])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

file_path = "best_model_{epoch:08d}_{val_loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath=file_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
callbacks_list = [checkpoint]

# Tien hanh train
model.fit(train_input, train_labels, epochs=100, batch_size=32,
          validation_split=0.2, callbacks = callbacks_list)

# Tien hanh eval tren tap test
loss, acc = model.evaluate(test_input, test_labels, batch_size=32)

print("Done!")
print("Loss: %.4f, accuracy: %.4f" % (loss, acc))
