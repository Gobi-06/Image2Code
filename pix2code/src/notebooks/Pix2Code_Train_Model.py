#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import hashlib
import shutil
import sys

import numpy as np
import cv2

IMAGE_SIZE = 256

platform_list = ["android", "ios","web"]


BASE_DIR = "/Users/abhjha8/Image2Code/pix2code/src/dataset/{0}/all_data"
TRAINING_SET_NAME = "training_set"
EVALUATION_SET_NAME = "eval_set"
distribution = 6 # to decide the ratio of training and evaluation samples size


input_path_list = []

for platform in platform_list:
    input_path = BASE_DIR.format(platform)
    input_path_list.append(input_path)


# # PREPARE THE DATASET

# In[2]:


for input_path in input_path_list:
    paths = []

    for f in os.listdir(input_path):
        if f.find(".gui") != -1:
            path_gui = "{}/{}".format(input_path, f)
            file_name = f[:f.find(".gui")]

            if os.path.isfile("{}/{}.png".format(input_path, file_name)):
                path_img = "{}/{}.png".format(input_path, file_name)
                paths.append(file_name)

    evaluation_samples_number = len(paths) / (distribution + 1)
    training_samples_number = evaluation_samples_number * distribution

    assert training_samples_number + evaluation_samples_number == len(paths)

    print("Splitting datasets, training samples: {}, evaluation samples: {}".format(training_samples_number, evaluation_samples_number))

    np.random.shuffle(paths)

    eval_set = []
    train_set = []
    hashes = []
    for path in paths:
        with open("{}/{}.gui".format(input_path, path), 'r', encoding='utf-8') as f:
            chars = ""
            for line in f:
                chars += line
            content_hash = chars.replace(" ", "").replace("\n", "")
            content_hash = hashlib.sha256(content_hash.encode('utf-8')).hexdigest()

            if len(eval_set) == evaluation_samples_number:
                train_set.append(path)
            else:
                is_unique = True
                for h in hashes:
                    if h is content_hash:
                        is_unique = False
                        break

                if is_unique:
                    eval_set.append(path)
                else:
                    train_set.append(path)

            hashes.append(content_hash)

    assert len(eval_set) == evaluation_samples_number
    assert len(train_set) == training_samples_number

    if not os.path.exists("{}/{}".format(os.path.dirname(input_path), EVALUATION_SET_NAME)):
        os.makedirs("{}/{}".format(os.path.dirname(input_path), EVALUATION_SET_NAME))

    if not os.path.exists("{}/{}".format(os.path.dirname(input_path), TRAINING_SET_NAME)):
        os.makedirs("{}/{}".format(os.path.dirname(input_path), TRAINING_SET_NAME))

    for path in eval_set:
        shutil.copyfile("{}/{}.png".format(input_path, path), "{}/{}/{}.png".format(os.path.dirname(input_path), EVALUATION_SET_NAME, path))
        shutil.copyfile("{}/{}.gui".format(input_path, path), "{}/{}/{}.gui".format(os.path.dirname(input_path), EVALUATION_SET_NAME, path))

    for path in train_set:
        shutil.copyfile("{}/{}.png".format(input_path, path), "{}/{}/{}.png".format(os.path.dirname(input_path), TRAINING_SET_NAME, path))
        shutil.copyfile("{}/{}.gui".format(input_path, path), "{}/{}/{}.gui".format(os.path.dirname(input_path), TRAINING_SET_NAME, path))

    print("Training dataset: {}/training_set".format(os.path.dirname(input_path), path))
    print("Evaluation dataset: {}/eval_set".format(os.path.dirname(input_path), path))


# In[3]:


def get_preprocessed_img(img_path, image_size):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (image_size, image_size))
    img = img.astype('float32')
    img /= 255
    return img


TRAIN_IMAGE_DIR = "/Users/abhjha8/Image2Code/pix2code/src/dataset/{0}/training_set"
TRAIN_FEATURES_DIR = "/Users/abhjha8/Image2Code/pix2code/src/dataset/{0}/training_features"
input_path_list = []
output_path_mapping = dict()
for platform in platform_list:
    input_path = TRAIN_IMAGE_DIR.format(platform)
    output_path_mapping[input_path] = TRAIN_FEATURES_DIR.format(platform)
    if not os.path.exists(output_path_mapping[input_path]):
        os.makedirs(output_path_mapping[input_path])
    input_path_list.append(input_path)


print("Converting images to numpy arrays...")
for input_path in input_path_list:
    output_path = output_path_mapping[input_path]
    for f in os.listdir(input_path):
        if f.find(".png") != -1:
            img = get_preprocessed_img("{}/{}".format(input_path, f), IMAGE_SIZE)
            file_name = f[:f.find(".png")]

            np.savez_compressed("{}/{}".format(output_path, file_name), features=img)
            retrieve = np.load("{}/{}.npz".format(output_path, file_name))["features"]

            assert np.array_equal(img, retrieve)

            shutil.copyfile("{}/{}.gui".format(input_path, file_name), "{}/{}.gui".format(output_path, file_name))

    print("Numpy arrays saved in {}".format(output_path))


# # Training the data

# In[4]:


from keras.models import model_from_json


class AModel:
    def __init__(self, input_shape, output_size, output_path):
        self.model = None
        self.input_shape = input_shape
        self.output_size = output_size
        self.output_path = output_path
        self.name = ""

    def save(self):
        model_json = self.model.to_json()
        with open("{}/{}.json".format(self.output_path, self.name), "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights("{}/{}.h5".format(self.output_path, self.name))

    def load(self, name=""):
        output_name = self.name if name == "" else name
        with open("{}/{}.json".format(self.output_path, output_name), "r") as json_file:
            loaded_model_json = json_file.read()
        self.model = model_from_json(loaded_model_json)
        self.model.load_weights("{}/{}.h5".format(self.output_path, output_name))


# In[5]:


from keras.layers import Input, Dense, Dropout, \
                         RepeatVector, LSTM, concatenate, \
                         Conv2D, MaxPooling2D, Flatten
from keras.models import Sequential, Model
from keras.optimizers.legacy import RMSprop
from keras import *

class pix2code(AModel):
    def __init__(self, input_shape, output_size, output_path):
        AModel.__init__(self, input_shape, output_size, output_path)
        self.name = "pix2code"

        image_model = Sequential()
        image_model.add(Conv2D(32, (3, 3), padding='valid', activation='relu', input_shape=input_shape))
        image_model.add(Conv2D(32, (3, 3), padding='valid', activation='relu'))
        image_model.add(MaxPooling2D(pool_size=(2, 2)))
        image_model.add(Dropout(0.25))

        image_model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
        image_model.add(Conv2D(64, (3, 3), padding='valid', activation='relu'))
        image_model.add(MaxPooling2D(pool_size=(2, 2)))
        image_model.add(Dropout(0.25))

        image_model.add(Conv2D(128, (3, 3), padding='valid', activation='relu'))
        image_model.add(Conv2D(128, (3, 3), padding='valid', activation='relu'))
        image_model.add(MaxPooling2D(pool_size=(2, 2)))
        image_model.add(Dropout(0.25))

        image_model.add(Flatten())
        image_model.add(Dense(1024, activation='relu'))
        image_model.add(Dropout(0.3))
        image_model.add(Dense(1024, activation='relu'))
        image_model.add(Dropout(0.3))

        image_model.add(RepeatVector(CONTEXT_LENGTH))

        visual_input = Input(shape=input_shape)
        encoded_image = image_model(visual_input)

        language_model = Sequential()
        language_model.add(LSTM(128, return_sequences=True, input_shape=(CONTEXT_LENGTH, output_size)))
        language_model.add(LSTM(128, return_sequences=True))

        textual_input = Input(shape=(CONTEXT_LENGTH, output_size))
        encoded_text = language_model(textual_input)

        decoder = concatenate([encoded_image, encoded_text])

        decoder = LSTM(512, return_sequences=True)(decoder)
        decoder = LSTM(512, return_sequences=False)(decoder)
        decoder = Dense(output_size, activation='softmax')(decoder)

        self.model = Model(inputs=[visual_input, textual_input], outputs=decoder)

        optimizer = RMSprop(learning_rate=0.0001, clipvalue=1.0)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    def fit(self, images, partial_captions, next_words):
        self.model.fit([images, partial_captions], next_words, shuffle=False, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
        self.save()

    def fit_generator(self, generator, steps_per_epoch):
        self.model.fit(generator, steps_per_epoch=steps_per_epoch, epochs=EPOCHS, verbose=1)
        self.save()

    def predict(self, image, partial_caption):
        return self.model.predict([image, partial_caption], verbose=0)[0]

    def predict_batch(self, images, partial_captions):
        return self.model.predict([images, partial_captions], verbose=1)


# In[6]:


CONTEXT_LENGTH = 48
IMAGE_SIZE = 256
BATCH_SIZE = 64
EPOCHS = 10
STEPS_PER_EPOCH = 72000


START_TOKEN = "<START>"
END_TOKEN = "<END>"
PLACEHOLDER = " "
SEPARATOR = '->'


class Vocabulary:
    def __init__(self):
        self.binary_vocabulary = {}
        self.vocabulary = {}
        self.token_lookup = {}
        self.size = 0

        self.append(START_TOKEN)
        self.append(END_TOKEN)
        self.append(PLACEHOLDER)

    def append(self, token):
        if token not in self.vocabulary:
            self.vocabulary[token] = self.size
            self.token_lookup[self.size] = token
            self.size += 1

    def create_binary_representation(self):
        if sys.version_info >= (3,):
            items = self.vocabulary.items()
        else:
            items = self.vocabulary.iteritems()
        for key, value in items:
            binary = np.zeros(self.size)
            binary[value] = 1
            self.binary_vocabulary[key] = binary

    def get_serialized_binary_representation(self):
        if len(self.binary_vocabulary) == 0:
            self.create_binary_representation()

        string = ""
        if sys.version_info >= (3,):
            items = self.binary_vocabulary.items()
        else:
            items = self.binary_vocabulary.iteritems()
        for key, value in items:
            array_as_string = np.array2string(value, separator=',', max_line_width=self.size * self.size)
            string += "{}{}{}\n".format(key, SEPARATOR, array_as_string[1:len(array_as_string) - 1])
        return string

    def save(self, path):
        output_file_name = "{}/words.vocab".format(path)
        output_file = open(output_file_name, 'w')
        output_file.write(self.get_serialized_binary_representation())
        output_file.close()
        print("Saved vocabulary file")

    def retrieve(self, path):
        input_file = open("{}/words.vocab".format(path), 'r')
        buffer = ""
        for line in input_file:
            try:
                separator_position = len(buffer) + line.index(SEPARATOR)
                buffer += line
                key = buffer[:separator_position]
                value = buffer[separator_position + len(SEPARATOR):]
                value = np.fromstring(value, sep=',')

                self.binary_vocabulary[key] = value
                self.vocabulary[key] = np.where(value == 1)[0][0]
                self.token_lookup[np.where(value == 1)[0][0]] = key

                buffer = ""
            except ValueError:
                buffer += line
        input_file.close()
        self.size = len(self.vocabulary)


# In[7]:


class Utils:
    @staticmethod
    def sparsify(label_vector, output_size):
        sparse_vector = []

        for label in label_vector:
            sparse_label = np.zeros(output_size)
            sparse_label[label] = 1

            sparse_vector.append(sparse_label)

        return np.array(sparse_vector)

    @staticmethod
    def get_preprocessed_img(img_path, image_size):
        import cv2
        img = cv2.imread(img_path)
        img = cv2.resize(img, (image_size, image_size))
        img = img.astype('float32')
        img /= 255
        return img

    @staticmethod
    def show(image):
        import cv2
        cv2.namedWindow("view", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("view", image)
        cv2.waitKey(0)
        cv2.destroyWindow("view")


# In[8]:


class Dataset:
    def __init__(self):
        self.input_shape = None
        self.output_size = None

        self.ids = []
        self.input_images = []
        self.partial_sequences = []
        self.next_words = []

        self.voc = Vocabulary()
        self.size = 0

    @staticmethod
    def load_paths_only(path):
        print("Parsing data...")
        gui_paths = []
        img_paths = []
        for f in os.listdir(path):
            if f.find(".gui") != -1:
                path_gui = "{}/{}".format(path, f)
                gui_paths.append(path_gui)
                file_name = f[:f.find(".gui")]

                if os.path.isfile("{}/{}.png".format(path, file_name)):
                    path_img = "{}/{}.png".format(path, file_name)
                    img_paths.append(path_img)
                elif os.path.isfile("{}/{}.npz".format(path, file_name)):
                    path_img = "{}/{}.npz".format(path, file_name)
                    img_paths.append(path_img)

        assert len(gui_paths) == len(img_paths)
        return gui_paths, img_paths

    def load(self, path, generate_binary_sequences=False):
        print("Loading data...")
        for f in os.listdir(path):
            if f.find(".gui") != -1:
                gui = open("{}/{}".format(path, f), 'r')
                file_name = f[:f.find(".gui")]

                if os.path.isfile("{}/{}.png".format(path, file_name)):
                    img = Utils.get_preprocessed_img("{}/{}.png".format(path, file_name), IMAGE_SIZE)
                    self.append(file_name, gui, img)
                elif os.path.isfile("{}/{}.npz".format(path, file_name)):
                    img = np.load("{}/{}.npz".format(path, file_name))["features"]
                    self.append(file_name, gui, img)

        print("Generating sparse vectors...")
        self.voc.create_binary_representation()
        self.next_words = self.sparsify_labels(self.next_words, self.voc)
        if generate_binary_sequences:
            self.partial_sequences = self.binarize(self.partial_sequences, self.voc)
        else:
            self.partial_sequences = self.indexify(self.partial_sequences, self.voc)

        self.size = len(self.ids)
        assert self.size == len(self.input_images) == len(self.partial_sequences) == len(self.next_words)
        assert self.voc.size == len(self.voc.vocabulary)

        print("Dataset size: {}".format(self.size))
        print("Vocabulary size: {}".format(self.voc.size))

        self.input_shape = self.input_images[0].shape
        self.output_size = self.voc.size

        print("Input shape: {}".format(self.input_shape))
        print("Output size: {}".format(self.output_size))

    def convert_arrays(self):
        print("Convert arrays...")
        self.input_images = np.array(self.input_images)
        self.partial_sequences = np.array(self.partial_sequences)
        self.next_words = np.array(self.next_words)

    def append(self, sample_id, gui, img, to_show=False):
        if to_show:
            pic = img * 255
            pic = np.array(pic, dtype=np.uint8)
            Utils.show(pic)

        token_sequence = [START_TOKEN]
        for line in gui:
            line = line.replace(",", " ,").replace("\n", " \n")
            tokens = line.split(" ")
            for token in tokens:
                self.voc.append(token)
                token_sequence.append(token)
        token_sequence.append(END_TOKEN)

        suffix = [PLACEHOLDER] * CONTEXT_LENGTH

        a = np.concatenate([suffix, token_sequence])
        for j in range(0, len(a) - CONTEXT_LENGTH):
            context = a[j:j + CONTEXT_LENGTH]
            label = a[j + CONTEXT_LENGTH]

            self.ids.append(sample_id)
            self.input_images.append(img)
            self.partial_sequences.append(context)
            self.next_words.append(label)

    @staticmethod
    def indexify(partial_sequences, voc):
        temp = []
        for sequence in partial_sequences:
            sparse_vectors_sequence = []
            for token in sequence:
                sparse_vectors_sequence.append(voc.vocabulary[token])
            temp.append(np.array(sparse_vectors_sequence))

        return temp

    @staticmethod
    def binarize(partial_sequences, voc):
        temp = []
        for sequence in partial_sequences:
            sparse_vectors_sequence = []
            for token in sequence:
                sparse_vectors_sequence.append(voc.binary_vocabulary[token])
            temp.append(np.array(sparse_vectors_sequence))

        return temp

    @staticmethod
    def sparsify_labels(next_words, voc):
        temp = []
        for label in next_words:
            temp.append(voc.binary_vocabulary[label])

        return temp

    def save_metadata(self, path):
        metadata_array = np.array([self.input_shape, self.output_size, self.size], dtype=object)
        np.save("{}/meta_dataset".format(path), metadata_array)


# In[9]:


class Generator:
    @staticmethod
    def data_generator(voc, gui_paths, img_paths, batch_size, generate_binary_sequences=False, verbose=False, loop_only_one=False):
        assert len(gui_paths) == len(img_paths)
        voc.create_binary_representation()

        while 1:
            batch_input_images = []
            batch_partial_sequences = []
            batch_next_words = []
            sample_in_batch_counter = 0

            for i in range(0, len(gui_paths)):
                if img_paths[i].find(".png") != -1:
                    img = Utils.get_preprocessed_img(img_paths[i], IMAGE_SIZE)
                else:
                    img = np.load(img_paths[i])["features"]
                gui = open(gui_paths[i], 'r')

                token_sequence = [START_TOKEN]
                for line in gui:
                    line = line.replace(",", " ,").replace("\n", " \n")
                    tokens = line.split(" ")
                    for token in tokens:
                        voc.append(token)
                        token_sequence.append(token)
                token_sequence.append(END_TOKEN)

                suffix = [PLACEHOLDER] * CONTEXT_LENGTH

                a = np.concatenate([suffix, token_sequence])
                for j in range(0, len(a) - CONTEXT_LENGTH):
                    context = a[j:j + CONTEXT_LENGTH]
                    label = a[j + CONTEXT_LENGTH]

                    batch_input_images.append(img)
                    batch_partial_sequences.append(context)
                    batch_next_words.append(label)
                    sample_in_batch_counter += 1

                    if sample_in_batch_counter == batch_size or (loop_only_one and i == len(gui_paths) - 1):
                        if verbose:
                            print("Generating sparse vectors...")
                        batch_next_words = Dataset.sparsify_labels(batch_next_words, voc)
                        if generate_binary_sequences:
                            batch_partial_sequences = Dataset.binarize(batch_partial_sequences, voc)
                        else:
                            batch_partial_sequences = Dataset.indexify(batch_partial_sequences, voc)

                        if verbose:
                            print("Convert arrays...")
                        batch_input_images = np.array(batch_input_images)
                        batch_partial_sequences = np.array(batch_partial_sequences)
                        batch_next_words = np.array(batch_next_words)

                        if verbose:
                            print("Yield batch")
                        yield ([batch_input_images, batch_partial_sequences], batch_next_words)

                        batch_input_images = []
                        batch_partial_sequences = []
                        batch_next_words = []
                        sample_in_batch_counter = 0


# In[ ]:


random_seed_val = 1234
def run(input_path, output_path, is_memory_intensive=True, pretrained_model=None):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        
    dataset = Dataset()
    dataset.load(input_path, generate_binary_sequences=True)
    dataset.save_metadata(output_path)
    dataset.voc.save(output_path)

    if not is_memory_intensive:
        dataset.convert_arrays()

        input_shape = dataset.input_shape
        output_size = dataset.output_size

        print(len(dataset.input_images), len(dataset.partial_sequences), len(dataset.next_words))
        print(dataset.input_images.shape, dataset.partial_sequences.shape, dataset.next_words.shape)
    else:
        gui_paths, img_paths = Dataset.load_paths_only(input_path)

        input_shape = dataset.input_shape
        output_size = dataset.output_size
        steps_per_epoch = dataset.size / BATCH_SIZE

        voc = Vocabulary()
        voc.retrieve(output_path)

        generator = Generator.data_generator(voc, gui_paths, img_paths, batch_size=BATCH_SIZE, generate_binary_sequences=True)

    model = pix2code(input_shape, output_size, output_path)

    if pretrained_model is not None:
        model.model.load_weights(pretrained_model)

    if not is_memory_intensive:
        model.fit(dataset.input_images, dataset.partial_sequences, dataset.next_words)
    else:
        model.fit_generator(generator, steps_per_epoch=steps_per_epoch)

if __name__ == "__main__":
    np.random.seed(random_seed_val)
    input_path = "/Users/abhjha8/Image2Code/pix2code/src/dataset/web/training_set"
    output_path = "/Users/abhjha8/Image2Code/pix2code/src/models/model_web"
    use_generator = True
    pretrained_weigths = None
    run(input_path, output_path, is_memory_intensive=use_generator, pretrained_model=pretrained_weigths)


# In[ ]:




