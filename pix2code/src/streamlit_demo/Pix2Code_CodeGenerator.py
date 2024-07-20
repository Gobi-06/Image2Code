
import os
import numpy as np
import os
import cv2

import string
import random
import json
import os
from os.path import basename


# In[2]:


class Imageprocessing_utils:
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
        img = cv2.imread(img_path)
        img = cv2.resize(img, (image_size, image_size))
        img = img.astype('float32')
        img /= 255
        return img

    @staticmethod
    def show(image):
        cv2.namedWindow("view", cv2.WINDOW_AUTOSIZE)
        cv2.imshow("view", image)
        cv2.waitKey(0)
        cv2.destroyWindow("view")


# In[3]:


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


class Node:
    def __init__(self, key, value, data=None):
        self.key = key
        self.value = value
        self.data = data
        self.parent = None
        self.root = None
        self.children = []
        self.level = 0

    def add_children(self, children, beam_width):
        for child in children:
            child.level = self.level + 1
            child.value = child.value * self.value

        nodes = sorted(children, key=lambda node: node.value, reverse=True)
        nodes = nodes[:beam_width]

        for node in nodes:
            self.children.append(node)
            node.parent = self

        if self.parent is None:
            self.root = self
        else:
            self.root = self.parent.root
        child.root = self.root

    def remove_child(self, child):
        self.children.remove(child)

    def max_child(self):
        if len(self.children) == 0:
            return self

        max_childs = []
        for child in self.children:
            max_childs.append(child.max_child())

        nodes = sorted(max_childs, key=lambda child: child.value, reverse=True)
        return nodes[0]

    def show(self, depth=0):
        print(" " * depth, self.key, self.value, self.level)
        for child in self.children:
            child.show(depth + 2)


class BeamSearch:
    def __init__(self, beam_width=1):
        self.beam_width = beam_width

        self.root = None
        self.clear()

    def search(self):
        result = self.root.max_child()

        self.clear()
        return self.retrieve_path(result)

    def add_nodes(self, parent, children):
        parent.add_children(children, self.beam_width)

    def is_valid(self):
        leaves = self.get_leaves()
        level = leaves[0].level
        counter = 0
        for leaf in leaves:
            if leaf.level == level:
                counter += 1
            else:
                break

        if counter == len(leaves):
            return True

        return False

    def get_leaves(self):
        leaves = []
        self.search_leaves(self.root, leaves)
        return leaves

    def search_leaves(self, node, leaves):
        for child in node.children:
            if len(child.children) == 0:
                leaves.append(child)
            else:
                self.search_leaves(child, leaves)

    def prune_leaves(self):
        leaves = self.get_leaves()

        nodes = sorted(leaves, key=lambda leaf: leaf.value, reverse=True)
        nodes = nodes[self.beam_width:]

        for node in nodes:
            node.parent.remove_child(node)

        while not self.is_valid():
            leaves = self.get_leaves()
            max_level = 0
            for leaf in leaves:
                if leaf.level > max_level:
                    max_level = leaf.level

            for leaf in leaves:
                if leaf.level < max_level:
                    leaf.parent.remove_child(leaf)

    def clear(self):
        self.root = None
        self.root = Node("root", 1.0, None)

    def retrieve_path(self, end):
        path = [end.key]
        data = [end.data]
        while end.parent is not None:
            end = end.parent
            path.append(end.key)
            data.append(end.data)

        result_path = []
        result_data = []
        for i in range(len(path) - 2, -1, -1):
            result_path.append(path[i])
            result_data.append(data[i])
        return result_path, result_data


# In[7]:


class Sampler:
    def __init__(self, voc_path, input_shape, output_size, context_length):
        self.voc = Vocabulary()
        self.voc.retrieve(voc_path)

        self.input_shape = input_shape
        self.output_size = output_size

        print("Vocabulary size: {}".format(self.voc.size))
        print("Input shape: {}".format(self.input_shape))
        print("Output size: {}".format(self.output_size))

        self.context_length = context_length

    def predict_greedy(self, model, input_img, require_sparse_label=True, sequence_length=150, verbose=False):
        current_context = [self.voc.vocabulary[PLACEHOLDER]] * (self.context_length - 1)
        current_context.append(self.voc.vocabulary[START_TOKEN])
        if require_sparse_label:
            current_context = Imageprocessing_utils.sparsify(current_context, self.output_size)

        predictions = START_TOKEN
        out_probas = []

        for i in range(0, sequence_length):
            if verbose:
                print("predicting {}/{}...".format(i, sequence_length))

            probas = model.predict(input_img, np.array([current_context]))
            prediction = np.argmax(probas)
            out_probas.append(probas)

            new_context = []
            for j in range(1, self.context_length):
                new_context.append(current_context[j])

            if require_sparse_label:
                sparse_label = np.zeros(self.output_size)
                sparse_label[prediction] = 1
                new_context.append(sparse_label)
            else:
                new_context.append(prediction)

            current_context = new_context

            predictions += self.voc.token_lookup[prediction]

            if self.voc.token_lookup[prediction] == END_TOKEN:
                break

        return predictions, out_probas

    def recursive_beam_search(self, model, input_img, current_context, beam, current_node, sequence_length):
        probas = model.predict(input_img, np.array([current_context]))

        predictions = []
        for i in range(0, len(probas)):
            predictions.append((i, probas[i], probas))

        nodes = []
        for i in range(0, len(predictions)):
            prediction = predictions[i][0]
            score = predictions[i][1]
            output_probas = predictions[i][2]
            nodes.append(Node(prediction, score, output_probas))

        beam.add_nodes(current_node, nodes)

        if beam.is_valid():
            beam.prune_leaves()
            if sequence_length == 1 or self.voc.token_lookup[beam.root.max_child().key] == END_TOKEN:
                return

            for node in beam.get_leaves():
                prediction = node.key

                new_context = []
                for j in range(1, self.context_length):
                    new_context.append(current_context[j])
                sparse_label = np.zeros(self.output_size)
                sparse_label[prediction] = 1
                new_context.append(sparse_label)

                self.recursive_beam_search(model, input_img, new_context, beam, node, sequence_length - 1)

    def predict_beam_search(self, model, input_img, beam_width=3, require_sparse_label=True, sequence_length=150):
        predictions = START_TOKEN
        out_probas = []

        current_context = [self.voc.vocabulary[PLACEHOLDER]] * (self.context_length - 1)
        current_context.append(self.voc.vocabulary[START_TOKEN])
        if require_sparse_label:
            current_context = Imageprocessing_utils.sparsify(current_context, self.output_size)

        beam = BeamSearch(beam_width=beam_width)

        self.recursive_beam_search(model, input_img, current_context, beam, beam.root, sequence_length)

        predicted_sequence, probas_sequence = beam.search()

        for k in range(0, len(predicted_sequence)):
            prediction = predicted_sequence[k]
            probas = probas_sequence[k]
            out_probas.append(probas)

            predictions += self.voc.token_lookup[prediction]

        return predictions, out_probas


# In[8]:


class CodeGenerator_utils:
    @staticmethod
    def get_random_text(length_text=10, space_number=1, with_upper_case=True):
        results = []
        while len(results) < length_text:
            char = random.choice(string.ascii_letters[:26])
            results.append(char)
        if with_upper_case:
            results[0] = results[0].upper()

        current_spaces = []
        while len(current_spaces) < space_number:
            space_pos = random.randint(2, length_text - 3)
            if space_pos in current_spaces:
                break
            results[space_pos] = " "
            if with_upper_case:
                results[space_pos + 1] = results[space_pos - 1].upper()

            current_spaces.append(space_pos)

        return ''.join(results)

    @staticmethod
    def get_ios_id(length=10):
        results = []

        while len(results) < length:
            char = random.choice(string.digits + string.ascii_letters)
            results.append(char)

        results[3] = "-"
        results[6] = "-"

        return ''.join(results)

    @staticmethod
    def get_android_id(length=10):
        results = []

        while len(results) < length:
            char = random.choice(string.ascii_letters)
            results.append(char)

        return ''.join(results)


# In[9]:


class Node_Generator:
    def __init__(self, key, parent_node, content_holder):
        self.key = key
        self.parent = parent_node
        self.children = []
        self.content_holder = content_holder

    def add_child(self, child):
        self.children.append(child)

    def show(self):
        # print(self.key)
        for child in self.children:
            child.show()

    def render(self, mapping, rendering_function=None):
        content = ""
        for child in self.children:
            content += child.render(mapping, rendering_function)

        value = mapping[self.key]
        if rendering_function is not None:
            value = rendering_function(self.key, value)

        if len(self.children) != 0:
            value = value.replace(self.content_holder, content)

        return value


# In[10]:


class Compiler:
    def __init__(self, dsl_mapping_file_path):
        with open(dsl_mapping_file_path) as data_file:
            self.dsl_mapping = json.load(data_file)

        self.opening_tag = self.dsl_mapping["opening-tag"]
        self.closing_tag = self.dsl_mapping["closing-tag"]
        self.content_holder = self.opening_tag + self.closing_tag

        # self.root = Node("body", None, self.content_holder)

    def compile(self, input_file_path, output_file_path, rendering_function=None):
        self.root = Node_Generator("body", None, self.content_holder)

        dsl_file = open(input_file_path)
        current_parent = self.root
        for token in dsl_file:
            token = token.replace(" ", "").replace("\n", "")

            if token.find(self.opening_tag) != -1:
                token = token.replace(self.opening_tag, "")

                element = Node_Generator(token, current_parent, self.content_holder)
                current_parent.add_child(element)
                current_parent = element
            elif token.find(self.closing_tag) != -1:
                current_parent = current_parent.parent
            else:
                tokens = token.split(",")
                for t in tokens:
                    element = Node_Generator(t, current_parent, self.content_holder)
                    current_parent.add_child(element)

        output_html = self.root.render(self.dsl_mapping, rendering_function=rendering_function)
        with open(output_file_path, 'w') as output_file:
            output_file.write(output_html)



# To generate gui files

model_web = "/Users/abhjha8/Image2Code/pix2code/src/models/model_web"
# model_android = "/Users/abhjha8/kaggle_competitions/Image2Code/pix2code/model_android"
# model_ios = "/Users/abhjha8/kaggle_competitions/Image2Code/pix2code/model_ios"

trained_weights_path = "/Users/abhjha8/Image2Code/pix2code/src/models/model_web"
trained_model_name = "pix2code"
input_path = "/Users/abhjha8/Image2Code/pix2code/src/dataset/web/eval_set"
gui_output_path = os.path.join(os.getcwd(),"uploaded_files")
search_method = "greedy"

if not os.path.exists(gui_output_path):
    os.makedirs(gui_output_path)

meta_dataset = np.load("{}/meta_dataset.npy".format(trained_weights_path), allow_pickle=True)

input_shape = meta_dataset[0]
output_size = meta_dataset[1]

model = pix2code(input_shape, output_size, trained_weights_path)
model.load(trained_model_name)

sampler = Sampler(trained_weights_path, input_shape, output_size, CONTEXT_LENGTH)



def generate_gui_file(image_file_path):
    print("Image file path",image_file_path)
    evaluation_img = Imageprocessing_utils.get_preprocessed_img(image_file_path, IMAGE_SIZE)

    f = image_file_path.split("/")[-1]
    file_name = f[:f.find(".png")]

    if search_method == "greedy":
        result, _ = sampler.predict_greedy(model, np.array([evaluation_img]))
        # print("Result greedy: {}".format(result))
    else:
        beam_width = int(search_method)
        # print("Search with beam width: {}".format(beam_width))
        result, _ = sampler.predict_beam_search(model, np.array([evaluation_img]), beam_width=beam_width)
        # print("Result beam: {}".format(result))

    with open("{}/{}.gui".format(gui_output_path, file_name), 'w') as out_f:
        out_f.write(result.replace(START_TOKEN, "").replace(END_TOKEN, ""))
    complete_gui_output_path = "{}/{}.gui".format(gui_output_path, file_name)
    return gui_output_path,file_name


def render_content_with_text(key, value):
    FILL_WITH_RANDOM_TEXT = True
    TEXT_PLACE_HOLDER = "[]"
    if FILL_WITH_RANDOM_TEXT:
        if key.find("btn") != -1:
            value = value.replace(TEXT_PLACE_HOLDER, CodeGenerator_utils.get_random_text())
        elif key.find("title") != -1:
            value = value.replace(TEXT_PLACE_HOLDER, CodeGenerator_utils.get_random_text(length_text=5, space_number=0))
        elif key.find("text") != -1:
            value = value.replace(TEXT_PLACE_HOLDER,
                                  CodeGenerator_utils.get_random_text(length_text=56, space_number=7, with_upper_case=False))
    return value


def generate_html_file(gui_output_path,file_name):
    input_folder_path = gui_output_path
    output_folder_path = "{}/{}".format(gui_output_path,"uploaded_files")
    print(file_name)
    print(input_folder_path)
    print(output_folder_path)
    dsl_path = "/Users/abhjha8/Image2Code/pix2code/src/utils/web-dsl-mapping.json"
    compiler = Compiler(dsl_path)
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)
    exception_count=0;total_count=0
    output_file_path = ""
    for input_file in os.listdir(input_folder_path):
        if input_file.find(".gui")==-1:
            continue

        # print("GUI output path ",gui_output_path)
        # if input_file!=file_name:
        #     continue
        input_file = "{}/{}".format(input_folder_path, input_file)
        
        file_uid = basename(input_file)[:basename(input_file).find(".")]
        path = input_file[:input_file.find(file_uid)]
        input_file_path = "{}/{}.gui".format(path, file_uid)
        output_file_path = "{}/{}.html".format(output_folder_path, file_uid)
        
        try:
            compiler.compile(input_file_path, output_file_path, rendering_function=render_content_with_text)
            print("Compiled successfully ",input_file_path)
        except Exception as e:
            print("Not able to compile file: {}".format(input_file))
            print("Exception raised {}".format(e))
            exception_count+=1
            continue
        total_count+=1
    return output_file_path
    # print("Total files: {} and exception count: {}".format(total_count, exception_count))





