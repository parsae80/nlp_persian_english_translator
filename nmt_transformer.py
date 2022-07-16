import pathlib
import random
import string
import re
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization

lines_persian = ['']
lines_english = ['']
with open('/content/TEP.en-fa.fa', 'r', encoding='utf-8') as faText:
    for line in faText:
        for word in line.split(' '):
            if word == '.\n':
                line = line.replace(word,'')
        lines_persian.append(line)

with open('/content/TEP.en-fa.en', 'r', encoding='utf-8') as enText:
    for line in enText:
        for word in line.split(' '):
            if word == '.\n':
                line = line.replace(word, '')
        lines_english.append(line)


text_pairs = list(zip(lines_english, lines_persian))
random.shuffle(text_pairs)
num_val_samples = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples : num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples :]

print(f"{len(text_pairs)} total pairs")
print(f"{len(train_pairs)} training pairs")
print(f"{len(val_pairs)} validation pairs")
print(f"{len(test_pairs)} test pairs")

strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

vocab_size = 162535
sequence_length = 20
batch_size = 64


def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")


eng_vectorization = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length,
)
persian_vectorization = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length + 1,
    standardize=custom_standardization,
)
train_eng_texts = [pair[0] for pair in train_pairs]
train_persian_texts = [pair[1] for pair in train_pairs]
eng_vectorization.adapt(train_eng_texts)
persian_vectorization.adapt(train_persian_texts)


def format_dataset(lines_english, lines_persian):
    lines_english = eng_vectorization(lines_english)
    lines_persian = persian_vectorization(lines_persian)
    return (
        {
            "encoder_inputs": lines_english,
            "decoder_inputs": lines_persian[:, :-1],
        },
        lines_persian[:, 1:],
    )


def make_dataset(pairs):
    eng_texts, persian_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    persian_texts = list(persian_texts)
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, persian_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset)
    return dataset.shuffle(2048).prefetch(16).cache()


train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)
