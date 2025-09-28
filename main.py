"""
The following is a character level RNN (with LSTM), it is simple and was made to get some hands on experience and not for some peractical purpose.
we current have wo datasets shakespear test and naomes of dinosaurs (both are from Andre Karapthy's Gitub)
We can also add another dataset to Sample new texts (just set the parameters accordingly)
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide TF info logs
import tensorflow as tf
import numpy as np

# Choose dataset (can we anything else)
data = "dino"  # "shakespeare" or "dino" (can be anyting else, if text is present)
t = False       # True to train, False to generate text on pretrained weights
checkpoint_dir = "./training_checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

if data == "shakespeare":
    path = "shakespeare.txt"
    weights_path = os.path.join(checkpoint_dir, "shakespeare_model.weights.h5")
    EPOCHS = 20
    BATCH_SIZE = 64
    SEQ_LENGTH = 100
    EMBEDDING_DIM = 256
    RNN_UNITS = 512
    NUM_GENERATE = 1000
elif data == "dino":
    path = "dino_names.txt"
    weights_path = os.path.join(checkpoint_dir, "dino_model.weights.h5")
    EPOCHS = 200
    BATCH_SIZE = 32
    SEQ_LENGTH = 15
    EMBEDDING_DIM = 128
    RNN_UNITS = 256
    NUM_GENERATE = 50
else:
    raise ValueError("DATASET must be 'shakespeare' or 'dino'")

TEMPERATURE = 0.3

# Load dataset
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

# Preprocess text
vocab = sorted(set(text))
vocab_size = len(vocab)
char2idx = {c: i for i, c in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(SEQ_LENGTH + 1, drop_remainder=True)

def split_input_target(chunk):
    return chunk[:-1], chunk[1:]

dataset = sequences.map(split_input_target)
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# defining model
def build_model(vocab_size, embedding_dim, rnn_units, batch_size=None):
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(None,)),
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim),
        tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=False,
                             recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

if t:
    model = build_model(vocab_size, EMBEDDING_DIM, RNN_UNITS, batch_size=BATCH_SIZE)
else:
    model = build_model(vocab_size, EMBEDDING_DIM, RNN_UNITS, batch_size=1)

def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

model.compile(optimizer='adam', loss=loss)

# Text generation
def generate_text(model, start_string, num_words=3, temperature=0.4, max_generate=100):
    """
    Generate multiple words (dino names) from a seed string.
    Stops after generating `num_words` names separated by newline characters.
    """
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    words_generated = 0

    for i in range(max_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()
        
        next_char = idx2char[predicted_id]
        text_generated.append(next_char)
        input_eval = tf.expand_dims([predicted_id], 0)

        # Count words by newline characters
        if next_char == '\n':
            words_generated += 1
            if words_generated >= num_words:
                break

    return start_string + ''.join(text_generated)


# Train or load weights
if t:
    print(f"Training model on {data} dataset...")
    model.fit(dataset, epochs=EPOCHS, verbose=1)
    model.save_weights(weights_path)
    print(f"Weights successfully saved to: {os.path.abspath(weights_path)}")
else:
    if os.path.exists(weights_path):
        print(f"Loading weights from {weights_path}")
        model.load_weights(weights_path)
    else:
        raise ValueError(f"No weights found for {data}. Set TRAINING=True to train the model first.")

# User input & generate
seed_text = input(f"Enter seed text to generate {data} text: ")
generated = generate_text(model, seed_text, num_words=3, temperature=TEMPERATURE, max_generate=NUM_GENERATE)
print("\n--- Generated text ---\n")
print(generated)