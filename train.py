import dataset
import util
import model
import os

from tensorflow import reshape, GradientTape
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.optimizers.legacy import Adam
from datetime import datetime

# Train the model for 2 epochs
total_epochs = 10
# Limit the dataset to 10 for testing
dataset_paths = dataset.load_dataset_path(100)
dataset_size = len(dataset_paths)

weights_path = "./weights/{}".format(
    datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
print("Weights will be saved at: {}".format(weights_path))
os.mkdir(weights_path)

llm = model.LLM()
evaluate = CategoricalCrossentropy()
optimizer = Adam(learning_rate=0.01)

for epoch in range(0, total_epochs):
    print("Epoch ", epoch + 1)
    total_loss = 0
    for i in range(0, dataset_size):
        step = i + 1

        sentence = dataset.load_text(dataset_paths[i][0])
        sentence_vector = util.get_sentence_vector(sentence)

        summary = dataset.load_text(dataset_paths[i][1])
        summary_vector = util.get_sentence_vector(summary)

        label_vector = summary_vector[1:]
        summary_vector = reshape(summary_vector[:-1], [-1, util.TOKENS_LEN])

        with GradientTape() as tape:
            predicted_vector = llm.train_predicting(
                sentence_vector, summary_vector)
            loss = evaluate(label_vector, predicted_vector)
        weights = llm.get_trainable_weights()
        grads = tape.gradient(
            loss, weights)

        optimizer.apply_gradients(zip(grads, weights))

        total_loss += loss
        print("Step: {}/{}, Loss: {}".format(step,
              dataset_size, loss.numpy()), end="\r")

    print("Final Loss: ", total_loss.numpy() / dataset_size)

llm.save_weights(weights_path)
