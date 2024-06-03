import dataset
import util
import model

from tensorflow import reshape, GradientTape, shape
from tensorflow.keras.losses import CategoricalCrossentropy

# Train the model for 2 epochs
total_epochs = 1
# Limit the dataset to 10 for testing
dataset_paths = dataset.load_dataset_path(1)
dataset_size = len(dataset_paths)

llm = model.LLM()

evaluate = CategoricalCrossentropy()

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
        encoder_weights, decoder_weights, char_model_weights = llm.get_trainable_weights()
        encoder_grads = tape.gradient(loss, encoder_weights)
        decoder_grads = tape.gradient(loss, decoder_weights)
        char_model_grads = tape.gradient(loss, char_model_weights)

        total_loss += loss
        print("Step: {}/{}, Loss: {}".format(step,
              dataset_size, loss.numpy()), end="\r")

    print("Final Loss: ", total_loss.numpy() / dataset_size)
