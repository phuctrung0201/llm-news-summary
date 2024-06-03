import dataset
import util
import model

from tensorflow import reshape
from tensorflow.keras.losses import CategoricalCrossentropy

# Train the model for 2 epochs
total_epochs = 2
# Limit the dataset to 10 for testing
dataset_paths = dataset.load_dataset_path(10)
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

        predicted_vector = llm.train_predicting(
            sentence_vector, summary_vector)

        loss = evaluate(label_vector, predicted_vector)
        total_loss += loss
        print("Step: {}/{}, Loss: {}".format(step,
              dataset_size, loss.numpy()), end="\r")

    print("Final Loss: ", total_loss.numpy() / dataset_size)
