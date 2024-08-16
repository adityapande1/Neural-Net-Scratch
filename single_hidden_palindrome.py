from cs772 import (
    Neural_Network,
    Linear,
    ReLU,
    Sigmoid,
    BinaryCrossEntropy,
    Datagen,
    SGD,
    train_neural_net,
    Square,
)
from cs772.metrics import calculate_precision, calculate_recall
import pandas as pd
import numpy as np

num_features = 10
num_targets = 1
hidden_features = 1
batch_size = 1024
lr = 1e-1
epochs = 100
seed = 11

if __name__ == "__main__":
    np.random.seed(seed=seed)
    ## Data part
    df = pd.read_csv("data.csv", dtype={"string": "str"})
    labels = {"NP": int(0), "P": int(1)}
    df["target"] = df["target"].map(labels)
    trainX = np.array([[float(i) for i in [*x]] for x in df["string"]])
    trainY = df["target"].values.reshape(-1, 1)
    ## Neural Net part
    nn_layers = [
        Linear(
            num_inputs=num_features,
            num_outputs=hidden_features,
            weights_init=np.array(
                [
                    [
                        -13,
                        26,
                        11,
                        6,
                        -3,
                        3,
                        -6,
                        -11,
                        -26,
                        13,
                    ]
                ],
            ).T,
            bias_initializer=np.array([[0]]).T,
        ),
        Square(),
        Linear(
            num_inputs=hidden_features,
            num_outputs=num_targets,
            # bias_initializer=new_bias,
            weights_init=np.array([[-30]]).T,
            bias_initializer=np.array([10]).T,
        ),
        Sigmoid(),
    ]
    neural_network = Neural_Network(layers=nn_layers)
    train_datagen = Datagen((trainX, trainY), batch_size=batch_size)
    for trainX, trainY in train_datagen:
        train_preds = neural_network.forward(trainX)
        t_precision = calculate_precision(trainY, train_preds, threshold=0.50)
        t_recall = calculate_recall(trainY, train_preds, threshold=0.50)
        f1_score = 2 * ((t_precision * t_recall) / (t_precision + t_recall))
        print(
            f"Train Precision: {t_precision}\tTrain Recall: {t_recall}\tTrain F1 Score: {f1_score}"
        )
