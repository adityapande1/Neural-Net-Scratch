from cs772 import (
    Neural_Network,
    Linear,
    ReLU,
    Sigmoid,
    BinaryCrossEntropy,
    Datagen,
    SGD,
    Square,
    train_neural_net,
)
import pandas as pd
import numpy as np

num_features = 10
num_targets = 1
hidden_features = 1
batch_size = 10
lr = 1e-3
epochs = 1000
seed = 11

if __name__ == "__main__":
    np.random.seed(seed=seed)
    ## Data part
    df = pd.read_csv("data.csv", dtype={"string": "str"})
    labels = {"NP": int(0), "P": int(1)}
    df["target"] = df["target"].map(labels)
    df = df.sample(frac=1)
    trainX = np.array([[float(i) for i in [*x]] for x in df["string"]])
    trainY = df["target"].values.reshape(-1, 1)
    ## correcting bias of output for handling data imbalance
    new_bias = np.log(32 / 992)
    ## class weights calculation
    weight_for_p = (1 / 32) * (1024 / 2.0)
    weight_for_np = (1 / 992) * (1024 / 2.0)
    class_weights = {0: weight_for_np, 1: weight_for_p}
    print("Class Weights: ", {"NP": weight_for_np, "P": weight_for_p})
    ## Neural Net part
    nn_layers = [
        Linear(num_inputs=num_features, num_outputs=hidden_features),
        Square(),
        Linear(
            num_inputs=hidden_features,
            num_outputs=num_targets,
            bias_initializer=new_bias,
        ),
        Sigmoid(),
    ]
    neural_network = Neural_Network(layers=nn_layers)
    optimizer = SGD(lr=lr)
    # loss_function = BinaryCrossEntropy(class_weights=class_weights)
    loss_function = BinaryCrossEntropy()
    neural_network.compile(loss_function, optimizer)
    train_datagen = Datagen((trainX, trainY), batch_size=batch_size)
    train_hist, val_hist = train_neural_net(
        net=neural_network,
        epochs=epochs,
        train_datagen=train_datagen,
        val_datagen=None,
        record_after_epochs=1,
        early_stopping=True,
    )
