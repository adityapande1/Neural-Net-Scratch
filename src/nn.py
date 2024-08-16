import numpy as np
from cs772.metrics import calculate_precision, calculate_recall


class Neural_Network:
    def __init__(self, layers):
        self.layers = layers
        self.loss = np.inf
        self.loss_fn = None
        self.optimizer = None

    def forward(self, inputs):
        inp = inputs
        for layer in self.layers:
            layer.forward(inp)
            inp = layer.activations
        return inp

    def compile(self, loss_fn, optimizer):
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def optimize(self, y_true, y_pred):
        if self.loss_fn is None or self.optimizer is None:
            raise Exception("compile neural network first!")
        loss = self.loss_fn.forward(y_true, y_pred)
        self.loss_fn.backward(y_true, y_pred)
        delta = self.loss_fn.grads_inputs
        for layer in reversed(self.layers):
            layer.backward(delta)
            delta = layer.grads_inputs
        self.optimizer.update_params(self.layers)
        return loss

    def calculate_loss(self, y_true, y_pred):
        return self.loss_fn.forward(y_true, y_pred)


def train_neural_net(
    net,
    epochs,
    train_datagen,
    val_datagen=None,
    precision_thresh=0.5,
    recall_thresh=0.5,
    early_stopping=True,
    patience=5,
    record_after_epochs=1,
    verbose=True,
):
    # train trackers
    train_precisions = []
    train_recalls = []
    train_losses = []
    # val trackers
    val_precisions = []
    val_recalls = []
    val_losses = []

    best_loss = np.inf
    patience_counter = 0
    t_counter = record_after_epochs
    v_counter = record_after_epochs
    for epoch in range(epochs):
        print("Epoch: ", epoch + 1)
        for trainX, trainY in train_datagen:
            train_preds = net.forward(trainX)
            t_precision = calculate_precision(
                trainY, train_preds, threshold=precision_thresh
            )
            t_recall = calculate_recall(trainY, train_preds, threshold=recall_thresh)
            t_loss = net.optimize(trainY, train_preds)
        if epoch == t_counter - 1:
            train_precisions.append(t_precision)
            train_recalls.append(t_recall)
            train_losses.append(t_loss)
            if verbose:
                f1_score = 2 * ((t_precision * t_recall) / (t_precision + t_recall))
                print(
                    f"Train Loss: {t_loss}\tTrain Precision: {t_precision}\tTrain Recall: {t_recall}\tTrain F1 Score: {f1_score}\tmean loss grads: {np.mean(net.loss_fn.grads_inputs)}"
                )
            t_counter += record_after_epochs
        if val_datagen is not None:
            for valX, valY in val_datagen:
                val_preds = net.forward(valX)
                v_loss = net.calculate_loss(valY, val_preds)
                v_precision = calculate_precision(
                    valY, val_preds, threshold=precision_thresh
                )
                v_recall = calculate_recall(valY, val_preds, threshold=recall_thresh)

            if epoch == v_counter - 1:
                val_precisions.append(v_precision)
                val_recalls.append(v_recall)
                val_losses.append(v_loss)
                if verbose:
                    print(
                        f"Valid Loss: {v_loss}\tValid Precision: {v_precision}\tValid Recall: {v_recall}"
                    )
                v_counter += record_after_epochs

            if best_loss > v_loss:
                best_loss = v_loss
                patience_counter = 0
            else:
                patience_counter += 1

        if early_stopping and patience_counter == patience:
            print(
                "EARLY STOPPING Validation loss did not improved, best validation loss: ",
                best_loss,
            )
            break

    return (
        {
            "precisions": train_precisions,
            "recalls": train_recalls,
            "losses": train_losses,
        },
        {"precisions": val_precisions, "recalls": val_recalls, "losses": val_losses},
    )
