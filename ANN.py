from keras.layers.core import Dense
from keras.models import Sequential

## This class defines an Artificial Neural Network.
class ANN:
    ## We pass in the name, the no of input layers, no. of output layers, a list of neurons in the hidden layers,
    ## the activation function, and optimizer function.
    def __init__(
        self, name, input_dim, output_dim, layers, activation, loss, optimizer
    ):
        ## We will be using a very simple sequential model.
        model = Sequential()

        ## Setting up the layers.
        first = True
        for num_nodes in layers:
            if first:
                model.add(Dense(num_nodes, input_dim=input_dim, activation=activation))
                first = False
            else:
                model.add(Dense(num_nodes))
        model.add(Dense(output_dim, activation='linear'))
        model.compile(loss=loss, optimizer=optimizer)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.name = name
        self.model = model

    ## Here we simply fit the input data=X to the labels=y.
    def train(self, X, y, batch_size):
        return self.model.fit(
            X, y,
            batch_size=batch_size,
            epochs=1,
            verbose=0
        )

    ## Predict the q-values for a given state.
    def predict(self, state):
        return self.model.predict(state)

    # Weights management
    ## Save weights for a given ANN.
    def set(self, weights):
        self.model.set_weights(weights)

    ## Return the weights of a given ANN
    def get(self):
        return self.model.get_weights()
