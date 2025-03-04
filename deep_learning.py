import numpy as np
from tqdm import tqdm

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

class Net:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.01):

        self.learning_rate = learning_rate

        self.w = []
        self.b = []
        last_size = input_size

        for i in range(len(hidden_layers)):
            self.w.append(np.random.randn(last_size, hidden_layers[i]))
            last_size = hidden_layers[i]
            self.b.append(np.random.randn(hidden_layers[i]))
        
        self.w.append(np.random.randn(last_size, output_size))
        self.b.append(np.random.randn(output_size))

    
    def forward(self, X):

        self.h_in = []
        self.h_out = []

        self.h_in.append(np.dot(X, self.w[0]) + self.b[0])
        self.h_out.append(sigmoid(self.h_in[0]))

        for w in range(1, len(self.w) - 1):
            self.h_in.append(np.dot(self.h_out[w - 1], self.w[w]) + self.b[w])
            self.h_out.append(sigmoid(self.h_in[w]))

        self.out_in = np.dot(self.h_out[-1], self.w[-1]) + self.b[-1]
        self.out_out = sigmoid(self.out_in)

        return self.out_out

    
    def backpropagate(self, X, y):
        
        out_err = y - self.out_out
        out_del = out_err * sigmoid_derivative(self.out_out)

        h_err = []
        h_del = []
        h_err.append(out_del.dot(self.w[-1].T))
        h_del.append(h_err[0] * sigmoid_derivative(self.h_out[-1]))

        for i in range(len(self.w) - 2, 0, -1):
            h_err.insert(0, h_del[0].dot(self.w[i].T))
            h_del.insert(0, h_err[0] * sigmoid_derivative(self.h_out[i - 1]))

        self.w[-1] += self.learning_rate * self.h_out[-1].T.dot(out_del)
        self.b[-1] += self.learning_rate * np.sum(out_del, axis=0)

        for i in range(len(h_del) - 1, 0, -1):
            self.w[i] += self.learning_rate * self.h_out[i - 1].T.dot(h_del[i])
            self.b[i] += self.learning_rate * np.sum(h_del[i], axis=0)

        self.w[0] += self.learning_rate * X.T.dot(h_del[0])
        self.b[0] += self.learning_rate * np.sum(h_del[0], axis=0)

    
    def train(self, X, y, epochs = 1000) -> float:

        for epoch in tqdm(range(epochs), desc="Training Progress"):
            output = self.forward(X)
            self.backpropagate(X, y)
            if epoch == epochs - 1:
                last_loss = mean_squared_error(y, output)
        
        return last_loss

    def predict(self, X, threshold=0.5):
        
        inp = np.dot(X, self.w[0])
        outp = sigmoid(inp)

        for i in range(1, len(self.w)):
            inp = np.dot(outp, self.w[i])
            outp = sigmoid(inp)
        
        if outp.shape[1] == 1:
            return (outp > threshold).astype(int)
        else:
            return np.argmax(outp, axis=1)

    

if __name__ == "__main__":

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]]) 

    input_size = 2
    hidden_layers = [12, 12, 12]
    output_size = 1

    nn = Net(input_size, hidden_layers, output_size, learning_rate=0.1)

    epochs = 10000
    last_loss = nn.train(X, y, epochs=epochs)
    print(f"Epoch {epochs}, Loss: {last_loss}")

    X_new = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    predictions = nn.predict(X_new)
    print("Dự đoán:", predictions)