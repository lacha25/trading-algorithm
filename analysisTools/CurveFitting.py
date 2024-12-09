import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from .Trainer import Trainer
from .Tools import *
# Define MLP Model


class MLP(nn.Module):
    def __init__(self,input_size, output_size):
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(input_size, 128),  
            nn.ReLU(),
            nn.Linear(128, 64),  
            nn.ReLU()
        )
        self.output = nn.Linear(64, output_size)  

    def forward(self, x):
        x = self.hidden(x)
        x = self.output(x)
        return x




def f_MLP_CurveFitting_t(prix,WindowSize=10):
    #considering the last 10 values to predict the next one
    prix=np.array(prix).flatten()
    X,y=[],[]
    for i in range(WindowSize,len(prix)):
        X.append(prix[i-WindowSize:i])
        y.append(prix[i])
    X,y=np.array(X),np.array(y)

    # Splitting the data into training and testing sets
    X_train, y_train = X[:int(0.8*len(X))], y[:int(0.8*len(y))]
    X_test, y_test = X[int(0.8*len(X)):], y[int(0.8*len(y)):]
  
    #check if ther is nan values
    if np.isnan(X_train).any() or np.isnan(y_train).any() or np.isnan(X_test).any() or np.isnan(y_test).any():
        return "There is nan values in the data"


    model = MLP(10,1)
    method_obj = Trainer(model, lr=1e-4, epochs=300, batch_size=128, device="cpu",isClassification=False)

    # Train the model
    print("Training the model...")
    print("Starting training...")
    try:
        method_obj.fit(X_train, y_train)
    except Exception as e:
        print("Error during training:", e)
    print("Training completed or halted.")

    train_preds = method_obj.predict(X_train)
    mse = mse_fn(train_preds, y_train)
    print(f"Training set: MSE = {mse:.3f}")
    # Evaluate the model on the training and validation sets

    val_preds = method_obj.predict( X_test)
    mse = mse_fn(val_preds, y_test)
    print(f"Validation set: MSE = {mse:.3f}")


    plt.figure(figsize=(10, 6))
    plt.plot(range(len(y_test)), y_test, label="Actual Prices", color="blue")
    plt.plot(range(len(y_test)), val_preds, label="Predicted Prices", color="red", linestyle="--")
    plt.xlabel("Index")
    plt.ylabel("Price")
    plt.title("Validation Set: Actual vs Predicted Prices")
    plt.legend()
    plt.savefig("validation_plot.png")
    torch.save(method_obj.model.state_dict(), "model_weights.pth")


def f_MLP_CurveFitting_e(prix):
    prix=np.array(prix).flatten()
    X,y=[],[]
    for i in range(10,len(prix)):
        X.append(prix[i-10:i])
        y.append(prix[i])
    X,y=np.array(X),np.array(y)

    model = MLP(10,1)

# Load the previously saved model weights
    model.load_state_dict(torch.load("model_weights.pth"))

    # Set the model to evaluation mode
    model.eval()

  
    X_new_tensor = torch.tensor(X, dtype=torch.float32)
    with torch.no_grad():
        predictions = model(X_new_tensor)

    mse = mse_fn(predictions, y)
    print(f"Validation set: MSE = {mse:.3f}")
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(y_test)), y, label="Actual Prices", color="blue")
    plt.plot(range(len(y_test)), predictions, label="Predicted Prices", color="red", linestyle="--")
    plt.xlabel("Index")
    plt.ylabel("Price")
    plt.title("Validation Set: Actual vs Predicted Prices")
    plt.legend()
    plt.savefig("validation_plot.png")



def f_linerar_CurveFitting(prix):
    #time = np.arange(len(prix))
    #time can be implemented as an other feature (time of the day, day of the week, etc) which could greatly improve the model
    X,y=[],[]
    for i in range(10,len(prix)):
        X.append(prix[i-10:i])
        y.append(prix[i])
    X,y=np.array(X),np.array(y)
    model = LinearRegression()
    model.fit(X, y)
    next_features = np.array(prix[-10:]).reshape(1, -1)
    next_prediction = model.predict(next_features)
    return predictions