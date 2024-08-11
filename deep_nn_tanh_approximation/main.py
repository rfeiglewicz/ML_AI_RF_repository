
import numpy as np
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd


def add_row_to_csv(x_hex, tanh_model, tanh_true, ulp_error, csv_file='data.csv'):
    new_row = pd.DataFrame({
        'x [hex]': [x_hex],
        'tanh(x)_model': [tanh_model],
        'tanh(x)_true': [tanh_true],
        'ulp_error': [ulp_error]
    })
    # Add new row to existing CSV file
    new_row.to_csv(csv_file, mode='a', header=False, index=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


N_SAMPLES = 10000
MIN_ARGUMENT = -10
MAX_ARGUMENT = 10

# Create the dataset of tanh function randomly picked values
X = np.random.uniform(MIN_ARGUMENT,MAX_ARGUMENT,N_SAMPLES)
Y = []
Y = np.asarray(Y)

for x in X:
    Y = np.append(Y,math.tanh(x))
# Plot Y = tanh(X) function
# plt.plot(X, Y,'o')
# plt.show()

# print (Y)
# print(X)

# Create tensor of Numpy arrays
X = torch.tensor(X, dtype=torch.float32).reshape(-1, 1).to(device)
Y = torch.tensor(Y, dtype=torch.float32).reshape(-1, 1).to(device)


class Tanh_approx_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(1, 16)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(16, 32)
        self.act2 = nn.ReLU()
        self.hidden3 = nn.Linear(32, 64)
        self.act3 = nn.ReLU()
        self.hidden4 = nn.Linear(64, 128)
        self.act4 = nn.ReLU()
        self.hidden5 = nn.Linear(128, 128)
        self.act5 = nn.ReLU()
        self.hidden6 = nn.Linear(128, 128)
        self.act6 = nn.ReLU()
        self.hidden7 = nn.Linear(128, 128)
        self.act7 = nn.ReLU()
        self.hidden8 = nn.Linear(128, 128)
        self.act8 = nn.ReLU()
        self.hidden9 = nn.Linear(128, 128)
        self.act9 = nn.ReLU()
        self.output = nn.Linear(128, 1)



    def forward(self, x_model):
        x_model = self.act1(self.hidden1(x_model))
        x_model = self.act2(self.hidden2(x_model))
        x_model = self.act3(self.hidden3(x_model))
        x_model = self.act4(self.hidden4(x_model))
        x_model = self.act5(self.hidden5(x_model))
        x_model = self.act6(self.hidden6(x_model))
        x_model = self.act7(self.hidden7(x_model))
        x_model = self.act8(self.hidden8(x_model))
        x_model = self.act9(self.hidden9(x_model))
        x_model = self.output(x_model)
        return x_model


tanh_model = Tanh_approx_model()

tanh_model.to(device)

print(tanh_model)

# train the model
loss_fn = nn.MSELoss()
optimizer = optim.Adam(tanh_model.parameters(), lr=0.001)

n_epochs = 100
batch_size = 100

for epoch in range(n_epochs):
    for i in range(0, len(X), batch_size):
        Xbatch = X[i:i + batch_size]
        y_pred = tanh_model(Xbatch)
        ybatch = Y[i:i + batch_size]
        loss = loss_fn(y_pred, ybatch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f'Finished epoch {epoch}, latest loss {loss}')


# compute accuracy (no_grad is optional)
with torch.no_grad():
    y_pred = tanh_model(X)
accuracy = (y_pred.round() == Y).float().mean()
print(f"Accuracy {accuracy}")

X_test = np.random.uniform(MIN_ARGUMENT,MAX_ARGUMENT,N_SAMPLES)
Y_test = []
Y_test = np.asarray(Y_test)

for x in X_test:
    Y_test = np.append(Y_test,math.tanh(x))

X_test = torch.tensor(X_test, dtype=torch.float32).reshape(-1, 1).to(device)
Y_test = torch.tensor(Y_test, dtype=torch.float32).reshape(-1, 1).to(device)



with torch.no_grad():
    Y_test_pred = tanh_model(X_test)

X_test = X_test.to('cpu')
Y_test_pred = Y_test_pred.to('cpu')

plt.plot(X_test, Y_test_pred,'ro')
plt.show()

# print (Y)
# print(X)

# Compute ULP error for predictions
# def ulp_error(y_true, y_pred):
#     """
#     Oblicz ULP jako miarę błędu między rzeczywistymi a przewidywanymi wartościami.
#     """
#     y_true_float32 = np.float32(y_true)
#     next_float32 = np.nextafter(y_true_float32, np.float32(np.inf))
#     # one_ulp_array = []
#     # one_ulp_array = np.asarray(one_ulp_array)
#     # for x in y_true_float32:
#     #     one_ulp_array = np.append(one_ulp_array,math.ulp(x))
#     y_pred_float32 = np.float32(y_pred)
#     abs_diff = np.abs(y_true_float32 - y_pred_float32)
#     ulp_error = np.float64(abs_diff) / np.float64(next_float32 - y_pred_float32)
#
#     return ulp_error

def ulp_error(y_true, y_pred):
    """
    Oblicz ULP jako miarę błędu między rzeczywistymi a przewidywanymi wartościami.
    ULP dla liczby x jest różnicą pomiędzy x a najbliższą możliwą do reprezentowania liczbą float.
    """
    y_true_float32 = np.float32(y_true)
    y_pred_float32 = np.float32(y_pred)

    # Oblicz ULP dla rzeczywistej wartości
    ulp_y_true = np.abs(np.nextafter(y_true_float32, np.float32(np.inf)) - y_true_float32)

    # Oblicz różnicę bezwzględną pomiędzy rzeczywistymi a przewidywanymi wartościami
    abs_diff = np.abs(y_true_float32 - y_pred_float32)

    # Oblicz błąd ULP
    ulp_error = abs_diff / ulp_y_true

    return ulp_error



# Predict using the trained model
with torch.no_grad():
    y_pred = tanh_model(X)

#Move data to cpu
Y_test = Y_test.cpu()

# Compute ULP error for each prediction
ulp_errors = ulp_error(Y_test.numpy(), Y_test_pred.numpy())
print(f"Maksymalny błąd ULP: {np.max(ulp_errors)}")


#create or open CSV file
csv_file = 'tan_ulp_error.csv'



ulp_error_whole = 0

x_iter = np.float32(MIN_ARGUMENT)

#move model to cpu
tanh_model = tanh_model.cpu()

while x_iter <= MAX_ARGUMENT:
    x_iter_tensor = torch.tensor([x_iter], dtype=torch.float32).reshape(-1, 1)

    with torch.no_grad():
        y_iter_model = tanh_model(x_iter_tensor)

    y_iter_true = np.tanh(x_iter)  # Oblicz wartość tanh dla aktualnego x
    y_iter_true_tensor = torch.tensor([y_iter_true], dtype=torch.float32).reshape(-1, 1)

    # Oblicz błędy ULP
    temp_ulp_errors = ulp_error(y_iter_true_tensor.numpy(), y_iter_model.numpy())

    # Zaktualizuj maksymalny błąd ULP
    ulp_error_whole = max(np.abs(ulp_error_whole), np.abs(np.max(temp_ulp_errors)))

    # x hex format
    x_iter_little_endian = x_iter.tobytes().hex()
    x_iter_big_endian = x_iter.tobytes()[::-1].hex()

    y_iter_model_hex_big_endian = y_iter_model.numpy().tobytes()[::-1].hex()
    y_iter_true = y_iter_true.tobytes()[::-1].hex()

    add_row_to_csv(x_iter_big_endian, y_iter_true, y_iter_model_hex_big_endian, temp_ulp_errors,csv_file)
    #print(f'Wartość tanh(x) dla x =  {x_iter_big_endian}')

    # Przejdź do następnej wartości
    x_iter = np.nextafter(x_iter, np.float32(MAX_ARGUMENT))

print("Bląd ulp na całym zakresie wynosi: " ,ulp_error_whole )

# Optionally, you can plot the results
plt.scatter(X, Y, label='True data')
plt.scatter(X, y_pred, label='Predicted data')
plt.legend()
plt.show()