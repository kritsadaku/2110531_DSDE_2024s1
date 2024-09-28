# 5_Time_series_forecasting_DataInGD

## get start with Pytorch
```python
pip install torchinfo
```
## get dataset 
```
wget https://github.com/pvateekul/2110531_DSDE_2023s1/raw/main/code/Week05_Intro_Deep_Learning/data/GOOG.csv
```
## import pandas and create the dataframe 
drop a column `Adj Close` from dataframe `df`
```python
import pandas as pd
df = pd.read_csv('GOOG.csv', index_col="Date")
df = df.drop(['Adj Close'], axis = 1) 
```

## plot the time series of Open price and Date
```python
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "plotly_white"

plot_template = dict(
    layout=go.Layout({
        "font_size": 18,
        "xaxis_title_font_size": 24,
        "yaxis_title_font_size": 24})
)

fig = px.line(df['Open'], labels=dict(
    created_at="Date", value="Open", variable="Sensor"
))
fig.update_layout(
  template=plot_template, legend=dict(orientation='h', y=1.02, title_text="")
)
fig.show()
```

## Create the target variable 
create the new column `Open_lead1`which value is given by day+1 and assign it to be target.
```python
target_col = "Open"
features = list(df.columns.difference([target_col]))

forecast_lead = 1
target = f"{target_col}_lead{forecast_lead}"

df[target] = df[target_col].shift(-forecast_lead)
df = df.iloc[:-forecast_lead]
```
currently, we have 
```
features: ['Close', 'High', 'Low', 'Volume']
target: Open_lead1
```

## Create a hold-out test set and preprocess the data
splits the DataFrame `df` into training, validation, and test sets 
```python
test_start = "2019-01-01"
val_start = "2018-01-01"

df_train = df.loc[:val_start].copy()
df_val = df.loc[val_start:test_start].copy()
df_test = df.loc[test_start:].copy()

print("Test set fraction:", round(len(df_test) / len(df),2))
```
print out the proportion of test data and total data

## Standardize the features and target, based on the training set
normalizes the training, validation, and test sets by scaling each column to have a mean of zero and a standard deviation of one based on the training data statistics.
```python
target_mean = df_train[target].mean()
target_stdev = df_train[target].std()

for c in df_train.columns:
    mean = df_train[c].mean()
    stdev = df_train[c].std()

    df_train[c] = (df_train[c] - mean) / stdev
    df_val[c] = (df_val[c] - mean) / stdev
    df_test[c] = (df_test[c] - mean) / stdev
```

## Create datasets that PyTorch `DataLoader` can work with
create class `SequenceDataset` which inherit from `torch.utils.data.Dataset` The `__getitem__` function allow the SequenceDataset class to be indexed like a list or an array, and return the value of x, y
```python
import torch
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, dataframe, target, features, sequence_length=5):
        self.features = features
        self.target = target
        self.sequence_length = sequence_length
        self.y = torch.tensor(dataframe[self.target].values).float()
        self.X = torch.tensor(dataframe[self.features].values).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if i >= self.sequence_length - 1:
            i_start = i - self.sequence_length + 1
            x = self.X[i_start:(i + 1), :]
        else:
            padding = self.X[0].repeat(self.sequence_length - i - 1, 1)
            x = self.X[0:(i + 1), :]
            x = torch.cat((padding, x), 0)

        return x, self.y[i]
```

example, defined `i = 27` and `sequence_length = 4` when call `train_dataset[i]` which is intance of SequenceDataset. So, the `i >= sequence_length-1` thus `i_start= 27 - 4 + 1` and return `X[24:27 + 1, :]` if `i < sequence_length-1` the padding will add the repeat the least value for concating the amount of tensor in x to be equal 4.
```python
#i >= sequence_length-1
i = 27
sequence_length = 4

train_dataset = SequenceDataset(
    df_train,
    target=target,
    features=features,
    sequence_length=sequence_length
)

X, y = train_dataset[i]
print(X)
```
```python
# i < sequence_length-1
i = 2
sequence_length = 4

train_dataset = SequenceDataset(
    df_train,
    target=target,
    features=features,
    sequence_length=sequence_length
)

X, y = train_dataset[i]
print(X)
```

Import the `Dataloader` and simple load the data by using `batch_size=3, shuffle=True`
```python
from torch.utils.data import DataLoader
torch.manual_seed(99)

train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)

X, y = next(iter(train_loader))
print(X.shape)
print(X)
```

## Create the datasets and data loaders for real

Note: In this tutorial we will use sequences of length 60 (60 days) to forcast 1 day ahead `sequence_length = 60`.
Fix the random state at `torch.manual_seed(101)` use `batch_size = 32`
```python
torch.manual_seed(101)

batch_size = 32
sequence_length = 60

train_dataset = SequenceDataset(
    df_train,
    target=target,
    features=features,
    sequence_length=sequence_length
)
val_dataset = SequenceDataset(
    df_val,
    target=target,
    features=features,
    sequence_length=sequence_length
)
test_dataset = SequenceDataset(
    df_test,
    target=target,
    features=features,
    sequence_length=sequence_length
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

X, y = next(iter(train_loader))

print("Features shape:", X.shape)
print("Target shape:", y.shape)
```
then print out the size of Features shape and Target shape
```bash
Features shape: torch.Size([32, 60, 4])
Target shape: torch.Size([32])
```

## Long short-term memory (LSTM)

## switch to use gpu
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

## create model ShallowRegressionLSTM class which inherit from `torch.nn.Module`

```python
from torch import nn

class ShallowRegressionLSTM(nn.Module):
    def __init__(self, num_features, hidden_units):
        super().__init__()
        self.num_features = num_features  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = 4

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]

        # initialize the hidden and cell state of the LSTM layer
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).to(device).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).to(device).requires_grad_()

        _, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn[-1]).flatten()  # get the output of the last hidden layer
        return out

```

assign learinging rate (alpha) = 5e-4 and number of hidden units in hidden layer = 60. assign loss function as `nn.MSELoss()` and optimizer as `torch.optim.Adam(model.parameters(), lr=learning_rate)`

```python
learning_rate = 5e-4
num_hidden_units = 60

model = ShallowRegressionLSTM(num_features=len(features), hidden_units=num_hidden_units)
model.to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
```

## summary the model
```python
from torchinfo import summary
summary(model, input_size=(32, 60, 4))
```
```bash
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
ShallowRegressionLSTM                    [32]                      --
├─LSTM: 1-1                              [32, 60, 60]              103,680
├─Linear: 1-2                            [32, 1]                   61
==========================================================================================
Total params: 103,741
Trainable params: 103,741
Non-trainable params: 0
Total mult-adds (M): 199.07
==========================================================================================
Input size (MB): 0.03
Forward/backward pass size (MB): 0.92
Params size (MB): 0.41
Estimated Total Size (MB): 1.37
==========================================================================================
```

## traing the model

import  `tqdm` to show the progress bar
```python
from tqdm.notebook import tqdm
```

create a function `train_model` and `test_model` to train and test the model. In test model, if `avg_loss < best_val_loss`, then save the model as `best_model.pth`, and return the best_val_loss. note: torch.no_grad() is used to disable gradient calculation during the forward pass.
```python
def train_model(data_loader, model, loss_function, optimizer):
    num_batches = len(data_loader)
    total_loss = 0
    model.train()

    for X, y in data_loader:
        X = X.to(device)
        y = y.to(device)
        output = model(X)
        loss = loss_function(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / num_batches
    print(f"Train loss: {avg_loss}")

def test_model(data_loader, model, loss_function, best_val_loss):

    num_batches = len(data_loader)
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for X, y in data_loader:
            X = X.to(device)
            y = y.to(device)
            output = model(X)
            total_loss += loss_function(output, y).item()

    avg_loss = total_loss / num_batches
    print(f"Test loss: {avg_loss}")
    if avg_loss < best_val_loss:
        best_val_loss = avg_loss
        torch.save(model.state_dict(), 'model.pth')
        print('Save new best model')
    return best_val_loss
```

create a loop to train and test the model
```python
best_val_loss = torch.inf
for ix_epoch in tqdm(range(100)):
    print(f"Epoch {ix_epoch}\n---------")
    train_model(train_loader, model, loss_function, optimizer=optimizer)
    best_val_loss = test_model(val_loader, model, loss_function, best_val_loss)
    print()
```

## Evaluation
create function `predict` to predict the output of the model and return the output.
Note `with torch.no_grad()` to avoid gradient computation
```python
def predict(data_loader, model):
    """Just like `test_loop` function but keep track of the outputs instead of the loss
    function.
    """
    output = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for X, _ in data_loader:
            X = X.to(device)
            y_star = model(X)
            output = torch.cat((output, y_star.detach().cpu()), 0)

    return output
```

evaluate the model on the test set and append column `Model forecast`
```python
train_eval_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

ystar_col = "Model forecast"
df_train[ystar_col] = predict(train_eval_loader, model).numpy()
df_val[ystar_col] = predict(val_loader, model).numpy()
df_test[ystar_col] = predict(test_loader, model).numpy()

df_out = pd.concat((df_train, df_val, df_test))[[target, ystar_col]]

for c in df_out.columns:
    df_out[c] = df_out[c] * target_stdev + target_mean

print(df_out)
```

calculate the MAPE and RMSE
```python
import numpy as np
import math
from sklearn.metrics import mean_squared_error

def MAPE(Y_actual,Y_Predicted):
    mape = np.mean(np.abs((Y_actual - Y_Predicted)/Y_actual))*100
    return mape

print( 'MPAE =', MAPE(df_test['Open_lead1'], df_test['Model forecast']) )
print( 'RMSE =', math.sqrt(mean_squared_error(df_test['Open_lead1'], df_test['Model forecast'])) )
```
plot the test set and forecast set
```python
fig = px.line(df_out, labels={'value': "Open", 'created_at': 'Date'})
fig.add_vline(x=val_start, line_width=4, line_dash="dash")
fig.add_vline(x=test_start, line_width=4, line_dash="dash")
# fig.add_annotation(xref="paper", x=0.75, yref="paper", y=0.8, text="Test set start", showarrow=False)
fig.update_layout(
  template=plot_template, legend=dict(orientation='h', y=1.02, title_text="")
)
fig.show()
```

## swith model from LSTM to Gated Recurrent Unit  (GRU)

create a class `ShallowRegressionGRU` to create a GRU model. and switch model from LSTM to GRU in `forward` function
```python
from torch import nn

class ShallowRegressionGRU(nn.Module):
    def __init__(self, num_features, hidden_units):
        super().__init__()
        self.num_features = num_features  # this is the number of features
        self.hidden_units = hidden_units
        self.num_layers = 4

        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)

    def forward(self, x):
        batch_size = x.shape[0]

        # initialize the hidden and cell state of the LSTM layer
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).to(device).requires_grad_()

        _, hn = self.gru(x, h0)
        out = self.linear(hn[-1]).flatten()  # get the output of the last hidden layer
        return out
```

switch model from LSTM to GRU at variable `model`
```python
learning_rate = 5e-4
num_hidden_units = 60

model = ShallowRegressionLSTM(num_features=len(features), hidden_units=num_hidden_units)
model.to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
```
summary the model again
```bash
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
ShallowRegressionLSTM                    [32]                      --
├─LSTM: 1-1                              [32, 60, 60]              103,680
├─Linear: 1-2                            [32, 1]                   61
==========================================================================================
Total params: 103,741
Trainable params: 103,741
Non-trainable params: 0
Total mult-adds (M): 199.07
==========================================================================================
Input size (MB): 0.03
Forward/backward pass size (MB): 0.92
Params size (MB): 0.41
Estimated Total Size (MB): 1.37
==========================================================================================
```

use a function `train_model` and `test_model` to train and test the model. again.

<br/>
use function `predict` to predict the output of the model again.

<br/>
calculate the MAPE and RMSE again.

</br>
and plot the test set and forecast set again.
