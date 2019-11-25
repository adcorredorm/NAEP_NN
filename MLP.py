import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

data = pd.read_csv('processed_data/features_30m.csv')
data = data.drop(['Unnamed: 0', 'ID'], axis=1)
labels = pd.read_csv('public_data/data_train_label.csv')
labels = labels.drop(['STUDENTID'], axis=1)

Y = labels['EfficientlyCompletedBlockB']
X = data

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= 0.3, random_state=27)

clf = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
              beta_1=0.9, beta_2=0.999, early_stopping=False,
              epsilon=1e-08, hidden_layer_sizes=(1000, 50),
              learning_rate='constant', learning_rate_init=0.001,
              max_iter=200, momentum=0.9, n_iter_no_change=10,
              nesterovs_momentum=True, power_t=0.5, random_state=1,
              shuffle=True, solver='lbfgs', tol=0.0001,
              validation_fraction=0.1, verbose=False, warm_start=False)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(accuracy_score(y_test, y_pred))
