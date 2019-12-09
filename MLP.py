import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

df = pd.read_csv('processed_data/data_a_features_label.csv')
df = df.drop([ 'STUDENTID','EfficientlyCompletedBlockA','EfficientlyCompletedBlockB'], axis=1)
df = df.values
scaler = preprocessing.MinMaxScaler()
df_scaled = scaler.fit_transform(df)
data = pd.DataFrame(df_scaled)

labels = pd.read_csv('public_data/data_train_label.csv')
labels = labels.drop(['STUDENTID'], axis=1)

Y = labels['EfficientlyCompletedBlockB']
X = data

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= 0.3, random_state=27)

clf = MLPClassifier(activation='logistic', alpha=1e-05, batch_size='auto',
              beta_1=0.9, beta_2=0.999, early_stopping=False,
              epsilon=1e-08, hidden_layer_sizes=(100, 50),
              learning_rate='constant', learning_rate_init=0.001,
              max_iter=400, momentum=0.9, n_iter_no_change=10,
              nesterovs_momentum=True, power_t=0.5, random_state=1,
              shuffle=True, solver='adam', tol=0.0001,
              validation_fraction=0.1, verbose=False, warm_start=False)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))

# kappa
kappa = cohen_kappa_score(y_test, y_pred)
print('Cohens kappa: %f' % kappa)
# ROC AUC
auc = roc_auc_score(y_test, y_pred)
print('ROC AUC: %f' % auc)
# confusion matrix
matrix = confusion_matrix(y_test, y_pred)
print(matrix)