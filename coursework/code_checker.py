
from sklearn.metrics import accuracy_score

from solution_example import fit_preprocess, load_and_preprocess, fit_model, \
    predict


train_path = r'https://raw.githubusercontent.com/iraola/ML4CE-AD/main/coursework/data/data_train.csv'

preprocess_params = fit_preprocess(train_path)
X_train, y_train = load_and_preprocess(train_path, preprocess_params)
model = fit_model(X_train)
y_train_pred = predict(X_train, model)

print('Accuracy is: ', accuracy_score(y_train, y_train_pred))
