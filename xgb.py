import pandas as pd
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier


# Seperate training and testing label from training and testing dataset
def seperateLabels(dataset):
    labels = dataset.pop('Survived')

    print('===Labels===')
    print(len(labels))

    return dataset, labels

# Convert categorical columns to one-hot
def oneHot(dataset):
    print("===one hot encoding===")
    #dataset = dataset.dropna()
    dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')
    print(len(dataset.index))
    print(dataset.isnull().sum().sum())
    return dataset



# Cross validation
def crossVal(params, X, y):
    # define model
    classifier = XGBClassifier(**params)
    # cross validation
    scores = cross_val_score(classifier, X, y, cv=5,
                             scoring='accuracy', n_jobs=-1, verbose=1)
    print("Accuracy: %f (+/- %f)" % (scores.mean(), scores.std() * 2))



# train the model for final evaluation
def final_eval(params, X, y, test_dataset):
    # define model
    classifier = XGBClassifier(**params)
    classifier.fit(X, y)
    # predict
    pred = classifier.predict(test_dataset)
    # print the pred
    print(pred)
    #save final format
    test_dataset['Survived'] = pred 
    result = pd.concat([test_dataset['PassengerId'], test_dataset['Survived']], axis=1)
    print(result)
    result.to_csv('data/result.csv', index=False)


def main():
    print('Load Data')
    raw_dataset = pd.read_csv(
        'data/cleaned_train.csv', skipinitialspace=True, verbose=True)
    train_dataset = raw_dataset.copy()
    raw_dataset = pd.read_csv(
        'data/cleaned_test.csv', skipinitialspace=True, verbose=True)
    test_dataset = raw_dataset.copy()
    print(len(train_dataset.index))
    print(len(test_dataset.index))

    # get X and y
    X, y = seperateLabels(train_dataset)

    # one-hot encoding
    X = oneHot(X)
    test_dataset = oneHot(test_dataset)
    con_X = pd.concat([X, test_dataset])
    test_X = con_X.iloc[len(X.index):]
    X = con_X.iloc[:len(X.index)]
    print(X.info())
    print(test_X.info())

    #deal with final na values
    X = X.fillna(0)
    test_X = test_X.fillna(0)

    # parameters
    params = {}
    params['n_estimators'] = 1000
    params['max_depth'] = 500
    params['eta'] = 0.1
    params['gamma'] = 3
    params['min_child_weight'] = 0
    params['subsample'] = 0.8
    params['colsample_bytree'] = 1
    params['colsample_bylevel'] = 1
    params['colsample_bynode'] = 0.5

    # cross validation
    crossVal(params, X, y)

    #final evaluation
    #final_eval(params, X, y, test_X)


if __name__ == "__main__":
    main()
