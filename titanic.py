import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tflearn.data_utils import load_csv
from tflearn.datasets import titanic


def load():
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.max_rows', 100)

    # Download the Titanic dataset
    titanic.download_dataset('titanic_dataset.csv')

    # Load CSV file, indicate that the first column represents labels
    data, labels = load_csv('titanic_dataset.csv', target_column=0, has_header=True,
                            categorical_labels=False, n_classes=2)

    # Make a df out of it for convenience
    df = pd.DataFrame(data, columns=["pclass", "name", "sex", "age", "sibsp", "parch", "ticket", "fare"])

    df = df.drop(columns=['name', 'ticket'])

    # bin ages
    df['age'] = df['age'].astype('float64')

    group_names = ['A', 'B', 'C', 'D', 'E']
    bins = [-1, 17, 35, 50, 65, 1000]
    df['age_group'] = pd.cut(df['age'], bins=bins, labels=group_names)

    res = pd.get_dummies(df['age_group'], prefix='age_group')
    df = pd.concat([df, res], axis=1)
    df = df.drop(columns=['age', 'age_group'])

    # normalize parch
    # df['parch'] = df['parch'].astype('float64')
    # df["parch"] = df["parch"] / df["parch"].max()

    # normalize age
    # df['age'] = df['age'].astype('float64')
    # df["age"] = df["age"] / df["age"].max()

    # normalize fare
    df['fare'] = df['fare'].astype('float64')
    df["fare"] = df["fare"] / df["fare"].max()

    # normalize sibsp
    df['sibsp'] = df['sibsp'].astype('float64')
    df["sibsp"] = df["sibsp"] / df["sibsp"].max()

    # normalize parch
    df['parch'] = df['parch'].astype('float64')
    df["parch"] = df["parch"] / df["parch"].max()

    # convert sex
    df = df.replace(["male", "female"], [0, 1])

    print(df)

    return df, labels


def split():
    # Do a test / train split
    X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.4, random_state=42)

    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)

    print('Total number of rows: %d' % df.shape[0])
    print('Total number of rows in X_train: %d' % X_train.shape[0])
    print('Total number of rows in X_test: %d' % X_test.shape[0])
    return X_train, X_test, y_train, y_test


def trainNN():
    model = tf.keras.models.Sequential()

    # input layer
    model.add(tf.keras.layers.Dense(10, input_shape=(10,)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Dropout(0.3))

    # hidden layers
    model.add(tf.keras.layers.Dense(256))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Dropout(0.3))

    # hidden layers
    model.add(tf.keras.layers.Dense(256))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.keras.layers.Dropout(0.3))

    # hidden layers
    model.add(tf.keras.layers.Dense(2, activation="sigmoid"))

    # output layer
    model.add(tf.keras.layers.Dense(1, activation='linear'))

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train.values, y_train.values, epochs=300)
    model.evaluate(X_test.values, y_test.values)


def trainRandomForest():
    parameters = {'n_estimators': 100}

    model = RandomForestClassifier(**parameters)
    model.fit(X_train.values, y_train.values.ravel())

    RF_predictions = model.predict(X_test)
    score = accuracy_score(y_test, RF_predictions)
    print(score)


df, labels = load()
X_train, X_test, y_train, y_test = split()
trainNN()
trainRandomForest()
