import matplotlib
matplotlib.use('Agg')
from flask import Flask, render_template
import csv
from flask import request, abort
import os
import pandas as pd
import json
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.special import expit
import numpy as np
from flask import current_app, flash, jsonify, make_response, redirect, url_for
from os import path

# Set the default color cycle
matplotlib.rcParams['axes.prop_cycle'] = matplotlib.cycler(color=['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black']) #'purple', 'pink', 'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold' 

HOST_NAME = "localhost"
HOST_PORT = 3000
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
UPLOAD_FOLDER = "csv"

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    # r.headers['Cache-Control'] = 'public, max-age=0'
    return r
def plot_regression(df, feature, target):
    # load the model from disk
    try:
        os.remove('static/graph.png')
    except:
        pass
    loaded_model = pickle.load(open("model/model.sav", 'rb'))
    predictions = loaded_model.predict(df['encoded_feature'].values.reshape(-1, 1))
    df['predictions'] = predictions
    print(df.head())
    fig = plt.figure()
    ax = sns.scatterplot(data=df, x="encoded_target", y="predictions")
    ax.set_xlabel("actual " + target)
    ax.set_ylabel("predicted " + target)
    ax.legend()
    ax.figure.savefig('static/graph.png')

def plot_classification(df, feature, target):
    # load the model from disk
    try:
        os.remove('static/graph.png')
    except:
        pass
    loaded_model = pickle.load(open("model/model.sav", 'rb'))
    predictions = loaded_model.predict(df['encoded_feature'].values.reshape(-1, 1))
    df['predictions'] = predictions
    df['predictions'] = df['predictions'].apply(lambda value: "correct prediction" if value == 1 else "incorrect prediction")
    fig = plt.figure()
    ax = sns.countplot(x=df["predictions"])
    ax.set_xlabel(target + " Predictions")
    ax.figure.savefig('static/graph.png')

def plot_clustering(df, feature, target):
    # load the model from disk
    try:
        os.remove('static/graph.png')
    except:
        pass
    loaded_model = pickle.load(open("model/model.sav", 'rb'))
    predictions = loaded_model.predict(df[['encoded_feature', 'encoded_target']])
    df['predictions'] = predictions    
    colors = np.array(['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'purple', 'pink', 'brown', 'orange', 'teal', 'coral', 'lightblue', 'lime', 'lavender', 'turquoise', 'darkgreen', 'tan', 'salmon', 'gold'])
    number_unique_values = len(df[feature].unique().tolist())
    fig = plt.figure()
    ax = sns.scatterplot(data=df, x=feature, y=target, hue="predictions")
    ax.set_xlabel(feature)
    ax.set_ylabel(target)
    ax.figure.savefig('static/graph.png')


def plot_svm(df, feature, target):
    # load the model from disk
    loaded_model = pickle.load(open("model/model.sav", 'rb'))
    predictions = loaded_model.predict_proba(df['encoded_feature'].values.reshape(-1, 1))
    predictions = [value[1] for value in predictions]
    # print(predictions)
    # df['predictions'] = predictions

    loss = expit(df['encoded_feature'].values * loaded_model.coef_ + loaded_model.intercept_).ravel()
    df['loss'] = loss
    # print([value[0] for loaded_model.predict_proba(np.arange(-250, 250).reshape(-1, 1))])
    # plt.pyplot.plot(np.arange(-500, 500), [value[0] for value in loaded_model.predict_proba(np.arange(-500, 500).reshape(-1, 1))])
    # plt.pyplot.show()
    # ax.pyplot.savefig('graph/graph1.png')
    
    # plt.plot(X_test, loss, label="Logistic Regression Model", color="red", linewidth=3)

    ax = sns.scatterplot(data=df, x="encoded_feature", y="encoded_target")
    # sns.lineplot(data=df, x='encoded_feature', y='loss')
    plt.pyplot.plot(np.arange(-500, 500), [value[0] for value in loaded_model.predict_proba(np.arange(-500, 500).reshape(-1, 1))])
    
    ax.set_xlabel(feature)
    ax.set_ylabel(target)
    ax.figure.savefig('static/graph.png')
    # print(result)


def plot_logistic(df, feature, target):
    # load the model from disk
    loaded_model = pickle.load(open("model/model.sav", 'rb'))
    predictions = loaded_model.predict_proba(df['encoded_feature'].values.reshape(-1, 1))
    predictions = [value[1] for value in predictions]
    # print(predictions)
    # df['predictions'] = predictions

    loss = expit(df['encoded_feature'].values * loaded_model.coef_ + loaded_model.intercept_).ravel()
    df['loss'] = loss
    print(loss)
    # print([value[0] for loaded_model.predict_proba(np.arange(-250, 250).reshape(-1, 1))])
    # plt.pyplot.plot(np.arange(-500, 500), [value[0] for value in loaded_model.predict_proba(np.arange(-500, 500).reshape(-1, 1))])
    # plt.pyplot.show()
    # ax.pyplot.savefig('graph/graph1.png')
    
    # plt.plot(X_test, loss, label="Logistic Regression Model", color="red", linewidth=3)

    ax = sns.scatterplot(data=df, x="encoded_feature", y="encoded_target")
    # sns.lineplot(data=df, x='encoded_feature', y='loss')
    plt.pyplot.plot(np.arange(-500, 500), [value[0] for value in loaded_model.predict_proba(np.arange(-500, 500).reshape(-1, 1))])
    
    ax.set_xlabel(feature)
    ax.set_ylabel(target)
    ax.figure.savefig('static/graph.png')


def kmeans(df):
    elbow = KElbowVisualizer(KMeans(), k=(1, 10))
    elbow.fit(df[['encoded_feature', 'encoded_target']])
    print(elbow.elbow_value_)
    kmeans = KMeans(n_clusters = elbow.elbow_value_, init ='k-means++').fit(df[["encoded_feature", "encoded_target"]])
    filename = 'model.sav'
    os.remove('model/model.sav')
    pickle.dump(kmeans, open("model/" + filename, 'wb'))

def knn(df):
    X = df["encoded_feature"].values.reshape(-1, 1)
    y = df["encoded_target"].values.reshape(-1, 1)
    knn = KNeighborsClassifier(n_neighbors=3).fit(X, y)
    filename = 'model.sav'
    pickle.dump(knn, open("model/" + filename, 'wb'))

def svm_classifier(df):
    X = df["encoded_feature"].values.reshape(-1, 1)
    y = df["encoded_target"].values.reshape(-1, 1)
    svm = SVC().fit(X, y)
    filename = 'model.sav'
    pickle.dump(svm, open("model/" + filename, 'wb'))

def logistic_regression(df):
    X = df["encoded_feature"].values.reshape(-1, 1)
    y = df["encoded_target"].values.reshape(-1, 1)
    lreg = LogisticRegression(C=1e5).fit(X, y)
    filename = 'model.sav'
    os.remove('model/model.sav')
    pickle.dump(lreg, open("model/" + filename, 'wb'))

def linear_regression(df):
    X = df["encoded_feature"].values.reshape(-1, 1)
    y = df["encoded_target"].values.reshape(-1, 1)
    reg = LinearRegression().fit(X, y)
    filename = 'model.sav'
    pickle.dump(reg, open("model/" + filename, 'wb'))

def delete_nans(df, feature, target):
    df = df[df[feature].notna()]
    df = df[df[target].notna()]
    return df

def preprocess_feature(df, feature, target):
    df = delete_nans(df, feature, target)
    if (df[feature].dtype.name == "object"):
        le = preprocessing.LabelEncoder()
        encoder = le.fit(df[feature])
        df['encoded_feature'] = encoder.transform(df[feature]) 
        df.head()
        
        pickle.dump(encoder, open("preprocess/feature_encoder.sav", 'wb'))
    else:
        df['encoded_feature'] = df[feature]
        
    if (df[target].dtype.name == "object"):
        le = preprocessing.LabelEncoder()
        encoder = le.fit(df[target])
        df['encoded_target'] = encoder.transform(df[target]) 
        df.head()
        print("7obi rak")
        pickle.dump(encoder, open("preprocess/target_encoder.sav", 'wb'))
    else:
        df['encoded_target'] = df[target]
    return df   

def predict(input_feature1, type1, input_feature2=None, type2=None):
    loaded_model = pickle.load(open("model/model.sav", 'rb'))
    if (type1 == "int64" or type1 == "float64"):
        if (type1 == "int64"):
            input_feature1 = int(input_feature1)
        else:
            input_feature1 = float(input_feature1)
        try:
            if input_feature2 is None:
                prediction = loaded_model.predict(np.array([input_feature1]))
            else:
                prediction = loaded_model.predict(np.array([[input_feature1, input_feature2]]))
        except:        
            prediction = loaded_model.predict(np.array([input_feature1]).reshape(-1, 1))
        if (path.exists("preprocess/target_encoder.sav")):
            loaded_target_encoder = pickle.load(open("preprocess/target_encoder.sav", 'rb'))
            return loaded_target_encoder.inverse_transform(prediction)[0]
        else:
            return prediction[0]
    else:
        loaded_feature_encoder = pickle.load(open("preprocess/feature_encoder.sav", 'rb'))
        try:
            input_feature1 = loaded_feature_encoder.transform(np.array([input_feature1]))
            if input_feature2 is None:
                prediction = loaded_model.predict(input_feature1)
            else:
                prediction = loaded_model.predict(input_feature1)
        except:     
            prediction = loaded_model.predict(input_feature1.reshape(-1, 1))
        if (path.exists("preprocess/target_encoder.sav")):
            loaded_target_encoder = pickle.load(open("preprocess/target_encoder.sav", 'rb'))
            return loaded_target_encoder.inverse_transform(prediction)[0]
        else:
            return prediction[0]  

@app.route('/predict/<model>/<feature>/<target>', defaults={'input_feature1': None, 'input_feature2': None}, methods=["GET"])
@app.route('/predict/<model>/<feature>/<target>/<input_feature1>', defaults={'input_feature2': None}, methods=["POST"])
@app.route("/predict/<model>/<feature>/<target>/<input_feature1>/<input_feature2>", methods=["POST"])
def predict_value(model, feature, target, input_feature1, input_feature2):
    if request.method == "GET":     
        if (model == "unsupervised"):
            return render_template("prediction.html", predict=False, model=True, feature=feature, target=target)
        else:
            return render_template("prediction.html", predict=False, model=False, feature=feature, target=None)
    else:
        file_name = UPLOAD_FOLDER + "/" + os.listdir(UPLOAD_FOLDER)[0]
        df = pd.read_csv(file_name)
        df = preprocess_feature(df, feature, target) 
        # input_forms = json.loads(request.data.decode("utf-8"))
        # input_feature1 = input_forms["input_feature1"]
        type1 = df[feature].dtype
        if (model == "unsupervised"):   
            # input_feature2 = input_forms["input_feature2"]
            type2 = df[target].dtype

            if ((type1 == "object" and input_feature1 not in df[feature].unique().tolist()) or (type2 == "object" and input_feature2 not in df[target].unique_values)):
                return jsonify({'error' : 'Error!'}), 404
            result = {'prediction': f"{predict(input_feature1, type1, input_feature2, type2)}" } 
        else:
            if (type1 == "object" and input_feature1 not in df[feature].unique().tolist()):
                return jsonify({'error' : 'Error!'}), 404
            result = {'prediction': f"{predict(input_feature1, type1)}" }  
        
        return jsonify(result)


@app.route("/train", methods=["POST"])
def train():
    if request.method == "POST":
        input_forms = json.loads(request.data.decode("utf-8"))
        target = input_forms["target"]
        print(target)
        feature = input_forms["feature"]
        model = input_forms["model"]
        # input_feature1 = input_forms["input_feature1"]
        
        # input_feature2 = input_forms["input_feature2"]
        
        file_name = UPLOAD_FOLDER + "/" + os.listdir(UPLOAD_FOLDER)[0]
        df = pd.read_csv(file_name)
        # type1 = df[feature].dtype
        # type2 = df[target].dtype
        # print(type1)
        # print(type2)
        df = preprocess_feature(df, feature, target)        
        is_unsupervised = False
        if (model == "linear_regression"):
            linear_regression(df)
            plot_regression(df, feature, target)
            
        if (model == "logistic_regression"):
            logistic_regression(df)
            plot_classification(df, feature, target)
            
        if (model == "svm"):
            svm_classifier(df)
            plot_classification(df, feature, target)

        if (model == "knn"):
            knn(df)
            plot_classification(df, feature, target)

        if (model == "kmeans"):
            is_unsupervised = True
            kmeans(df)
            plot_clustering(df, feature, target)

        if is_unsupervised == True:
            return jsonify({'url': "/predict" + f"/unsupervised" + f"/{feature}" + f"/{target}"})
        else:
            return jsonify({'url': "/predict" + f"/supervised" + f"/{feature}" + f"/{target}"})


@app.route("/", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        csv_file = request.files["csv"] # name provided in index.html
        if csv_file:
            for file_name in os.listdir(UPLOAD_FOLDER):
                file = UPLOAD_FOLDER + "/" + file_name
                os.remove(file)

            csv_location = os.path.join(
                UPLOAD_FOLDER,
                csv_file.filename # give the name of the image .jpg
            )
            csv_file.save(csv_location)
            # generate the prediction
            # pred = predict(image_location, MODEL)[0]
            try:
                os.remove('preprocess/target_encoder.sav')
                os.remove('preprocess/feature_encoder.sav')
            except:
                pass
            with open(csv_location) as file:
                reader = csv.reader(file)
                header = next(reader)
                rows = list(reader)
                # Get only first ten rows
                if (len(rows) >= 10):
                    rows = rows[:10]

                return render_template("home.html", header=header, rows=rows)

            # return render_template("index.html", prediction=pred, image_loc=image_file.filename)        
    return render_template("index.html", header=None, rows=None)

 
if __name__ == "__main__":
  app.run(HOST_NAME, HOST_PORT)