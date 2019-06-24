import numpy as np
import pandas as pd
import argparse
from argparse import RawDescriptionHelpFormatter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from scipy.stats.stats import pearsonr



def normalize(X_train, X_test):

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        return X_train, X_test

def Random_Forest_Regressor_model(train_features, train_labels, trees, test_features):

        regressor = RandomForestRegressor(n_estimators=trees, random_state=0)
        regressor.fit(train_features, train_labels)

        predicted_labels = regressor.predict(test_features)

        return predicted_labels

def metrics_calc(prediction, actual):

        rmse = np.sqrt(metrics.mean_squared_error(actual, prediction))
        nrmse = rmse / (np.max(actual)-np.min(actual))
        pcc = pearsonr(actual, prediction)

        return rmse, nrmse, pcc



d="Hello There"

parser = argparse.ArgumentParser(description=d, formatter_class=RawDescriptionHelpFormatter)
parser.add_argument("-f", type=str, required=True, default="features.csv",
                         help="(required) provide the path to features.csv file.\n"
                         "The file should have headers and the first column \n"
                         "should contain the indices of the row")
parser.add_argument("-l", type=str, required=True, default="labels.csv",
                         help="(required) provide the path to labels.csv file. \n"
                         "The file should NOT have any header NOR any label")
parser.add_argument("-e", type=int, default=100,
                         help="provide the number of estimators to be used for model building\n"
                         "default value is set to 100")
parser.add_argument("-split", type=float, default=0.7,
                         help="provide the proportion for splitting the training and \n"
                         "test data. Default is set 0.7 for training data.")

parser.add_argument("-out", type=str, default="prediction.csv",
                          help="provide the path to output file for predictions \n"
                          "File containing prediction will be provided only if this is used")

args = parser.parse_args()

df_features = pd.read_csv(args.f)
df_features = df_features.drop(df_features.columns[0], axis=1)

labels = pd.read_csv(args.l, header=None)
labels = np.ravel(labels)

if args.split is None:
    train_prop = 0.7
else:
    train_prop = args.split

if args.e is None:
    estimator = 100
else:
    estimator = args.e

X_train, X_test, y_train, y_test = train_test_split(df_features, labels, train_size = train_prop, random_state=0)

X_train, X_test = normalize(X_train, X_test)

y_pred = Random_Forest_Regressor_model(X_train, y_train, estimator, X_test)

rmse, nrmse, pcc = metrics_calc(y_pred, y_test)


print("Root Mean Square Error:", rmse)
print("Normalized RMSE:", nrmse)
print("Pearson Correlation Coefficient:", pcc)

                                                                                                                 
