#PLEASE WRITE THE GITHUB URL BELOW!
#https://github.com/CRJeong/assignment.git

import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def load_dataset(dataset_path):
	data = pd.read_csv(dataset_path)
	return data


def dataset_stat(dataset_df):
	feats = dataset_df.shape[1]-1
	class0 = len(dataset_df.loc[dataset_df['target'] == 0])
	class1 = len(dataset_df.loc[dataset_df['target'] == 1])

	return feats, class0, class1


def split_dataset(dataset_df, testset_size):
	x = dataset_df.drop(columns="target", axis=1)
	y = dataset_df["target"]
	train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=testset_size)

	return train_x, test_x, train_y, test_y


def decision_tree_train_test(x_train, x_test, y_train, y_test):
	dt = make_pipeline(StandardScaler(), DecisionTreeClassifier())
	dt.fit(x_train, y_train)
	prd = dt.predict(x_test)

	accu = (accuracy_score(prd, y_test))
	prc = precision_score(y_test, prd)
	rcall = recall_score(y_test, prd)

	return accu, prc, rcall


def random_forest_train_test(x_train, x_test, y_train, y_test):
	rft = make_pipeline(StandardScaler(), RandomForestClassifier())
	rft.fit(x_train, y_train)
	prd = rft.predict(x_test)

	accu = (accuracy_score(prd, y_test))
	prc = precision_score(y_test, prd)
	rcall = recall_score(y_test, prd)

	return accu, prc, rcall


def svm_train_test(x_train, x_test, y_train, y_test):
	svm = make_pipeline(StandardScaler(), SVC())
	svm.fit(x_train, y_train)
	prd = svm.predict(x_test)

	accu = (accuracy_score(prd, y_test))
	prc = precision_score(y_test, prd)
	rcall = recall_score(y_test, prd)

	return accu, prc, rcall


def print_performances(acc, prec, recall):
	#Do not modify this function!
	print("Accuracy: ", acc)
	print("Precision: ", prec)
	print("Recall: ", recall)


if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print("Number of features: ", n_feats)
	print("Number of class 0 data entries: ", n_class0)
	print("Number of class 1 data entries: ", n_class1)

	print("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print("\nSVM Performances")
	print_performances(acc, prec, recall)
