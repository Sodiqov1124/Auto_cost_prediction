import pandas as pd
"""Company GM Uzbekistan decided to sell their cars around the world
 and they gave me this project to predict their car's cost
   via compare with wolrd car's character's """
df = pd.read_csv("auto_costs.csv")
print(df.loc[3])

#First I have cleaned this dataset
def clean_data(data):
    df = pd.read_csv(data)
    df = df.drop(columns = "ID").drop(columns = "name")
    df["doornumbers"] = df["doornumbers"].replace("two","2").replace("four", "4")
    df["aspiration"] = df["aspiration"].replace("std", "1").replace("turbo", "0")
    df["fueltypes"] = df["fueltypes"].replace("gas", "0").replace("diesel", "1")
    df["carbody"] = df["carbody"].replace("sedan", "0").replace("hatchback", "1").replace("wagon", "2").replace("hardtop", "3").replace("convertible", "4")
    df["drivewheels"] = df["drivewheels"].replace("fwd", "0").replace("rwd", "1").replace("4wd", "2")
    df["enginelocation"] = df["enginelocation"].replace("front", "0").replace("rear", "1")
    df["enginetype"] = df["enginetype"].replace("ohc","0").replace("dohc","1").replace("rotor","2").replace("ohcv","3")
    df["fuelsystem"] = df["fuelsystem"].replace("1bbl", "0").replace("2bbl", "1").replace("mpfi", "2").replace("mfi","3").replace("4bbl","4").replace("idi","5").replace("spdi","6").replace("spfi","6")
    df["cylindernumber"] = df["cylindernumber"].replace("four", "4").replace("three", "43").replace("six", "6").replace("two", "2").replace("five", "5").replace("eight", "8").replace("twelve", "12")
    return df
dataset = clean_data(data = "auto_costs.csv")

print(dataset.describe())
print(dataset.corr())
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
def show_with_heatmap(data):
    plt.figure(figsize=(14, 12))
    ddf = dataset.copy()
    ddf = ddf.corr()

    sns.heatmap(ddf, annot=True, cmap='YlGnBu', cbar_kws={"orientation": "horizontal"})
    plt.show()
show_with_heatmap(dataset)
def set_style(data):
    sns.set_style("whitegrid")
    plt.figure(figsize=(15, 10))
    sns.distplot(dataset.price)
    plt.show()
set_style(dataset)

from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot as plt
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn import model_selection
from sklearn.model_selection import KFold
def my_print_and_test_models(dt):
    m1 = DecisionTreeClassifier()
    m2 = GaussianNB()
    m3 = KNeighborsClassifier()
    m4 = LogisticRegression(solver='liblinear', multi_class='ovr')
    m5 = LinearDiscriminantAnalysis()
    m6 = SVC(gamma='auto')

    array = dt.values
    X = array[:,0:4]
    y = array[:,4]
    X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size=0.3, random_state=1)
    mr1 = cross_val_score(m1, X_train, Y_train, cv=KFold(n_splits=2), scoring='accuracy')
    mr2 = cross_val_score(m2, X_train, Y_train, cv=KFold(n_splits=2), scoring='accuracy')
    mr3 = cross_val_score(m3, X_train, Y_train, cv=KFold(n_splits=2), scoring='accuracy')
    mr4 = cross_val_score(m4, X_train, Y_train, cv=KFold(n_splits=2), scoring='accuracy')
    mr5 = cross_val_score(m5, X_train, Y_train, cv=KFold(n_splits=2), scoring='accuracy')
    mr6 = cross_val_score(m6, X_train, Y_train, cv=KFold(n_splits=2), scoring='accuracy')
    model1 = DecisionTreeRegressor()
    model1.fit(X_train, Y_train)
    predictions = model1.predict(X_validation)
    print("DecisionTreeRegressor:",model1.score(X_validation, predictions))
    print('%s: %f (%f)' % ('DecisionTree', mr1.mean(), mr1.std()))
    print('%s: %f (%f)' % ('GaussianNB', mr2.mean(), mr2.std()))
    print('%s: %f (%f)' % ('KNeighbors', mr3.mean(), mr3.std()))
    print('%s: %f (%f)' % ('LogisticRegression', mr4.mean(), mr4.std()))
    print('%s: %f (%f)' % ('LinearDiscriminant', mr5.mean(), mr5.std()))
    print('%s: %f (%f)' % ('SVM', mr6.mean(), mr6.std()))




# my_print_and_test_models(dataset)
def new_value(data):
    predict = "price"
    data = data[["fueltypes","wheelbase", "carlength",
                 "carwidth", "carheight", "curbweight",
                 "cylindernumber", "enginelocation", "aspiration",
                 "compressionratio", "horsepower", "peakrpm",
                 "citympg", "highwaympg", "price"]]
    x = np.array(data.drop([predict], axis = 1))
    y = np.array(data[predict])

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)

    from sklearn.tree import DecisionTreeRegressor
    model = DecisionTreeRegressor()
    model.fit(xtrain, ytrain)
    predictions = model.predict(xtest)
    print("Model quyidagi aniqlikda ishlamoqda:",round(r2_score(ytest, predictions),2))
    model.score(xtest, predictions)
    Nissan_X_Trail_HP_4_x2_Visia_Engine = np.array([1,270.6,464.3,182.0,169.5,1505,4,0,0,10.5,163,5600,30,44]).reshape(1, -1)
    malibu2 = np.array([1,111.4,193.8,73,57.6,3184,4,0,0,9.5,160,5700,25,34]).reshape(1, -1)
    pred_car = model.predict(malibu2)
    print("Bu avtomobilning  taxminiy narxi $",round(pred_car[0]))
new_value(dataset)