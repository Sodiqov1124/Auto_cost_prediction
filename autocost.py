import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
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
def show_with_heatmap(data):
    plt.figure(figsize=(14, 12))
    ddf = dataset.copy()
    ddf = ddf.corr()
    sns.heatmap(ddf, annot=True, cmap='YlGnBu', cbar_kws={"orientation": "horizontal"})
    plt.show()
show_with_heatmap(dataset)
def set_style(data):
    # plt.figure(figsize=(15, 10))
    sns.displot(data=dataset, x=dataset['price'], kde=True)
    plt.show()

set_style(dataset)
def new_value(data):
    predict = "price"
    data = data[["fueltypes","wheelbase", "carlength",
                 "carwidth", "carheight", "curbweight",
                 "cylindernumber", "enginelocation", "aspiration",
                 "compressionratio", "horsepower", "peakrpm",
                 "citympg", "highwaympg", "price"]]
    x = np.array(data.drop([predict], axis = 1))
    y = np.array(data[predict])
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.2)
    model = DecisionTreeRegressor()
    model.fit(xtrain, ytrain)
    predictions = model.predict(xtest)
    print("Model quyidagi aniqlikda ishlamoqda:",round(r2_score(ytest, predictions),2))
    model.score(xtest, predictions)
    Nissan_X_Trail_HP_4_x2_Visia_Engine = np.array([1,270.6,464.3,182.0,169.5,1505,4,0,0,10.5,163,5600,30,44]).reshape(1, -1)
    malibu2 = np.array([1,111.4,193.8,73,57.6,3184,4,0,0,9.5,160,5700,25,34]).reshape(1, -1)
    pred_car = model.predict(malibu2)
    print("Bu avtomobilning  taxminiy narxi bunday:$",round(pred_car[0]))
new_value(dataset)
