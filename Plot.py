import sqlite3
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)

from flask import Flask, render_template
app = Flask(__name__, template_folder='./')

@app.route("/")
@app.route("/index")
def show_index():
    conn =sqlite3.connect('Healthcare_data.db')
    result = conn.execute("Select age, avg(chol) from heart group by age")
    Age = []
    Chol = []
    for row in result:
        Age.append(int(row[0]))
        Chol.append(row[1])
    x1 = np.array(Age)
    y1 = np.array(Chol)
    
    # estimating coefficients
    X1 = x1[:, np.newaxis]
    model.fit(X1,y1)
    print(model.coef_)
    print(model.intercept_)
    xfit1 = np.linspace(x1[0], x1[len(x1)-1]*1.1)
    Xfit1 = xfit1[:, np.newaxis]
    yfit1 = model.predict(Xfit1)
    
    #b1 = estimate_coef(x1, y1)
    #print("Estimated coefficients:\n
    #b_0 = {} \nb_1 = {}".format(b1[0], b1[1]))
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.xlabel("Age")
    plt.ylabel("Cholosterol")
    plt.title("Age (x) vs Cholosterol (y) level")
    #plt.xticks(rotation=45)
    plt.scatter(x1,y1)
    plt.plot(xfit1, yfit1)
    #plot_regression_line(x1, y1, b1)
 
    #-------------- Age vs Chol
    result = conn.execute("Select age, count(cp) from heart group by age")
    CAge = []
    Cpchest = []
    for row in result:   
        CAge.append(int(row[0]))
        Cpchest.append(row[1])
    x2 = np.array(CAge)
    y2 = np.array(Cpchest)

    # estimating coefficients
    X2 = x2[:, np.newaxis]
    model.fit(X2,y2)
    print(model.coef_)
    print(model.intercept_)
    xfit2 = np.linspace(x2[0], x2[len(x2)-1]*1.1)
    Xfit2 = xfit2[:, np.newaxis]
    yfit2 = model.predict(Xfit2)
    
    conn.close()

    plt.subplot(2, 1, 2)
    plt.xlabel("Age")
    plt.ylabel("Chest Pain")
    plt.title("Age (x) vs chest pains (y)")

    plt.scatter(x2,y2)
    plt.plot(xfit2, yfit2)
    #plt.xticks(rotation=45)

    plt.savefig('static/plot.png')

    return render_template('index.html')

#-----------------------------
@app.route("/plot1")
def plot1():
    print("request plot1: Age (x) vs Cholosterol (y) level")
    conn =sqlite3.connect('Healthcare_data.db')
    result = conn.execute("Select age, avg(chol) from heart group by age")
    Age = []
    Chol = []
    for row in result:
        Age.append(int(row[0]))
        Chol.append(row[1])
    x1 = np.array(Age)
    y1 = np.array(Chol)

    # estimating coefficients
    X1 = x1[:, np.newaxis]
    model.fit(X1,y1)
    print(model.coef_)
    print(model.intercept_)
    xfit1 = np.linspace(x1[0], x1[len(x1)-1]*1.1)
    Xfit1 = xfit1[:, np.newaxis]
    yfit1 = model.predict(Xfit1)
    
    conn.close()
    plt.clf()
    plt.xlabel("Age")
    plt.ylabel("Cholosterol")
    plt.title("Age (x) vs Cholosterol (y) level")
    plt.xticks(rotation=45)
    plt.scatter(x1,y1)
    plt.plot(xfit1, yfit1)
    plt.savefig('static/plot1.png')
    plt.clf()
    return render_template('plot1.html')
    
@app.route("/plot2")
def plot2():
    print("request plot2: Age (x) vs chest pains (y)")
    conn =sqlite3.connect('Healthcare_data.db')

    #-------------- Age vs Chol
    result = conn.execute("Select age, count(cp) from heart group by age")
    CAge = []
    Cpchest = []
    for row in result:   
        CAge.append(int(row[0]))
        Cpchest.append(row[1])
    x2 = np.array(CAge)
    y2 = np.array(Cpchest)

    # estimating coefficients
    X2 = x2[:, np.newaxis]
    model.fit(X2,y2)
    print(model.coef_)
    print(model.intercept_)
    xfit2 = np.linspace(x2[0], x2[len(x2)-1]*1.1)
    Xfit2 = xfit2[:, np.newaxis]
    yfit2 = model.predict(Xfit2)   
    conn.close()
    
    plt.clf()
    plt.xlabel("Age")
    plt.ylabel("Chest Pains")
    plt.title("Age (x) vs chest pains (y)")
    plt.xticks(rotation=45)
    plt.scatter(x2,y2)
    plt.plot(xfit2, yfit2)
    plt.savefig('static/plot2.png')
    plt.clf()
    return render_template('plot2.html')
     

if __name__ == "__main__":
   app.run() 
