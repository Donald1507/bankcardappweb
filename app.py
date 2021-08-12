#Importing the libraries and modules
import csv
from os import sep
import pickle
import numpy as np
import pandas as pd
import sqlite3
from flask import Flask, request, jsonify, render_template
from pandas.core.frame import DataFrame

#Define the Flask application name
app = Flask(__name__)

#Loading the ML model
model = pickle.load(open('model.pkl', 'rb'))

#List of the features
features_list = ['Gender','Marital_Status','Income_Category','Card_Category','Months_Inactive_12_mon','Avg_Utilization_Ratio','Total_Relationship_Count']

#Encoding function for "object" type features of the dataframe
def encodage(dataframe):
    code = {
        'Attrited Customer':1, 'Existing Customer':0,
        'F':0, 'M':1,
        'Single':0, 'Married':1, 'Divorced':2, 
        'Less than $40K':0, '$40K - $60K':1,'$60K - $80K':2, '$80K - $120K':3, '$120K +':4,
        'Blue':0, 'Silver':1, 'Gold':2, 'Platinum':3,
        'Uneducated':0, 'College':1, 'High School':2, 'Graduate':3, 'Post-Graduate':4, 'Doctorate':5
    }
    for col in dataframe.select_dtypes('object').columns:
        dataframe.loc[:,col] = dataframe[col].map(code)
    
    return dataframe

#Route for "index.html" page
@app.route("/")
def home():
    return render_template("index.html", title="Accueil")


#Route for "one_pred.html" page    
@app.route("/one_pred/", methods=['GET','POST'])
def one_prediction():
    
    if request.method == 'GET':
        return render_template('one_pred.html')
    
    elif request.method == 'POST':
    
        #For rendering results on HTML GUI
        features_values = [float(x) for x in request.form.values()]
        #Putting the values of the form in a dataframe
        final_features = pd.DataFrame(features_values, index = features_list).T
        #Using the model for predicting
        prediction = model.predict(final_features)
        prediction = prediction[0]

        #Final messages after predication
        if prediction == 'Existing Customer':
            output = 'Le client restera abonné au service de la carte bancaire.'
        else:
            output = 'Le client envisage de mettre fin au service de la carte bancaire.'

        return render_template("one_pred.html", prediction_text='{}'.format(output))


#Route for "file_pred.html" page
@app.route('/file_pred', methods=['GET', 'POST'])
def file_prediction():
    
    if request.method == 'GET':
        return render_template('file_pred.html')

    elif request.method == 'POST':
        
        #For rending file on HTML
        file = request.files['chosen_file']
        
        #Opening as a dataframe ("data") when the file is a csv file 
        data = pd.read_csv(file, sep=";")
        #New dataframe ("df") with specific columns for prediction
        df = data[['CLIENTNUM','Gender','Marital_Status','Income_Category','Card_Category','Months_Inactive_12_mon','Avg_Utilization_Ratio','Total_Relationship_Count','Attrition_Flag']]
        #Columns of "df" dataframe
        fieldnames = [item for item in df.columns]
        
        df1 = df.T #Transpose the "df" dataframe for taking easily a client information with pandas library
        
        new_results=[]
        for i in range(df1.shape[1]):
            new_results.append(df1[i].tolist()) #Put each row (clint information) in "new_results" list
        
        
        df_new = df.drop('CLIENTNUM', axis=1) #remove the 'CLIENTNUM' column (useless column)
        df_new = df_new.drop('Attrition_Flag', axis=1)   #remove the 'Attrition_Flag' column (target) 
        df_new =  encodage(df_new)              #encoding the dataframe of features only ("df_new")
        prediction = model.predict(df_new)      #predict the "df_new" dataframe
        
        return render_template('file_pred.html', new_results=new_results, fieldnames=fieldnames, prediction=prediction, len= len)
 
 
#Route for "final_file_pred.html" page    
@app.route('/final_file_pred', methods = ['POST'])
def final_file_prediction():
    
    #For rending clientnum of selected clients of "file_pred.html"
    client_num_list = request.form.getlist('client_num') #list of strings
    client_num_list = list(map(int, client_num_list))   #convert to list of int
    
    #rending all the informations of the clients (type: string with list inside)
    data = request.form.get('all_data')  
    #bring out the list inside the string
    data = eval(data)                     
    #create a dataframe with all data and specific columns
    data = pd.DataFrame(data, columns=['CLIENTNUM','Gender','Marital_Status','Income_Category','Card_Category','Months_Inactive_12_mon','Avg_Utilization_Ratio','Total_Relationship_Count','Attrition_Flag'])

    #create a empty dataframe 
    df = pd.DataFrame(columns=['CLIENTNUM','Gender','Marital_Status','Income_Category','Card_Category','Months_Inactive_12_mon','Avg_Utilization_Ratio','Total_Relationship_Count','Attrition_Flag'])
    
    #Adding informations of selected clients in "df" dataframe
    for client_num in client_num_list:
        client_info = data.loc[data['CLIENTNUM'] == client_num]
        df = df.append(client_info,ignore_index=True)
    
    #Using the same process of "file_prediction()" function
       
    columns_df = [item for item in df.columns] 
        
    df1 = df.T      
        
    new_results=[]
    for i in range(df1.shape[1]):
        new_results.append(df1[i].tolist())
                
    df_new = df.drop('CLIENTNUM', axis=1)
    df_new = df_new.drop('Attrition_Flag', axis=1)        
    df_new =  encodage(df_new)
    prediction = model.predict(df_new)
    
    return render_template('final_file_pred.html', new_results=new_results, fieldnames=columns_df, prediction=prediction, len= len)

#Database connection
connection = sqlite3.connect("bankcard.db")
cursor = connection.cursor()

#Creation of "Client" table
sql="""
    CREATE TABLE IF NOT EXISTS Client (
            CLIENTNUM INT,
            Attrition_Flag TEXT,
            Customer_Age INT,
            Gender TEXT,
            Dependent_count	INT,
            Education_Level	TEXT,
            Marital_Status TEXT,	
            Income_Category	TEXT,
            Card_Category TEXT,	
            Months_on_book INT,
            Total_Relationship_Count INT,	
            Months_Inactive_12_mon INT,	
            Contacts_Count_12_mon INT,	
            Credit_Limit FLOAT,	
            Total_Revolving_Bal	FLOAT,
            Avg_Open_To_Buy	FLOAT,
            Total_Amt_Chng_Q4_Q1 FLOAT,	
            Total_Trans_Amt	INT,
            Total_Trans_Ct	INT,
            Total_Ct_Chng_Q4_Q1	FLOAT,
            Avg_Utilization_Ratio FLOAT
            ) """
            
cursor.execute(sql) #execute the query

#reading the csv file
"""with open('Dataset.csv', 'r') as file: 
    no_records = 1
    for row in file:
        cursor.execute("INSERT INTO Client VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", row.split(";"))
        connection.commit()
        no_records +=1
connection.close()
print("\n{} Enregistrements transférés!".format(no_records))"""
        
#Route for "db_pred.html"                
@app.route('/db_pred', methods=['GET', 'POST'])
def db_prediction():
    
    if request.method == 'GET':
        return render_template('db_pred.html')
    
    elif request.method == 'POST':
        
        #database connection
        connection = sqlite3.connect("bankcard.db")
        cursor = connection.cursor()
        
        #rending "search_count" on HTML 
        search_count = request.form.get('search_count')
        
        #select clients randomly in the database with specific columns       
        cursor.execute("SELECT CLIENTNUM, Gender, Marital_Status, Income_Category, Card_Category, " 
                       "Months_Inactive_12_mon, Avg_Utilization_Ratio, Total_Relationship_Count, Attrition_Flag FROM Client ORDER BY RANDOM() LIMIT " 
                       "{}".format(search_count))
        
        random_query = cursor.fetchall()
        
        random_list = np.array(random_query) 
        
        #Using the same process of "file_prediction()" function
        
        df = pd.DataFrame(random_list, columns=['CLIENTNUM','Gender', 'Marital_Status', 'Income_Category', 'Card_Category', 'Months_Inactive_12_mon', 'Avg_Utilization_Ratio', 'Total_Relationship_Count', 'Attrition_Flag'])
        
        columns_df = [item for item in df.columns]
        
        df1 = df.T
        
        new_results=[]
        for i in range(df1.shape[1]):
            new_results.append(df1[i].tolist())
                
        df_new = df.drop('CLIENTNUM', axis=1)
        df_new = df_new.drop('Attrition_Flag', axis=1)        
        df_new =  encodage(df_new)
        prediction = model.predict(df_new)
    
        return render_template('db_pred.html',new_results=new_results, fieldnames=columns_df, prediction=prediction, len= len)


@app.route('/final_db_pred', methods=['POST'])
def final_db_prediction():
    
    if request.method == 'POST':
        
        #Database connection
        connection = sqlite3.connect("bankcard.db")
        cursor = connection.cursor()
        
        #For rending clientnum of selected clients of "file_pred.html"
        client_num_list = request.form.getlist('client_num')
        client_num_list = list(map(int, client_num_list))
        
        chosen_clients = []
        
        for client_num in client_num_list:
            cursor.execute("SELECT CLIENTNUM, Gender, Marital_Status, Income_Category, Card_Category, " 
                            "Months_Inactive_12_mon, Avg_Utilization_Ratio, Total_Relationship_Count, Attrition_Flag FROM Client WHERE CLIENTNUM=" 
                            "{}".format(client_num)) 
            
            client_info = cursor.fetchone()
            chosen_clients.append(client_info)
            client_info = []
        
        chosen_clients = np.array(chosen_clients)  
        
        #Using the same process of "file_prediction()" function 
        
        df = pd.DataFrame(chosen_clients, columns=['CLIENTNUM','Gender', 'Marital_Status', 'Income_Category', 'Card_Category', 'Months_Inactive_12_mon', 'Avg_Utilization_Ratio', 'Total_Relationship_Count', 'Attrition_Flag'])
        
        columns_df = [item for item in df.columns]
        
        df1 = df.T
        
        new_results=[]
        for i in range(df1.shape[1]):
            new_results.append(df1[i].tolist())
                
        df_new = df.drop('CLIENTNUM', axis=1)
        df_new = df_new.drop('Attrition_Flag', axis=1)        
        df_new =  encodage(df_new)
        prediction = model.predict(df_new)
    
        return render_template('final_db_pred.html',new_results=new_results, fieldnames=columns_df, prediction=prediction, len= len)

connection.commit()
connection.close()


if __name__ == "__main__":
    app.run(debug=True)
    
