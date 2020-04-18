import datetime
import pandas as pd
import numpy as np
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import storage
import pyrebase

from datetime import date, timedelta
import urllib.request, json 
import time 
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import os
import csv
from IPython.display import display
from Model import trainData
import random
#from google.cloud import storage

from matplotlib.patches import Ellipse
import seaborn as sns

# signal processing
from scipy import signal
from scipy.ndimage import label
from scipy.stats import zscore
from scipy.interpolate import interp1d
from scipy.integrate import trapz

# misc
import warnings

#generate pdf
from reportlab.pdfgen import canvas
from reportlab.lib.colors import Color, lightblue, black


cred = credentials.Certificate("serene_firebase.json")
app = firebase_admin.initialize_app(cred ,  {
    'storageBucket': 'serene-2dfd6.appspot.com',
}, name='[DEFAULT]')
db = firestore.client()


#REST API:
from flask import Flask
app = Flask(__name__)#referance the file
@app.route("/home/<string:id>/<int:du>")

def hello(id , du):
    duration= du
    userID = id #UqTdL3T7MteuQHBe1aNfSE9u0Na2


    dates =[]
    for x in range(0 ,duration):
        today=date.today() 
        yesterday = today - datetime.timedelta(days=1)
        start_date = (yesterday-timedelta(days=duration-x)).isoformat()
        dates.append(start_date)

    today = datetime.datetime.now()
    timestamp = today.strftime("%Y-%m-%d %H:%M:%S")
    bucket = storage.bucket(app=app)


    df= pd.DataFrame()
    # loop through the storage and get the data
    sleep =[]
    for x in range(0 ,len(dates)):
        #Sleep
        blob = bucket.blob(userID+"/fitbitData/"+dates[x]+"/"+dates[x]+"-sleep.json")
        # download the file 
        u = blob.generate_signed_url(datetime.timedelta(seconds=300), method='GET')
        try:
            with urllib.request.urlopen(u) as url:
                data = json.loads(url.read().decode())
                sleepMinutes = data['summary']["totalMinutesAsleep"]
        except:
            pass

        #Activity (Steps)
        blob = bucket.blob(userID+"/fitbitData/"+dates[x]+"/"+dates[x]+"-activity.json")
        # download the file 
        u = blob.generate_signed_url(datetime.timedelta(seconds=300), method='GET')
        try:
            with urllib.request.urlopen(u) as url:
                data = json.loads(url.read().decode())
                steps = data['summary']["steps"]
        except:
            pass

        #heartrate
        blob = bucket.blob(userID+"/fitbitData/"+dates[x]+"/"+dates[x]+"-heartrate.json")
        u = blob.generate_signed_url(datetime.timedelta(seconds=300), method='GET')
        try:
            with urllib.request.urlopen(u) as url:
                data = json.loads(url.read().decode())
                df_heartrate = pd.DataFrame(data['activities-heart-intraday']['dataset'])

            df_heartrate.time.apply(str)
            df_heartrate['time'] = pd.to_datetime(df_heartrate['time'])
            df_heartrate['hour'] = df_heartrate['time'].apply(lambda time: time.strftime('%H'))
            df_heartrate.drop(['time'],axis=1, inplace = True)
            heart_rate = df_heartrate.groupby(["hour"], as_index=False).mean()
            heart_rate['sleepMin'] = sleepMinutes
            heart_rate['TotalSteps'] = steps
            heart_rate['date'] = dates[x]
            heart_rate = heart_rate.astype({"hour": int})  
        except:
            pass

        # append dataframe
        df = df.append(heart_rate, ignore_index = True)

    # get location from database
    loc_df = pd.DataFrame()
    locID = []
    locations = db.collection(u'PatientLocations').where(u'patientID', u'==', userID ).stream()

    for location in locations:
        loc = location.to_dict()
        locID.append(location.id)
        loc_df = loc_df.append(pd.DataFrame(loc,index=[0]),ignore_index=True)

    loc_df['id'] = locID


    loc_df.drop(['anxietyLevel', 'lat','lng', 'patientID'  ], axis=1, inplace = True)

    loc_df.time.apply(str)
    loc_df['time'] = pd.to_datetime(loc_df['time'])
    loc_df['date'] = pd.to_datetime(loc_df['time'], format='%Y:%M:%D').dt.date
    loc_df['hour'] = loc_df['time'].apply(lambda time: time.strftime('%H'))
    loc_df.drop(['time'], axis=1, inplace = True)
    loc_df.hour = loc_df.hour.astype(int) 
    loc_df.date = loc_df.date.astype(str)
    df.date = df.date.astype(str)

    dfinal = pd.merge(left=df, 
                      right = loc_df,
                      how = 'left',
                      left_on=['hour','date'],
                      right_on=['hour','date']).ffill()

    #test model 
    train_df = dfinal.rename(columns={'value': 'Heartrate'})

    Labeled_df = pd.DataFrame()
    Labeled_df = trainData(train_df)

    Labeled_df.drop(['lon'],axis=1, inplace = True)

    # Replace missing values because it doesn't exist
    Labeled_df['name'].fillna("Not given", inplace=True)
    Labeled_df['id'].fillna("Not given", inplace=True)

    # Update firebase with the user anxiety level 
    for row in Labeled_df.itertuples():
        if row.id != 'Not given':
            if row.Label == 'Low' or row.Label == 'LowA':
                anxietyLevel = 1
            elif row.Label == 'Meduim':
                anxietyLevel = 2
            else:
                anxietyLevel = 3 
            doc_ref = db.collection(u'PatientLocations').document(row.id)
            doc_ref.update({
                                u'anxietyLevel':anxietyLevel
                         })

    # Show the highest level 
    df_high = pd.DataFrame()
    df_high = Labeled_df[Labeled_df.Label == 'High']

    docDf = pd.DataFrame()
    doc_ref = db.collection(u'Patient').document(userID)
    doc = doc_ref.get().to_dict()
    docDf = docDf.append(pd.DataFrame(doc,index=[0]),ignore_index=True)

    age1 = docDf['age'].values
    name1 = docDf['name'].values
    emp1 = docDf['employmentStatus'].values
    mar1 = docDf['maritalStatus'].values
    income1 = docDf['monthlyIncome'].values
    chronicD1 = docDf['chronicDiseases'].values
    smoke1 = docDf['smokeCigarettes'].values
    gad1 = docDf['GAD-7ScaleScore'].values

    age = age1[0] 
    name = name1[0]
    emp = emp1[0]
    mar = mar1[0]
    income = income1[0]
    chronicD = chronicD1[0]
    smoke = smoke1[0]
    gad = gad1[0]


    compareAge = int(age)

    sleepMin = Labeled_df['sleepMin'].mean()
    totalSteps = Labeled_df['TotalSteps'].mean()

    sleepRecomendation = False
    stepsRecomendation = False
    recomendedSteps = 'No recomendation'

    if sleepMin < 360:
        sleepRecomendation = True
    if compareAge < 20 and compareAge > 11:
        if totalSteps < 6000:
            stepsRecomendation = True
            recomendedSteps = '6000'
    if compareAge < 66 and compareAge > 19:  
         if totalSteps < 3000:
            stepsRecomendation = True
            recomendedSteps = '3000'

    sleepMin = sleepMin / 60


    float("{:.2f}".format(sleepMin))
    float("{:.2f}".format(totalSteps))

    # store recomendation in database
    ID = random.randint(1500000,10000000)
    doc_rec = db.collection(u'LastGeneratePatientReport').document(str(ID))
    doc_rec.set({
        u'steps': totalSteps,
        u'patientID':userID,
        u'sleepMin': sleepMin,
        u'sleepRecomendation': sleepRecomendation,
        u'stepsRecomendation': stepsRecomendation,
        u'recommended_steps': recomendedSteps
    })

    firebaseConfig = {
        "apiKey": "AIzaSyBoxoXwFm9TuFysjQYag0GB1NEPyBINlTU",
        "authDomain": "serene-2dfd6.firebaseapp.com",
        "databaseURL": "https://serene-2dfd6.firebaseio.com",
        "projectId": "serene-2dfd6",
        "storageBucket": "serene-2dfd6.appspot.com",
        "messagingSenderId": "461213981433",
        "appId": "1:461213981433:web:62428e3664182b3e58e028",
        "measurementId": "G-J66VP2Y3CR"
      }

    firebase = pyrebase.initialize_app(firebaseConfig)
    storage = firebase.storage()

    # Change Label values to num, to represent them in a barchart
    nums=[]
    for row in Labeled_df.itertuples():
        if row.Label == 'Low' or row.Label == 'LowA':
            nums.append(1)
        elif row.Label == 'Meduim':
            nums.append(2)
        else:
            nums.append(3)
    Labeled_df['numLabel'] = nums


    # Get anxiety level by day and store it in a new data frame
    plot_df = pd.DataFrame()
    avgAnxiety = []
    totalAnxiety = 0
    rowCount = 1
    for x in range(0 ,len(dates)):
        for row in Labeled_df.itertuples():
            if (row.date == dates[x]):
                rowCount += 1
                totalAnxiety += row.numLabel
        avgAnxiety.append(totalAnxiety/rowCount)


    plot_df['date'] = dates
    plot_df['Anxiety'] = avgAnxiety


    #divide dataframe into 15 rows (2 weeks)

    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    df3 = pd.DataFrame()
    df4 = pd.DataFrame()
    df5 = pd.DataFrame()
    df6 = pd.DataFrame()
    df7 = pd.DataFrame()
    df8 = pd.DataFrame()
    df9 = pd.DataFrame()
    df10 = pd.DataFrame()
    df11 = pd.DataFrame()
    df12 = pd.DataFrame()
    dfarray = []
    count = 0
    if(len(plot_df) > 15):
        df1 = plot_df[:15]
        df2 = plot_df[15:]
        dfarray.append(df1)
        dfarray.append(df2)
        if(len(df2)>15):
            count = (df2.last_valid_index() - (len(df2) - 15))
            df3 = df2[count:]
            dfarray.append(df3)
            if(len(df3)>15):
                count = (df3.last_valid_index() - (len(df3) - 15))
                df4 = df3[count:]
                dfarray.append(df4)
                if(len(df4)>15):
                    count = (df4.last_valid_index() - (len(df4) - 15))
                    df5 = df4[count:]
                    dfarray.append(df5)
                    if(len(df5)>15):
                        count = (df5.last_valid_index() - (len(df5) - 15))
                        df6 = df5[count:]
                        dfarray.append(df6)
                        if(len(df6)>15):
                            count = (df6.last_valid_index() - (len(df6) - 15))
                            df7 = df6[count:]
                            dfarray.append(df7)
                            if(len(df7)>15):
                                count = (df7.last_valid_index() - (len(df7) - 15))
                                df8 = df7[count:]
                                dfarray.append(df8)
                                if(len(df8)>15):
                                    count = (df8.last_valid_index() - (len(df8) - 15))
                                    df9 = df8[count:]
                                    dfarray.append(df9)
                                    if(len(df9)>15):
                                        count = (df9.last_valid_index() - (len(df9) - 15))
                                        df10 = df9[count:]
                                        dfarray.append(df10)
                                        if(len(df10)>15):
                                            count = (df10.last_valid_index() - (len(df10) - 15))
                                            df11 = df10[count:]
                                            dfarray.append(df11)
                                            if(len(df11)>15):
                                                count = (df11.last_valid_index() - (len(df11) - 15))
                                                df12 = df11[count:]
                                                dfarray.append(df12)


    # Plot AL
    for x in range(0,len(dfarray)):
        fig, ax = plt.subplots()

        # Draw the stem and circle
        ax.stem(dfarray[x].date, dfarray[x].Anxiety, basefmt=' ')
        plt.tick_params(axis='x', rotation=70)

        # Start the graph at 0
        ax.set_ylim(0, 3)
        ax.set_title('Anxiety level (Throughout week)')
        plt.xlabel('Date')
        plt.ylabel('Low        Meduim        High', fontsize= 12)
        ax.yaxis.set_label_coords(-0.1, 0.47)



        conv = str(x)
        fig.savefig('AL'+str(x)+'.png', dpi = 100)
        imagePath = 'AL'+str(x)+'.png'
        storage.child(userID+"/lastGeneratedPatientReport/AL"+str(x)+'.png').put('AL'+str(x)+'.png')
        os.remove('AL'+str(x)+'.png')

if __name__ == "__main__":
    app.run(debug=True)
