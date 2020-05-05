def reportD(pid): 
    from datetime import datetime, timedelta
    import pandas as pd
    import numpy as np
    import firebase_admin
    from firebase_admin import credentials
    from firebase_admin import firestore
    from firebase_admin import storage
    import pyrebase
    import math

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
    import matplotlib.patches as mpatches
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


    # In[35]:


    if not firebase_admin._apps:
            cred = credentials.Certificate("serene-firebase-adminsdk.json")
            app = firebase_admin.initialize_app(cred ,  {
            'storageBucket': 'serene-2dfd6.appspot.com',
            }, name='[DEFAULT]')
    else:
            app = firebase_admin.get_app()
    db = firestore.client()



    # In[36]:


    duration = 7
    userID = pid


    # In[37]:


    today = datetime.now()
    timestamp = today.strftime("%Y-%m-%d %H:%M:%S")
    bucket = storage.bucket(app=app)


    # ## Get data from storage and get list of dates 

    # In[6]:


    dates =[]
    for x in range(0 ,duration):
        today=date.today() 
        #yesterday = today - datetime.timedelta(days=1)
        start_date = (today-timedelta(days=duration-x)).isoformat()
        dates.append(start_date)


    # In[7]:


    df= pd.DataFrame()
    # loop through the storage and get the data
    sleep =[]
    for x in range(0 ,len(dates)):
        #Sleep
        blob = bucket.blob(userID+"/fitbitData/"+dates[x]+"/"+dates[x]+"-sleep.json")
        # download the file 
        u = blob.generate_signed_url(timedelta(seconds=300), method='GET')
        try:
            with urllib.request.urlopen(u) as url:
                data = json.loads(url.read().decode())
                sleepMinutes = data['summary']["totalMinutesAsleep"]
        except:
            pass

        #Activity (Steps)
        blob = bucket.blob(userID+"/fitbitData/"+dates[x]+"/"+dates[x]+"-activity.json")
        # download the file 
        u = blob.generate_signed_url(timedelta(seconds=300), method='GET')
        try:
            with urllib.request.urlopen(u) as url:
                data = json.loads(url.read().decode())
                steps = data['summary']["steps"]
        except:
            pass

        #heartrate
        blob = bucket.blob(userID+"/fitbitData/"+dates[x]+"/"+dates[x]+"-heartrate.json")
        u = blob.generate_signed_url(timedelta(seconds=300), method='GET')
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


    # ### Get user location

    # In[8]:


    # get location from database
    loc_df = pd.DataFrame()
    locID = []
    locations = db.collection(u'PatientLocations').where(u'patientID', u'==', userID ).stream()

    for location in locations:
        loc = location.to_dict()
        locID.append(location.id)
        loc_df = loc_df.append(pd.DataFrame(loc,index=[0]),ignore_index=True)

    loc_df['id'] = locID



    # In[9]:


    loc_df.drop(['anxietyLevel', 'lat','lng', 'patientID'  ], axis=1, inplace = True)


    # In[10]:


    loc_df.time.apply(str)
    loc_df['time'] = pd.to_datetime(loc_df['time'])
    loc_df['date'] = pd.to_datetime(loc_df['time'], format='%Y:%M:%D').dt.date
    loc_df['hour'] = loc_df['time'].apply(lambda time: time.strftime('%H'))
    loc_df.drop(['time'], axis=1, inplace = True)
    loc_df.hour = loc_df.hour.astype(int) 
    loc_df.date = loc_df.date.astype(str)
    df.date = df.date.astype(str)


    # In[11]:


    dfinal = pd.merge(left=df, 
                      right = loc_df,
                      how = 'left',
                      left_on=['hour','date'],
                      right_on=['hour','date']).ffill()


    # ### Test data into model

    # In[12]:


    #test model 
    train_df = dfinal.rename(columns={'value': 'Heartrate'})


    # In[13]:


    Labeled_df = pd.DataFrame()
    Labeled_df = trainData(train_df)


    # In[14]:


    Labeled_df.drop(['lon'],axis=1, inplace = True)


    # In[15]:


    # Replace missing values because it doesn't exist
    Labeled_df['name'].fillna("Not given", inplace=True)
    Labeled_df['id'].fillna("Not given", inplace=True)


    # ## Storage intilization

    # In[16]:


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


    # # AL

    # In[17]:


    sns.set( rc={'axes.facecolor': '#fcfeff'})


    # In[18]:
    # i added this code here because it will result an error down because of the datetime
    yesterday = today - timedelta(days=1)
    timestamp = yesterday.strftime("%Y-%m-%d")

    from datetime import  datetime


    # In[19]:


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


    # In[20]:


    # Get anxiety level by day and store it in a new data frame
    plot_df = pd.DataFrame()
    avgAnxiety = []
    days = []
    totalAnxiety = 0
    rowCount = 1
    for x in range(0 ,len(dates)):
        totalAnxiety = 0
        rowCount = 1
        for row in Labeled_df.itertuples():
            if (row.date == dates[x]):
                rowCount += 1
                totalAnxiety += row.numLabel
        day = (datetime.strptime(dates[x], "%Y-%m-%d")).strftime("%a")
        days.append(day)
        avgAnxiety.append(totalAnxiety/rowCount)


    plot_df['date'] = dates
    plot_df['Anxiety'] = avgAnxiety
    plot_df['day'] = days


    # In[42]:




    # In[21]:


    fig, ax = plt.subplots()
    c1 = '#9dd6f5'
    c2 = '#4ba0d1'
    c3 = '#23495f'
    bar_width = 0.25
    for t, y in zip(plot_df["day"], plot_df["Anxiety"]):

        c=""
        if(y <= 1):
            c = c1      
        elif (1 < y <= 2):
            c = c2       
        elif ( y > 2): 
            c = c3          
        plt.bar([t,t], [0,y],bar_width, color=c)

    colors = [[c1,c1],[c2,c2],[c3,c3]]          
    categories = ['Low','Meduim','High']

    #create dict
    legend_dict=dict(zip(categories,colors))
    #create patches
    patchList = []
    for key in legend_dict:
            data_key = mpatches.Patch(facecolor=legend_dict[key][0], 
                                      edgecolor=legend_dict[key][1], label=key)
            patchList.append(data_key)

    ax.legend(handles=patchList,ncol=len(categories), fontsize=12)   


    plt.tick_params(axis='x')
    # Start the graph at 0

    ax.set_ylim(0, 3)

    #fig.set_size_inches(15.5, 10)
    plt.tight_layout()


    #plt.xlabel('Date')

    ax.yaxis.set_label_coords(-0.02, 0.48)


    fig.savefig('AL.png', dpi = None)
    imagePath = "AL.png"
    link = storage.child(userID+"/dailyReport/AL.png").put('AL.png')
    os.remove("AL.png")


    # In[22]:


    linkDF = pd.DataFrame()
    linkDF = linkDF.append(pd.DataFrame(link,index=[0]),ignore_index=True)
    token1 = linkDF['downloadTokens'].values
    token = token1[0]
    AlLink = storage.child(userID+"/dailyReport/AL.png").get_url(token)


    # # Improvement

    # In[77]:


    # get yesterday improvement 
    #today = datetime.datetime.now()
    #yesterday = today - datetime.timedelta(days=1)
    #timestamp = yesterday.strftime("%Y-%m-%d")
    al = float(plot_df[plot_df.date == timestamp].Anxiety)
    today_al = float("{:.2f}".format(al))
    yesterday = plot_df.iloc[5]['Anxiety']
    today = plot_df.iloc[6]['Anxiety']
    imp = ((yesterday - today)/3)*100


    # get before yesterday to calculate the improvement
    try:
        doc_ref = db.collection(u'DailyReport').document('daily'+userID)
        doc = doc_ref.get().to_dict()
        prev = float(doc['anxiety_level'])
        # calculate the improvement
        #1. get today Al
        improvement= ((prev - today_al)/3)*100 
        doc_rec = db.collection(u'DailyReport').document('daily'+userID)
        doc_rec.set({

            u'AL_graph': AlLink,
            u'improvement': float("{:.2f}".format(improvement)),#float("{:.2f}".format(improvement))   
            u'anxiety_level' : float(today_al),
            u'date': str(today)
        })
        
        yesterday = plot_df.iloc[6]['Anxiety']
        today = plot_df.iloc[5]['Anxiety']

    except:
        doc_rec = db.collection(u'DailyReport').document('daily'+userID)
        doc_rec.set({

            u'AL_graph': AlLink,
            u'improvement': -1 ,
            u'anxiety_level' : float(today_al),
            u'date': str(today)
        })


    # In[ ]:
    json_df = plot_df[['Anxiety','date']].to_json()


   
    return json_df



