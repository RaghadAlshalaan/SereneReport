    #!/usr/bin/env python
    # coding: utf-8

    # In[1]:
def reportP(pid,dates , is_true):

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


    # In[2]:


    if not firebase_admin._apps:
            cred = credentials.Certificate("serene-firebase-adminsdk.json")
            app = firebase_admin.initialize_app(cred ,  {
            'storageBucket': 'serene-2dfd6.appspot.com',
            }, name='[DEFAULT]')
    else:
            app = firebase_admin.get_app()
    db = firestore.client()


    # In[3]:


    userID = pid
    GoogleCalendar = is_true


    # In[4]:


    today = datetime.datetime.now()
    timestamp = today.strftime("%Y-%m-%d %H:%M:%S")
    bucket = storage.bucket(app=app)


    # ## Get data from storage and get list of dates 

    # In[5]:
    df= pd.DataFrame()
    notAvailableDates = []
    # loop through the storage and get the data
    sleep =[]
    for x in range(0 ,len(dates)):
        #Sleep
        blob = bucket.blob(userID+"/fitbitData/"+dates[x]+"/"+dates[x]+"-sleep.json")
         #download the file 
        u = blob.generate_signed_url(datetime.timedelta(seconds=300), method='GET')
        try:
            with urllib.request.urlopen(u) as url:
                data = json.loads(url.read().decode())
                sleepMinutes = data['summary']["totalMinutesAsleep"]
        except:
            notAvailableDates.append(dates[x])
            pass

        #Activity (Steps)
        blob = bucket.blob(userID+"/fitbitData/"+dates[x]+"/"+dates[x]+"-activity.json")
        #download the file 
        u = blob.generate_signed_url(datetime.timedelta(seconds=300), method='GET')
        try:
            with urllib.request.urlopen(u) as url:
                data = json.loads(url.read().decode())
                steps = data['summary']["steps"]
        except:
            notAvailableDates.append(dates[x])
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
            notAvailableDates.append(dates[x])
            pass

         #append dataframe
        df = df.append(heart_rate, ignore_index = True)


    
    

    notAvailableDates
    notSyncedDates = pd.DataFrame()
    notSyncedDates ['date'] = notAvailableDates 


    # In[16]:


    notSyncedDates = notSyncedDates.drop_duplicates()


    # ### Get user location

    # In[17]:


    # get location from database
    loc_df = pd.DataFrame()
    locID = []
    locations = db.collection(u'PatientLocations').where(u'patientID', u'==', userID ).stream()

    for location in locations:
        loc = location.to_dict()
        locID.append(location.id)
        loc_df = loc_df.append(pd.DataFrame(loc,index=[0]),ignore_index=True)

    loc_df['id'] = locID


    # In[18]:


    loc_df.drop(['anxietyLevel', 'lat','lng', 'patientID'  ], axis=1, inplace = True)


    # In[19]:


    loc_df.time.apply(str)
    loc_df['time'] = pd.to_datetime(loc_df['time'])
    loc_df['date'] = pd.to_datetime(loc_df['time'], format='%Y:%M:%D').dt.date
    loc_df['hour'] = loc_df['time'].apply(lambda time: time.strftime('%H'))
    loc_df.drop(['time'], axis=1, inplace = True)
    loc_df.hour = loc_df.hour.astype(int) 
    loc_df.date = loc_df.date.astype(str)
    df.date = df.date.astype(str)


    # In[20]:


  

    dfinal = pd.merge(left=df, 
                      right = loc_df,
                      how = 'left',
                      left_on=['hour','date'],
                      right_on=['hour','date']).ffill()


    # ### Test data into model

    # In[23]:


    #test model 
    train_df = dfinal.rename(columns={'value': 'Heartrate'})


    # In[24]:


    Labeled_df = pd.DataFrame()
    Labeled_df = trainData(train_df)


    # In[25]:


    Labeled_df.drop(['lon'],axis=1, inplace = True)


    # In[26]:


    # Replace missing values because it doesn't exist
    Labeled_df['name'].fillna("Not given", inplace=True)
    Labeled_df['id'].fillna("Not given", inplace=True)
    #Labeled_df['anxiety_assigned'].fillna('Not given', inplace = True)


    # In[27]:




    # In[28]:


    # Update firebase with the user anxiety level 
    for row in Labeled_df.itertuples():
        if row.id != 'Not given':
            if row.Label == 'Low' or row.Label == 'LowA':
                anxietyLevel = '1'
            elif row.Label == 'Meduim':
                anxietyLevel = '2'
            else:
                anxietyLevel = '3' 

            if ((row.anxiety_assigned == False) or (row.anxiety_assigned == 'Not given')): 
                doc_ref = db.collection(u'PatientLocations').document(row.id)
                doc_ref.update({
                                u'anxietyLevel':anxietyLevel,
                                u'anxiety_assigned': True
                         })


    # ### Show the places with highest anxiety level

    # In[29]:


    # Show the highest level 
    df_high = pd.DataFrame()
    df_high = Labeled_df[Labeled_df.Label == 'High']


    # # Recommendation

    # In[30]:


    docDf = pd.DataFrame()
    doc_ref = db.collection(u'Patient').document(userID)
    doc = doc_ref.get().to_dict()
    docDf = docDf.append(pd.DataFrame(doc,index=[0]),ignore_index=True)


    # In[31]:


    age1 = docDf['age'].values
    name1 = docDf['name'].values
    emp1 = docDf['employmentStatus'].values
    mar1 = docDf['maritalStatus'].values
    income1 = docDf['monthlyIncome'].values
    chronicD1 = docDf['chronicDiseases'].values
    smoke1 = docDf['smokeCigarettes'].values
    gad1 = docDf['GAD-7ScaleScore'].values
    gender1 = docDf['gender'].values

    age = age1[0] 
    name = name1[0]
    emp = emp1[0]
    mar = mar1[0]
    income = income1[0]
    chronicD = chronicD1[0]
    smoke = smoke1[0]
    gad = gad1[0]
    gender = gender1[0]


    compareAge = int(age)


    # In[32]:


    sleepMin = Labeled_df['sleepMin'].mean()
    totalSteps = Labeled_df['TotalSteps'].astype(float).mean()

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


    sleepMin = float("{:.1f}".format(sleepMin))
    totalSteps = int(totalSteps)


    # ## Storage intilization

    # In[33]:


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

    # In[34]:


    sns.set( rc={'axes.facecolor': '#fcfeff'})


    # In[35]:


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


    # In[36]:


    # Get anxiety level by day and store it in a new data frame
    plot_df = pd.DataFrame()
    avgAnxiety = []
    totalAnxiety = 0
    rowCount = 1
    for x in range(0 ,len(dates)):
        totalAnxiety = 0
        rowCount = 1
        for row in Labeled_df.itertuples():
            if (row.date == dates[x]):
                rowCount += 1
                totalAnxiety += row.numLabel
        avgAnxiety.append(totalAnxiety/rowCount)


    plot_df['date'] = dates
    plot_df['Anxiety'] = avgAnxiety


    # In[37]:


    #divide dataframe into 15 rows (2 weeks) max is 3 months

    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    df3 = pd.DataFrame()
    df4 = pd.DataFrame()
    df5 = pd.DataFrame()
    df6 = pd.DataFrame()
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




    # In[38]:


    # Plot AL
    if(len(plot_df)<=15):
        fig, ax = plt.subplots()
        c1 = '#9dd6f5'
        c2 = '#4ba0d1'
        c3 = '#23495f'
        bar_width = 0.25
        for t, y in zip(plot_df["date"], plot_df["Anxiety"]):

            c=""
            if(y <= 1):
                c = c1      
            elif (1 < y <= 2):
                c = c2       
            elif ( y > 2): 
                c = c3          
            plt.bar([t,t], [0,y], bar_width,color=c )

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


        plt.tick_params(axis='x', rotation=70)

        # Start the graph at 0

        ax.set_ylim(0, 3)

        #fig.set_size_inches(15.5, 10)
        plt.tight_layout()


        plt.xlabel('Date')

        ax.yaxis.set_label_coords(-0.02, 0.48)



        fig.savefig('AL0.png', dpi = None)
        imagePath = 'AL0.png'
        link = storage.child(userID+"/lastGeneratedPatientReport/AL0.png").put('AL0.png')
        os.remove('AL0.png')


    else:   
        links = []
        for x in range(0,len(dfarray)):
            fig, ax = plt.subplots()
            c1 = '#9dd6f5'
            c2 = '#4ba0d1'
            c3 = '#23495f'
            bar_width = 0.25
            for t, y in zip(dfarray[x]["date"], dfarray[x]["Anxiety"]):

                c=""
                if(y <= 1):
                    c = c1      
                elif (1 < y <= 2):
                    c = c2       
                elif ( y > 2): 
                    c = c3          
                plt.bar([t,t], [0,y], bar_width,color=c )
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


            plt.tick_params(axis='x', rotation=70)

        # Start the graph at 0

            ax.set_ylim(0, 3)

            #fig.set_size_inches(15.5, 10)
            plt.tight_layout()


            plt.xlabel('Date')

            ax.yaxis.set_label_coords(-0.02, 0.48)


            fig.savefig('AL'+str(x)+'.png', dpi = None)
            imagePath = 'AL'+str(x)+'.png'
            link = storage.child(userID+"/lastGeneratedPatientReport/AL"+str(x)+'.png').put('AL'+str(x)+'.png')
            links.append(link)
            os.remove('AL'+str(x)+'.png')



    # ### save file to database first

    # In[39]:


    if (len(plot_df)<=15):
        linkDF = pd.DataFrame()
        linkDF = linkDF.append(pd.DataFrame(link,index=[0]),ignore_index=True)
        token1 = linkDF['downloadTokens'].values
        token = token1[0]
        AlLink = storage.child(userID+"/lastGeneratedPatientReport/AL0.png").get_url(token)
        doc_rec = db.collection(u'LastGeneratePatientReport').document('report'+userID)
        doc_rec.set({
            u'average_steps': totalSteps,
            u'patient_id':userID,
            u'average_sleep_hours': sleepMin,
            u'sleepRecomendation': sleepRecomendation,
            u'stepsRecomendation': stepsRecomendation,
            u'recommended_steps': recomendedSteps,
            u'number_of_AL_graphs': 1,
            u'AL_graph_0': AlLink,
            u'recommended_sleep_hours': '7-9'

        })
    else:
        firebase_links = []
        for x in range(0,len(links)):
            linkDF = pd.DataFrame()
            linkDF = linkDF.append(pd.DataFrame(link,index=[0]),ignore_index=True)
            token1 = linkDF['downloadTokens'].values
            token = token1[0]
            AlLink = storage.child(userID+"/lastGeneratedPatientReport/AL"+str(x)+'.png').get_url(token)
            firebase_links.append(AlLink)

        doc_rec = db.collection(u'LastGeneratePatientReport').document('report'+userID)
        doc_rec.set({
            u'average_steps': totalSteps,
            u'patient_id':userID,
            u'average_sleep_hours': sleepMin,
            u'sleepRecomendation': sleepRecomendation,
            u'stepsRecomendation': stepsRecomendation,
            u'recommended_steps': recomendedSteps,
            u'number_of_AL_graphs': len(links),
            u'recommended_sleep_hours': '7-9'

        })
        for x in range(0,len(links)):
            doc_rec = db.collection(u'LastGeneratePatientReport').document('report'+userID)
            doc_rec.update({
            u'AL_graph_'+ str(x): firebase_links[x]
            })


    # ## To generate graphs for PDF report

    # In[40]:


    fig, ax = plt.subplots()
    c1 = '#9dd6f5'
    c2 = '#4ba0d1'
    c3 = '#23495f'
    bar_width = 0.25
    for t, y in zip(plot_df["date"], plot_df["Anxiety"]):

        c=""
        if(y <= 1):
            c = c1      
        elif (1 < y <= 2):
            c = c2       
        elif ( y > 2): 
            c = c3          
        plt.bar([t,t], [0,y], bar_width,color=c )

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


    plt.tick_params(axis='x', rotation=70)

    # Start the graph at 0

    ax.set_ylim(0, 3)

    fig.set_size_inches(15.5, 10)
    plt.tight_layout()


    plt.xlabel('Date')

    ax.yaxis.set_label_coords(-0.02, 0.48)

    fig.savefig('ALpdf.png', dpi = None)





    # # Location Analysis

    # In[41]:


    # get location from database
    new_loc = pd.DataFrame()
    locID = []
    locations = db.collection(u'PatientLocations').where(u'patientID', u'==', userID ).stream()

    for location in locations:
        loc = location.to_dict()
        locID.append(location.id)
        new_loc = new_loc.append(pd.DataFrame(loc,index=[0]),ignore_index=True)

    new_loc['id'] = locID



    # In[42]:


    new_loc.time.apply(str)
    new_loc['time'] = pd.to_datetime(new_loc['time'])
    new_loc['date'] = pd.to_datetime(new_loc['time'], format='%Y:%M:%D').dt.date
    new_loc.drop(['time'], axis=1, inplace = True)
    new_loc.date = new_loc.date.astype(str)


    # In[43]:


    new_loc = new_loc[(new_loc.date >= dates[0]) & (new_loc.date <= dates[len(dates)-1])]


    # In[44]:


    names = []
    Name =""
    for row in new_loc.itertuples():
        Name  = row.nearestLoc         
        names.append(Name)


    # In[45]:


    new_name =pd.DataFrame()
    new_name ['name']= names


    # In[46]:


    new_name = new_name.drop_duplicates()
    new_name.dropna()


    # In[47]:


    fnames = []
    fName =""
    for row in new_name.itertuples():
        fName  = row.name
        fnames.append(fName)


    # In[48]:


    analysis_EN = pd.DataFrame()
    analysis_AR = pd.DataFrame()
    count = 0
    i = 0
    #label = ""
    locationName = ""
    near = ''
    nearLocs = []
    counts = []
    labels = []
    locationNames = []
    for x in range(0,len(fnames)):
        count = 0
        locName = fnames[i]
        for row in new_loc.itertuples():
            if(locName == row.nearestLoc):
                if(row.anxietyLevel=='3'):
                    count+=1
                    label = row.anxietyLevel
                    locationName = row.name
                    near = row.nearestLoc    


        i+=1           
        counts.append(count)
        #labels.append(label)
        locationNames.append(locationName)
        nearLocs.append(near)

    analysis_EN ['Location'] = locationNames
    analysis_EN ['Number of occurrences'] = counts
    #analysis_EN ['Anxiety Level'] = labels
    analysis_EN ['Nearest Location'] = nearLocs

    analysis_AR ['الموقع'] = locationNames
    analysis_AR ['عدد مرات الزيارة'] = counts
    #analysis_AR ['مستوى القلق'] = labels
    analysis_AR ['أقرب موقع'] = nearLocs


    # In[49]:


    newEn = analysis_EN.drop(analysis_EN[analysis_EN['Number of occurrences'] == 0].index, inplace= True)
    newAr = analysis_AR.drop(analysis_AR[analysis_AR['عدد مرات الزيارة'] == 0].index, inplace= True)


    # In[50]:


    #analysis_EN ['Anxiety Level'] = 'High'
    #analysis_AR  ['مستوى القلق'] = 'مرتفع'


    # In[51]:


    import six

    import arabic_reshaper
    from googletrans import Translator
    from bidi.algorithm import get_display

    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont


    # In[52]:


    def render_mpl_table(data, col_width=5.0, row_height=0.625, font_size=14, tran = '',
                         header_color='#23495f', row_colors=['#e1eff7', 'w'], edge_color='#23495f',
                         bbox=[0, 0, 1, 1], header_columns=0,
                        ax=None, **kwargs):


        if ax is None:
            size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
            fig, ax = plt.subplots(figsize=size)
            ax.axis('off')

        mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, cellLoc='center'  ,**kwargs)

        mpl_table.auto_set_font_size(False)
        mpl_table.set_fontsize(font_size)


        for k, cell in  six.iteritems(mpl_table._cells):
            cell.set_edgecolor(edge_color)
            if k[0] == 0 or k[1] < header_columns:
                cell.set_text_props(weight='bold', color='w')
                cell.set_facecolor(header_color)
            else:
                cell.set_facecolor(row_colors[k[0]%len(row_colors) ])
                cell.alignment = 'center'



        fig.savefig(tran+'.png', dpi = 100)
        return ax


    # In[53]:


    if(len(analysis_EN) > 0):
        for ind,row in analysis_EN.iterrows():
            analysis_EN.loc[ind,'Nearest Location']=get_display(arabic_reshaper.reshape(analysis_EN.loc[ind,'Nearest Location']))
    if(len(analysis_AR) > 0):
        for ind,row in analysis_AR.iterrows():
            analysis_AR.loc[ind,'أقرب موقع']=get_display(arabic_reshaper.reshape(analysis_AR.loc[ind,'أقرب موقع']))
           # analysis_AR.loc[ind,'مستوى القلق']=get_display(arabic_reshaper.reshape(analysis_AR.loc[ind,'مستوى القلق']))
        analysis_AR = analysis_AR.rename(columns={'الموقع': get_display(arabic_reshaper.reshape('الموقع')) })
        analysis_AR = analysis_AR.rename(columns={'التكرار': get_display(arabic_reshaper.reshape('التكرار')) })
        #analysis_AR = analysis_AR.rename(columns={'مستوى القلق': get_display(arabic_reshaper.reshape('مستوى القلق')) })
        analysis_AR = analysis_AR.rename(columns={'أقرب موقع': get_display(arabic_reshaper.reshape('أقرب موقع')) }) 



    # In[54]:


    if(len(analysis_EN) > 0):
        render_mpl_table(analysis_EN, header_columns=0, col_width=6, tran ='Location-EN')
    if(len(analysis_AR) > 0):
        render_mpl_table(analysis_AR, header_columns=0, col_width=6, tran ='Location-AR')


    # # Events with highest level of anxiety

    # In[59]:


    from dateutil.parser import parse


    # In[60]:


    if GoogleCalendar == 'true':
        events_df = pd.DataFrame()
        finalEvents_EN = pd.DataFrame()
        finalEvents_AR = pd.DataFrame()
        eventID = []
        events = db.collection(u'PatientEvents').where(u'patientID', u'==', userID ).stream()

        for event in events:
            ev = event.to_dict()
            eventID.append(event.id)
            events_df = events_df.append(pd.DataFrame(ev,index=[0]),ignore_index=True)     
        events_df['idEvent'] = eventID


    # In[61]:


        if len(events_df) > 0:
            newDates = []
            for row in events_df.itertuples():
                newDate = parse(row.date)
                newDates.append(newDate)
            events_df['newDate'] = newDates

            events_df = events_df.rename(columns={'name': 'eventName'})
            events_df.newDate.apply(str)
            events_df['date'] = pd.to_datetime(events_df['newDate'], format='%Y:%M:%D').dt.date
            events_df['hour'] = events_df['newDate'].apply(lambda time: time.strftime('%H'))
            events_df.hour = events_df.hour.astype(int) 
            events_df.date = events_df.date.astype(str)

            Test = Labeled_df

            merge_df = pd.merge(left=Test,right = events_df, how = 'left',left_on=['hour','date'],right_on=['hour','date']).ffill()

            merge_df['eventName'].fillna("Not given", inplace=True)
            merge_df= merge_df[merge_df.eventName != 'Not given']
        
            #finalEvents_EN = pd.DataFrame()
            #finalEvents_AR = pd.DataFrame()
            ev_name=''
            evNames = []
            #evLabels = []
            evDate = []
            for row in merge_df.itertuples():
                if row.eventName != ev_name:
                    if row.Label == 'High':
                        ev_name = row.eventName
                        #ev_label = row.Label
                        ev_date = row.date
                if(ev_name != ''):
                    evNames.append(ev_name)
                    #evLabels.append(ev_label)
                    evDate.append(ev_date)

            finalEvents_EN ['Event Name'] = evNames
            #finalEvents_EN['Anxiety Level'] = evLabels
            finalEvents_EN['Date'] = evDate

            finalEvents_AR ['اسم الحدث'] = evNames
            #finalEvents_AR['مستوى القلق'] = evLabels
            finalEvents_AR['تاريخ الحدث'] = evDate

            finalEvents_EN = finalEvents_EN.drop_duplicates()
            finalEvents_AR = finalEvents_AR.drop_duplicates()

            Final_duplicated = pd.DataFrame()
            eventId = ''
            eventIds = []
            event_Labels = []
            event_names = []
            hours = []
            for row in merge_df.itertuples():
                if eventId != row.idEvent:
                    eventId = row.idEvent
                if eventId != '':
                    eventIds.append(eventId)
                    event_Labels.append(row.Label)
                    event_names.append(row.eventName)
                    hours.append(row.hour)
            Final_duplicated ['id'] = eventIds
            Final_duplicated ['Label'] = event_Labels
            Final_duplicated ['name'] = event_names
            Final_duplicated ['hour'] = hours
            Final_duplicated = Final_duplicated.drop_duplicates(subset ='id')


            # Update firebase with the user anxiety level 
            for row in Final_duplicated.itertuples():
                if row.Label == 'High':
                    doc_ref = db.collection(u'PatientEvents').document(row.id)
                    doc_ref.update({
                                                u'anxietyLevel':'3'
                                        })
                else:
                    doc_ref = db.collection(u'PatientEvents').document(row.id)
                    doc_ref.delete()
            checkEvents_df = pd.DataFrame()
            cEventsID = []
            checkEvents = db.collection(u'PatientEvents').where(u'patientID', u'==', userID ).stream()

            for event in checkEvents:
                cEv = event.to_dict()
                cEventsID.append(event.id)
                checkEvents_df = checkEvents_df.append(pd.DataFrame(cEv,index=[0]),ignore_index=True)     
            checkEvents_df['id'] = cEventsID
            checkEvents_df.fillna("Not given", inplace=True)



            if len(checkEvents_df) > 0:
                checkEvents_df = checkEvents_df[checkEvents_df.anxietyLevel == 'Not given']
                for row in checkEvents_df.itertuples():
                    doc_ref = db.collection(u'PatientEvents').document(row.id)
                    doc_ref.delete()



        # In[62]:


            if(len(finalEvents_EN) > 0):
                finalEvents_EN = finalEvents_EN.drop_duplicates()
                finalEvents_AR = finalEvents_AR.drop_duplicates()



        # In[63]:


            if(len(finalEvents_EN) > 0):
                for ind,row in finalEvents_EN.iterrows():
                    try:
                        finalEvents_EN.loc[ind,'Event Name']=get_display(arabic_reshaper.reshape(
                           finalEvents_EN.loc[ind,'Event Name']))
                    except:
                        pass

            if(len(finalEvents_AR) > 0):
                #finalEvents_AR['مستوى القلق'] = 'مرتفع'
                for ind,row in finalEvents_AR.iterrows():
                    try:
                        finalEvents_AR.loc[ind,'اسم الحدث']=get_display(arabic_reshaper.reshape(finalEvents_AR.loc[ind,'اسم الحدث']))
                        #finalEvents_AR.loc[ind,'مستوى القلق']=get_display(arabic_reshaper.reshape(finalEvents_AR.loc[ind,'مستوى القلق']))
                    except:
                        pass
                finalEvents_AR = finalEvents_AR.rename(columns={'اسم الحدث': get_display(arabic_reshaper.reshape('اسم الحدث')) })
                #finalEvents_AR = finalEvents_AR.rename(columns={'مستوى القلق': get_display(arabic_reshaper.reshape('مستوى القلق')) })
                finalEvents_AR = finalEvents_AR.rename(columns={'تاريخ الحدث': get_display(arabic_reshaper.reshape('تاريخ الحدث')) })




            # In[64]:


            if(len(finalEvents_EN) > 0):
                render_mpl_table(finalEvents_EN, header_columns=0, col_width=4, tran ='Events-EN')
            if(len(finalEvents_AR) > 0):
                render_mpl_table(finalEvents_AR, header_columns=0, col_width=4, tran ='Events-AR')


    # In[ ]:






    # # Genertate patient report and save it in storage

    # In[65]:


    from reportlab.lib import colors


    # In[66]:


    pdfmetrics.registerFont(TTFont('Arabic', 'traditional-arabic/tradbdo.TTF'))


    # In[67]:


    pdf = canvas.Canvas('Patient-EN.pdf')
    pdf.setTitle('Patient report')



    pdf.drawImage("serene .png", 150, 730, width=300,height=130, mask= 'auto')

    pdf.setFillColor(colors.HexColor('#e1eff7'))
    pdf.roundRect(57,620, 485,50,4,fill=1, stroke= 0)




    pdf.setFont("Helvetica-Bold", 16)
    pdf.setFillColor(colors.HexColor('#23495f'))

    pdf.drawString(115,638, "Report Duration From: " + dates[0] +" To: "+ dates[len(dates)-1])

    pdf.setFillColor(colors.HexColor('#e1eff7'))
    pdf.roundRect(57,400, 485,200,4,fill=1, stroke= 0)

    pdf.setFont("Helvetica-Bold", 20)
    pdf.setFillColor(colors.HexColor('#23495f'))

    pdf.drawString(100,570, "Patient Information")

    pdf.setFont("Helvetica-Bold", 15)
    pdf.drawString(150,550, "Name: " )
    pdf.drawString(150,530, "Age: " )
    pdf.drawString(150,510, "Employment Status: " )
    pdf.drawString(150,490, "Martial Status: " )
    pdf.drawString(150,470, "Monthly Income: " )
    pdf.drawString(150,450, "Chronic Diseases: " )
    pdf.drawString(150,430, "Cigarette Smoker: " ) 
    pdf.drawString(150,410, "Gender: " )

    pdf.setFont("Helvetica", 15)
    pdf.setFillColor(black)
    pdf.drawString(210,550,  name)
    pdf.drawString(210,530,  age)
    pdf.drawString(310,510,  emp)
    pdf.drawString(260,490,  mar)
    pdf.drawString(290,470,  income)
    pdf.drawString(290,450, chronicD)
    pdf.drawString(290,430,  smoke)
    pdf.drawString(220,410,  gender) 


    pdf.setFillColor(colors.HexColor('#bfbfbf'))
    pdf.roundRect(370,560, 125,30,4,fill=1, stroke= 0)

    pdf.setFillColorRGB(1,1,1)
    pdf.drawString(375,570, "GAD-7 Score = ")

    pdf.setFont("Helvetica-Bold", 15)
    pdf.drawString(480,570, gad)

    pdf.setFillColor(colors.HexColor('#e1eff7'))
    pdf.roundRect(57,160, 485,200,4,fill=1, stroke= 0)


    pdf.setFont("Helvetica-Bold", 16)
    pdf.setFillColor(colors.HexColor('#23495f'))

    pdf.drawString(80,330, "Recommendations: ")
    pdf.drawString(100,300, "Sleep Recomendation: ")
    pdf.drawString(100,220, "Steps Recomendation: ")

    pdf.setFont("Helvetica", 11)
    pdf.setFillColor(black)

    if(sleepRecomendation == True):
        t = pdf.beginText(100,280)
        text = ["It is recommended to sleep at least 7-9 hours per day to get lower chance of anxiety." ,
               "Getting a good night’s sleep can improve your mental well-being and help you to better",
                "manage your anxiety."]
        for line in text:
            t.textLine(line)
        pdf.drawText(t)
    else:
        t = pdf.beginText(100,280)
        text = ["Good News! It seems you're getting enough sleep!",
                "Getting a good night’s sleep can improve your mental well-being and help you to better",
                "manage your anxiety."]
        for line in text:
            t.textLine(line)
        pdf.drawText(t)

    if(stepsRecomendation == True):
        t = pdf.beginText(100,200)
        text = ["It is recommended to walk at least "+recomendedSteps+" steps per day to get lower chance of anxiety." ,
                "Keep in mind that active people have lower rates of anxiety than sedentary people."]
        for line in text:
            t.textLine(line)
        pdf.drawText(t)

    else:
        t = pdf.beginText(100,200)
        text = ["Great Work! You're considered an active person based on your average steps per day.",
                "Keep in mind that active people have lower rates of anxiety than sedentary people."]
        for line in text:
            t.textLine(line)
        pdf.drawText(t)


    pdf.showPage()





    pdf.drawImage("serene .png", 150, 730, width=300,height=130, mask= 'auto')


    pdf.setFont("Helvetica-Bold", 20)
    pdf.setFillColor(colors.HexColor('#808080'))

    pdf.drawString(100,650, "Anxiety Level")

    pdf.drawImage("ALpdf.png", 57, 400, width=485,height=200)

    pdf.drawString(100,320, "Locations With Highest Level of Anxiety")
    if(len(analysis_EN) > 0):
        pdf.drawImage("Location-EN.png", 30, 200,width=570,height=100)
    else:
        pdf.setFont("Helvetica", 15)
        pdf.setFillColor(colors.HexColor('#23495f'))

        t = pdf.beginText(130,250)
        text = [
        name +" condition was stable through this period,", 
        "no locations with high anxiety level were detected." ]
        for line in text:
            t.textLine(line)

        pdf.drawText(t)


    pdf.showPage()

    pdf.drawImage("serene .png", 150, 730, width=300,height=130, mask= 'auto')

    pdf.setFont("Helvetica-Bold", 20)
    pdf.setFillColor(colors.HexColor('#808080'))
    pdf.drawString(100,650, "Events With Highest Level of Anxiety")

    if(GoogleCalendar == 'true' and len(finalEvents_EN)>0):
        pdf.drawImage("Events-EN.png", 30, 500, width=550,height=100)
    elif (GoogleCalendar == 'true' and len(finalEvents_EN)<0  ):
        pdf.setFont("Helvetica", 15)
        pdf.setFillColor(colors.HexColor('#23495f'))

        t = pdf.beginText(130,550)
        text = [
        "No events found associated with high anxiety level"]
        for line in text:
            t.textLine(line)

        pdf.drawText(t)
    else:
        pdf.setFont("Helvetica", 15)
        pdf.setFillColor(colors.HexColor('#23495f'))

        t = pdf.beginText(130,550)
        text = [
        name +" have not synced Google Calendar events,", 
        "with thier account." ]
        for line in text:
            t.textLine(line)

        pdf.drawText(t)

    if(len(notSyncedDates) != 0):  
        pdf.setFont("Helvetica", 12)
        pdf.setFillColor(colors.HexColor('#d40027'))
        pdf.drawString(75,95, "Note: Below dates are missing, because they were not synced correctly:")
        i = 70
        for row in notSyncedDates.itertuples():
            pdf.drawString(85,i, '- '+ row.date)
            i = i-20

    pdf.save()

    # In[68]:


    def translate(text):
        reshaped_text = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped_text)
        return bidi_text


    # In[69]:


    def from_en_to_ar(text):
        translator = Translator()
        result = translator.translate(text, dest='ar')
        return result.text


    # In[70]:


    pdf = canvas.Canvas('Patient-AR.pdf')
    pdf.setTitle('تقرير المريض')




    pdf.drawImage("serene .png", 150, 730, width=300,height=130, mask= 'auto')

    pdf.setFillColor(colors.HexColor('#e1eff7'))
    pdf.roundRect(57,620, 485,50,4,fill=1, stroke= 0)

    pdf.setFont("Arabic", 22)
    pdf.setFillColor(colors.HexColor('#23495f'))

    reportDuration = translate (u'مدة التقرير من')
    to = translate (u'إلى')
    pdf.drawString(125,638, dates[len(dates)-1] +' '+ to +' ' + dates[0] +' '+reportDuration)



    pdf.setFillColor(colors.HexColor('#e1eff7'))
    pdf.roundRect(57,400, 485,200,4,fill=1, stroke= 0)

    pdf.setFont("Arabic", 22)
    pdf.setFillColor(colors.HexColor('#23495f'))

    patient = translate (u'معلومات المريض')
    pdf.drawString(400,570, patient)


    na =  translate (u'الاسم:')
    ag =  translate (u'العمر:')
    Es =  translate (u'الحالة الوظيفية:')
    Ms =  translate (u'الحالة الإجتماعية:')
    Mi =  translate (u'الدخل الشهري:')
    Cd =  translate (u'الأمراض المزمنة:')
    Cs =  translate (u'مدخن:')
    ge = translate (u'الجنس:')

    pdf.setFont("Arabic", 18)
    pdf.drawString(445,550, na )
    pdf.drawString(446,530, ag)
    pdf.drawString(400,510, Es )
    pdf.drawString(390,490, Ms)
    pdf.drawString(400,470, Mi )
    pdf.drawString(395,450, Cd)
    pdf.drawString(445,430, Cs ) 
    pdf.drawString(446,410, ge ) 


    pdf.setFont("Arabic", 15)
    pdf.setFillColor(black)
    pdf.drawString(400,550,  name)
    pdf.drawString(420,530,  age)
    pdf.drawString(350,510,  translate(from_en_to_ar(emp)))
    pdf.drawString(355,490,  translate(from_en_to_ar(mar)))
    pdf.drawString(360,470,  income)
    pdf.drawString(360,450,  translate(from_en_to_ar(chronicD)))
    pdf.drawString(420,430,  translate(from_en_to_ar(smoke))) 
    pdf.drawString(420,410,  translate(from_en_to_ar(gender))) 


    pdf.setFillColor(colors.HexColor('#bfbfbf'))
    pdf.roundRect(100,560, 120,30,4,fill=1, stroke= 0)

    ga = translate (u'مقياس GAD-7 =')
    pdf.setFillColorRGB(1,1,1)
    pdf.drawString(120,570, ga)

    pdf.setFont("Helvetica-Bold", 15)
    pdf.drawString(105,570, gad)

    pdf.setFillColor(colors.HexColor('#e1eff7'))
    pdf.roundRect(57,160, 485,200,4,fill=1, stroke= 0)


    pdf.setFont("Arabic", 22)
    pdf.setFillColor(colors.HexColor('#23495f'))


    rec = translate (u'  التوصيات:')
    stepRec = translate(u'توصيات النشاط:')
    sleepRec = translate (u'توصيات النوم:')

    pdf.drawString(420,330, rec)
    pdf.drawString(380,300, sleepRec)
    pdf.drawString(360,220, stepRec)

    pdf.setFont("Arabic", 15)
    pdf.setFillColor(black)

    if(sleepRecomendation == True):

        text = translate (u'يوصى بالنوم لمدة تتراوح بين ٧-٩ ساعات في اليوم للحصول على أقل معدل قلق.')
        text2 = translate (u'يمكن أن يحسن النوم الجيد ليلاً من صحتك العقلية ويساعدك على إدارة قلقك بشكل أفضل.')
        pdf.drawString(120,280, text)
        pdf.drawString(80,260, text2)

    else:

        text = translate (u'أخبار جيدة! يبدو أنك تحصل على قسط كافٍ من اليوم.')
        text2 = translate (u'يمكن أن يحسن النوم الجيد ليلاً من صحتك العقلية ويساعدك على إدارة قلقك بشكل أفضل.')                   
        pdf.drawString(230,280, text)
        pdf.drawString(80,260, text2)


    if(stepsRecomendation == True):

        btwnText = translate (u'خطوة في اليوم للحصول على أقل معدل قلق. ')
        text = translate (u'يوصى بالمشي على الأقل')
        text2 = translate (u'تذكر أن الأشخاص الأكثر نشاطاً لديهم معدلات قلق أقل من أولئك الغير نشطين.')
        pdf.drawString(130,200, btwnText + recomendedSteps + text)
        pdf.drawString(120,180, text2)


    else:
        text = translate (u'عمل رائع! بناءاً على معدل خطواتك في اليوم الواحد، أنت تعتبر من الأشخاص النشطين.')
        text2 = translate (u'تذكر أن الأشخاص الأكثر نشاطاً لديهم معدلات قلق أقل من أولئك الغير نشطين.')
        pdf.drawString(100,200, text)
        pdf.drawString(130,180, text2)


    pdf.showPage()





    pdf.drawImage("serene .png", 150, 730, width=300,height=130, mask= 'auto')


    pdf.setFont("Arabic", 28)
    pdf.setFillColor(colors.HexColor('#808080'))

    AL = translate(u'مستوى القلق')

    pdf.drawString(400,650, AL)

    loc = translate (u'تحليل المواقع ذات مستوى قلق مرتفع ')

    pdf.drawImage("ALpdf.png", 57, 400, width=485,height=200)
    pdf.drawString(210,320, loc)
    if(len(analysis_AR) > 0):
        pdf.drawImage("Location-AR.png", 10, 200, width=570,height=100)
    else:
        pdf.setFont("Arabic", 18)
        pdf.setFillColor(colors.HexColor('#23495f'))

        text = translate (u'كانت الحالة مستقرة خلال هذه الفترة, لم يتم الكشف عن أي مواقع ذات مستوى قلق مرتفع.')
        pdf.drawString(50,250, text)


    pdf.showPage()


    pdf.drawImage("serene .png", 150, 730, width=300,height=130, mask= 'auto')

    pdf.setFont("Arabic", 28)
    pdf.setFillColor(colors.HexColor('#808080'))
    EV = translate(u'تحليل الأحداث ذات مستوى قلق مرتفع ')
    pdf.drawString(200,650, EV)


    if(GoogleCalendar == 'true' and len(finalEvents_AR)>0):
        pdf.drawImage("Events-AR.png", 20, 500, width=550,height=100)
    elif (GoogleCalendar == 'true' and len(finalEvents_EN)<0):
        pdf.setFont("Arabic", 18)
        pdf.setFillColor(colors.HexColor('#23495f'))

        text = translate (u' لم يتم الكشف أي أحداث ذات مستوى قلق مرتفع.')
        pdf.drawString(250,550, text)
    else:
        pdf.setFont("Arabic", 18)
        pdf.setFillColor(colors.HexColor('#23495f'))

        text = translate (u'المستخدم لم يقم بمزامنة أحداث تقويم قوقل بحساب سيرين.')
        pdf.drawString(210,550, text)

    if(len(notSyncedDates) != 0):  
        pdf.setFont("Arabic", 15)
        pdf.setFillColor(colors.HexColor('#d40027'))
        note = translate (u'.ملاحظة: التواريخ المذكورة أدناه غير متضمنة في التحليل لعدم مزامنة البيانات')
        pdf.drawString(200,95, note)
        i = 70
        for row in notSyncedDates.itertuples():
            pdf.setFont("Helvetica", 12)
            pdf.drawString(450,i,  row.date + ' -')
            i = i-20

    pdf.save()


    # In[71]:


    #new method
    doct = storage.child(userID+"/lastGeneratedPatientReport/patientReport-EN").put('Patient-EN.pdf')
    doct = storage.child(userID+"/lastGeneratedPatientReport/patientReport-AR").put('Patient-AR.pdf')


    # In[72]:


    os.remove('Patient-EN.pdf')
    os.remove('Patient-AR.pdf')

    if(len(plot_df)<=90):
         os.remove('ALpdf.png')
    else:
        for x in range(0,len(dfarray)):
            os.remove('AL'+str(x)+'pdf.png')
    if(GoogleCalendar == 'true' and len(finalEvents_AR) > 0 ):
        os.remove('Events-AR.png')
        os.remove('Events-EN.png')
    if(len(analysis_AR)>0):
        os.remove('Location-AR.png')
        os.remove('Location-EN.png')


    to_json = plot_df[['date','Anxiety']].to_json()
    
    return to_json
    # In[ ]:




    # In[ ]:




