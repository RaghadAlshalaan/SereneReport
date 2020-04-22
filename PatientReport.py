def reportP(pid,dates):
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
    import os
    from matplotlib import pyplot as plt
    import matplotlib.dates as mdates
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


    # In[66]:


    if not firebase_admin._apps:
            cred = credentials.Certificate("serene-firebase-adminsdk.json")
            app = firebase_admin.initialize_app(cred ,  {
            'storageBucket': 'serene-2dfd6.appspot.com',
            }, name='[DEFAULT]')
    else:
            app = firebase_admin.get_app()
    db = firestore.client()
    from firebase_admin import storage


    userID = pid


    # In[13]:


    today = datetime.datetime.now()
    timestamp = today.strftime("%Y-%m-%d %H:%M:%S")
    bucket = storage.bucket(app=app)





    # In[15]:


    notAvailableDates = []
    
     # get Sleep 
    blob = bucket.blob(userID+"/lastGeneratedPatientReport/sleep.json")
    u = blob.generate_signed_url(datetime.timedelta(seconds=300), method='GET')
    with urllib.request.urlopen(u) as url:
        data = json.loads(url.read().decode())
        sleepMinutes = data['sleep']
    sleepFile = pd.DataFrame(sleepMinutes)
    
    # analyse some data in sleep 
    if(len(sleepFile) > 0):
        Testdate = ''
        sleepMin = 0
        sleep_df = pd.DataFrame()
        sleepDate = []
        sleepMinArray = []
        for row in sleepFile.itertuples():
            if row.dateOfSleep == Testdate:
                sleepMin = row.minutesAsleep + sleepMin
                sleepDate = sleepDate[:-1]
                sleepMinArray = sleepMinArray[:-1]
            else:
                Testdate = row.dateOfSleep
                sleepMin = row.minutesAsleep
            sleepDate.append(Testdate)
            sleepMinArray.append(sleepMin)
        sleep_df ['date'] = sleepDate
        sleep_df ['sleepMin'] = sleepMinArray
        
     # get Steps 
    blob = bucket.blob(userID+"/lastGeneratedPatientReport/steps.json") 
    u = blob.generate_signed_url(datetime.timedelta(seconds=300), method='GET')
    with urllib.request.urlopen(u) as url:
        data = json.loads(url.read().decode())
        steps = data['activities-steps']
    stepsFile = pd.DataFrame(steps)
    stepsFile = stepsFile.rename(columns={'value': 'TotalSteps'})

    hr_df = pd.DataFrame()
    for x in range (0, len(dates)):
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
            heart_rate = df_heartrate.groupby(["hour"], as_index=False).max()
            #heart_rate['sleepMin'] = sleep_df [sleep_df.date == dates[x]].sleepMin
            #heart_rate['TotalSteps'] = stepsFile [stepsFile.dateTime == dates[x]].value
            heart_rate['date'] = dates[x]
            heart_rate = heart_rate.astype({"hour": int})  
        except:
            notAvailableDates.append(dates[x])
            pass
        hr_df = hr_df.append(heart_rate, ignore_index = True)

        merged = pd.merge(left=hr_df, 
                  right = sleep_df,
                  how = 'left',
                  left_on=['date'],
                  right_on=['date']).ffill()

        df = pd.merge(left=merged, 
                  right = stepsFile,
                  how = 'left',
                  left_on=['date'],
                  right_on=['dateTime']).ffill()

    # In[17]:


    notAvailableDates
    notSyncedDates = pd.DataFrame()
    notSyncedDates ['date'] = notAvailableDates 


    # In[18]:


    notSyncedDates = notSyncedDates.drop_duplicates()


    # ### Get user location

    # In[19]:


    # get location from database
    loc_df = pd.DataFrame()
    locID = []
    locations = db.collection(u'PatientLocations').where(u'patientID', u'==', userID ).stream()

    for location in locations:
        loc = location.to_dict()
        locID.append(location.id)
        loc_df = loc_df.append(pd.DataFrame(loc,index=[0]),ignore_index=True)

    loc_df['id'] = locID


    # In[20]:


    loc_df.drop(['anxietyLevel', 'lat','lng', 'patientID'  ], axis=1, inplace = True)


    # In[21]:


    loc_df.time.apply(str)
    loc_df['time'] = pd.to_datetime(loc_df['time'])
    loc_df['date'] = pd.to_datetime(loc_df['time'], format='%Y:%M:%D').dt.date
    loc_df['hour'] = loc_df['time'].apply(lambda time: time.strftime('%H'))
    loc_df.drop(['time'], axis=1, inplace = True)
    loc_df.hour = loc_df.hour.astype(int) 
    loc_df.date = loc_df.date.astype(str)
    df.date = df.date.astype(str)


    # In[22]:


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
    Labeled_df['anxiety_assigned'].fillna('Not given', inplace = True)


    # In[27]:


    # Update firebase with the user anxiety level 
    for row in Labeled_df.itertuples():
        if row.id != 'Not given':
            if row.Label == 'Low' or row.Label == 'LowA':
                anxietyLevel = '1'
            elif row.Label == 'Meduim':
                anxietyLevel = '2'
            else:
                anxietyLevel = '3' 
            if row.anxiety_assigned == False or row.anxiety_assigned == 'Not given': 
                doc_ref = db.collection(u'PatientLocations').document(row.id)
                doc_ref.update({
                                u'anxietyLevel':anxietyLevel,
                                u'anxiety_assigned': True
                         })


    # ### Show the places with highest anxiety level

    # In[28]:


    # Show the highest level 
    df_high = pd.DataFrame()
    df_high = Labeled_df[Labeled_df.Label == 'High']


    # # Improvements

    # # Recommendation

    # In[31]:


    docDf = pd.DataFrame()
    doc_ref = db.collection(u'Patient').document(userID)
    doc = doc_ref.get().to_dict()
    docDf = docDf.append(pd.DataFrame(doc,index=[0]),ignore_index=True)


    # In[32]:


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


    # In[33]:


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


    sleepMin = float("{:.1f}".format(sleepMin))
    totalSteps = int(totalSteps)


    # ## Storage intilization

    # In[34]:


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

    # In[35]:


    sns.set( rc={'axes.facecolor': '#fcfeff'})


    # In[36]:


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


    # In[37]:


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


    # In[38]:


    #divide dataframe into 15 rows (2 weeks) ) max is 3 months

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
                       


    # In[39]:


    # Plot AL
    if(len(plot_df)<15):
        fig, ax = plt.subplots()
        c1 = '#9dd6f5'
        c2 = '#4ba0d1'
        c3 = '#23495f'
        for t, y in zip(plot_df["date"], plot_df["Anxiety"]):

            c=""
            if(y <= 1):
                c = c1      
            elif (1 < y <= 2):
                c = c2       
            elif ( y > 2): 
                c = c3          
            ax.plot([t,t], [0,y], color=c, marker="o",markevery=(1,2),linewidth=4,markeredgewidth=4)

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
            for t, y in zip(dfarray[x]["date"], dfarray[x]["Anxiety"]):

                c=""
                if(y <= 1):
                    c = c1      
                elif (1 < y <= 2):
                    c = c2       
                elif ( y > 2): 
                    c = c3          
                ax.plot([t,t], [0,y], color=c, marker="o",markevery=(1,2),linewidth=4,markeredgewidth=4)

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

    # In[40]:


    if (len(plot_df)<15):
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

    # In[41]:





    


    # Plot AL
    
    fig, ax = plt.subplots()
    c1 = '#9dd6f5'
    c2 = '#4ba0d1'
    c3 = '#23495f'
    for t, y in zip(plot_df["date"], plot_df["Anxiety"]):

        c=""
        if(y <= 1):
            c = c1      
        elif (1 < y <= 2):
            c = c2       
        elif ( y > 2): 
            c = c3          
        ax.plot([t,t], [0,y], color=c, marker="o",markevery=(1,2),linewidth=4,markeredgewidth=4)

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

    # In[43]:


    # get location from database
    new_loc = pd.DataFrame()
    locID = []
    locations = db.collection(u'PatientLocations').where(u'patientID', u'==', userID ).stream()

    for location in locations:
        loc = location.to_dict()
        locID.append(location.id)
        new_loc = new_loc.append(pd.DataFrame(loc,index=[0]),ignore_index=True)

    new_loc['id'] = locID



    # In[44]:


    new_loc.time.apply(str)
    new_loc['time'] = pd.to_datetime(new_loc['time'])
    new_loc['date'] = pd.to_datetime(new_loc['time'], format='%Y:%M:%D').dt.date
    new_loc.drop(['time'], axis=1, inplace = True)
    new_loc.date = new_loc.date.astype(str)


    # In[45]:


    new_loc = new_loc[(new_loc.date >= dates[0]) & (new_loc.date <= dates[len(dates)-1])]


    # In[47]:


    names = []
    Name =""
    for row in new_loc.itertuples():
        Name  = row.nearestLoc         
        names.append(Name)


    # In[48]:


    new_name =pd.DataFrame()
    new_name ['name']= names


    # In[49]:


    new_name = new_name.drop_duplicates()
    new_name.dropna()


    # In[50]:


    fnames = []
    fName =""
    for row in new_name.itertuples():
        fName  = row.name
        fnames.append(fName)


    # In[51]:


    analysis_EN = pd.DataFrame()
    analysis_AR = pd.DataFrame()
    count = 0
    i = 0
    label = ""
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
        labels.append(label)
        locationNames.append(locationName)
        nearLocs.append(near)
    
    analysis_EN ['Location'] = locationNames
    analysis_EN ['Frequency'] = counts
    analysis_EN ['Anxiety Level'] = labels
    analysis_EN ['Nearest Location'] = nearLocs

    analysis_AR ['الموقع'] = locationNames
    analysis_AR ['التكرار'] = counts
    analysis_AR ['مستوى القلق'] = labels
    analysis_AR ['أقرب موقع'] = nearLocs


    # In[53]:


    newEn = analysis.drop(analysis[analysis['Frequency'] == 0].index, inplace= True)
    newAr = analysis_AR.drop(analysis_AR[analysis_AR['التكرار'] == 0].index, inplace= True)

    # In[54]:


    analysis_EN ['Anxiety Level'] = 'High'
    analysis_AR  ['مستوى القلق'] = 'مرتفع'


    # In[55]:


    import six

    import arabic_reshaper
    from bidi.algorithm import get_display

    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from googletrans import Translator



    # In[56]:


    def render_mpl_table(data, col_width=5.0, row_height=0.625, font_size=14,tran = '',
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


    # In[57]:


    if(len(analysis_EN) > 0):
        for ind,row in analysis_EN.iterrows():
               analysis_EN.loc[ind,'Nearest  Location']=get_display(arabic_reshaper.reshape(analysis_EN.loc[ind,'Nearest Location']))
    if(len(analysis_AR) > 0):
        for ind,row in analysis_AR.iterrows():
            analysis_AR.loc[ind,'أقرب موقع']=get_display(arabic_reshaper.reshape(analysis_AR.loc[ind,'أقرب موقع']))
            analysis_AR.loc[ind,'مستوى القلق']=get_display(arabic_reshaper.reshape(analysis_AR.loc[ind,'مستوى القلق']))
        analysis_AR = analysis_AR.rename(columns={'الموقع': get_display(arabic_reshaper.reshape('الموقع')) })
        analysis_AR = analysis_AR.rename(columns={'التكرار': get_display(arabic_reshaper.reshape('التكرار')) })
        analysis_AR = analysis_AR.rename(columns={'مستوى القلق': get_display(arabic_reshaper.reshape('مستوى القلق')) })
        analysis_AR = analysis_AR.rename(columns={'أقرب موقع': get_display(arabic_reshaper.reshape('أقرب موقع')) }) 
                           


    # In[58]:


    if(len(analysis_EN) > 0):
        render_mpl_table(analysis_EN, header_columns=0, col_width=5, tran ='Location-EN')
    if(len(analysis_AR) > 0):
        render_mpl_table(analysis_AR, header_columns=0, col_width=5, tran ='Location-AR')


    # # Events with highest level of anxiety Note array will be changed soon
   
    from dateutil.parser import parse
    events = pd.DataFrame()
    testEvents = ['2020-04-21 01:10:02.692263 Test','2020-04-21 11:10:02.692263 Home Work',
              '2020-04-21 12:10:02.692263 Final' ]
    date_result = ''
    name_result = ''
    Final_names = []
    Final_dates = []
    for x in range(0,len(testEvents)):
        result = parse(testEvents[x], fuzzy_with_tokens=True)
        date_result = result[0]
        name_result = result[1]
        Final_dates.append(date_result)
        Final_names.append(name_result)
    

    events ['time'] = Final_dates
    events ['eventName'] = Final_names

    events.time.apply(str)
    events['time'] = pd.to_datetime(events['time'])
    events['date'] = pd.to_datetime(events['time'], format='%Y:%M:%D').dt.date
    events['hour'] = events['time'].apply(lambda time: time.strftime('%H'))
    events.drop(['time'], axis=1, inplace = True)
    events.hour = events.hour.astype(int) 
    events.date = events.date.astype(str)

    Test = Labeled_df
    
    merge_df = pd.merge(left=Test, 
                  right = events,
                  how = 'left',
                  left_on=['hour','date'],
                  right_on=['hour','date']).ffill()
    merge_df['eventName'].fillna("Not given", inplace=True)
    merge_df= merge_df[merge_df.eventName != 'Not given']
    
    
    finalEvents_EN = pd.DataFrame()
    finalEvents_AR = pd.DataFrame()
    ev_name=''
    evNames = []
    evLabels = []
    evDate = []
    for row in merge_df.itertuples():
        if row.eventName != ev_name:
            if row.Label == 'High':
                ev_name = row.eventName
                ev_label = row.Label
                ev_date = row.date
        if(ev_name != ''):
            evNames.append(ev_name)
            evLabels.append(ev_label)
            evDate.append(ev_date)
    
    finalEvents_EN ['Event Name'] = evNames
    finalEvents_EN['Anxiety Level'] = evLabels
    finalEvents_EN['Date'] = evDate

    finalEvents_AR ['اسم الحدث'] = evNames
    finalEvents_AR['مستوى القلق'] = evLabels
    finalEvents_AR['تاريخ الحدث'] = evDate

    
    finalEvents_EN = finalEvents_EN.drop_duplicates()
    finalEvents_AR = finalEvents_AR.drop_duplicates()
    
    if(len(finalEvents_EN) > 0):
        for ind,row in finalEvents_EN.iterrows():
            try:
                finalEvents_EN.loc[ind,'Event Name']=get_display(arabic_reshaper.reshape(
                finalEvents_EN.loc[ind,'Event Name']))
            except:
                pass

    if(len(finalEvents_AR) > 0):
        finalEvents_AR['مستوى القلق'] = 'مرتفع'
        for ind,row in finalEvents_AR.iterrows():
            try:
                finalEvents_AR.loc[ind,'اسم الحدث']=get_display(arabic_reshaper.reshape(finalEvents_AR.loc[ind,'اسم الحدث']))
                finalEvents_AR.loc[ind,'مستوى القلق']=get_display(arabic_reshaper.reshape(finalEvents_AR.loc[ind,'مستوى القلق']))
            except:
                pass
        finalEvents_AR = finalEvents_AR.rename(columns={'اسم الحدث': get_display(arabic_reshaper.reshape('اسم الحدث')) })
        finalEvents_AR = finalEvents_AR.rename(columns={'مستوى القلق': get_display(arabic_reshaper.reshape('مستوى القلق')) })
        finalEvents_AR = finalEvents_AR.rename(columns={'تاريخ الحدث': get_display(arabic_reshaper.reshape('تاريخ الحدث')) })

             
            
    if(len(finalEvents_EN) > 0):
        render_mpl_table(finalEvents_EN, header_columns=0, col_width=4, tran ='Events-EN')
    if(len(finalEvents_AR) > 0):
        render_mpl_table(finalEvents_AR, header_columns=0, col_width=4, tran ='Events-AR')
    # # Genertate patient report and save it in storage

    # In[59]:


    from reportlab.lib import colors


    # In[60]:


    pdfmetrics.registerFont(TTFont('Arabic', 'traditional-arabic/tradbdo.TTF'))


    # In[61]:


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
    pdf.drawString(150,540, "Name: " )
    pdf.drawString(150,520, "Age: " )
    pdf.drawString(150,500, "Employment Status: " )
    pdf.drawString(150,480, "Martial Status: " )
    pdf.drawString(150,460, "Monthly Income: " )
    pdf.drawString(150,440, "Chronic Diseases: " )
    pdf.drawString(150,420, "Cigarette Smoker: " ) 

    pdf.setFont("Helvetica", 15)
    pdf.setFillColor(black)
    pdf.drawString(210,540,  name)
    pdf.drawString(210,520,  age)
    pdf.drawString(310,500,  emp)
    pdf.drawString(260,480,  mar)
    pdf.drawString(290,460,  income)
    pdf.drawString(290,440, chronicD)
    pdf.drawString(290,420,  smoke) 

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

    pdf.drawString(100,320, "Location Analysis")
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
    pdf.drawString(100,650, "Events Analysis")

    if(len(finalEvents_EN)>0):
        pdf.drawImage("Events-EN.png", 30, 500, width=550,height=100)
    else:
        pdf.setFont("Helvetica", 15)
        pdf.setFillColor(colors.HexColor('#23495f'))

        t = pdf.beginText(130,550)
        text = [
        name +" condition was stable through this period,", 
        "no event with high anxiety level were detected." ]
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


    # In[62]:


    def translate(text):
        reshaped_text = arabic_reshaper.reshape(text)
        bidi_text = get_display(reshaped_text)
        return bidi_text


    # In[63]:

    def from_en_to_ar(text):
        translator = Translator()
        result = translator.translate(text, dest='ar')
        return result.text


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


    pdf.setFont("Arabic", 18)
    pdf.drawString(445,540, na )
    pdf.drawString(446,520, ag)
    pdf.drawString(400,500, Es )
    pdf.drawString(390,480, Ms)
    pdf.drawString(400,460, Mi )
    pdf.drawString(395,440, Cd)
    pdf.drawString(445,420, Cs ) 

    pdf.setFont("Arabic", 15)
    pdf.setFillColor(black)
    pdf.drawString(400,540,  name)
    pdf.drawString(420,520,  age)
    pdf.drawString(350,500,  translate(from_en_to_ar(emp)))
    pdf.drawString(355,480,  translate(from_en_to_ar(mar)))
    pdf.drawString(360,460,  income)
    pdf.drawString(360,440,  translate(from_en_to_ar(chronicD)))
    pdf.drawString(420,420,  translate(from_en_to_ar(smoke))) 

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

    loc = translate (u'تحليل المواقع')

    pdf.drawImage("ALpdf.png", 57, 400, width=485,height=200)
    pdf.drawString(390,320, loc)
    if(len(analysis_AR) > 0):
        pdf.drawImage("Location-AR.png", 10, 200, width=570,height=100)
    else:
        pdf.setFont("Arabic", 18)
        pdf.setFillColor(colors.HexColor('#23495f'))

        text = translate (u'كانت الحالة مستقرة خلال هذه الفترة, لم يتم الكشف عن أي مواقع ذات مستوى قلق مرتفع.')
        pdf.drawString(60,250, text)


    pdf.showPage()


    pdf.drawImage("serene .png", 150, 730, width=300,height=130, mask= 'auto')

    pdf.setFont("Arabic", 28)
    pdf.setFillColor(colors.HexColor('#808080'))
    EV = translate(u'تحليل الأحداث')
    pdf.drawString(380,650, EV)


    if(len(finalEvents_AR)>0):
        pdf.drawImage("Events-AR.png", 20, 500, width=550,height=100)
    else:
        pdf.setFont("Arabic", 18)
        pdf.setFillColor(colors.HexColor('#23495f'))

        text = translate (u'كانت الحالة مستقرة خلال هذه الفترة, لم يتم الكشف عن أي أحداث ذات مستوى قلق مرتفع.')
        pdf.drawString(60,550, text)


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


    # In[64]:


    #new method
    doct = storage.child(userID+"/lastGeneratedPatientReport/patientReport-EN").put('Patient-EN.pdf')
    doct = storage.child(userID+"/lastGeneratedPatientReport/patientReport-AR").put('Patient-AR.pdf')


    # In[65]:


    os.remove('Patient-EN.pdf')
    os.remove('Patient-AR.pdf')

    if(len(plot_df)<=90):
         os.remove('ALpdf.png')
    else:
        for x in range(0,len(dfarray)):
            os.remove('AL'+str(x)+'pdf.png')

    if(len(finalEvents_AR) > 0 ):
        os.remove('Events-AR.png')
        os.remove('Events-EN.png')
    if(len(analysis_AR)>0):
        os.remove('Location-AR.png')
        os.remove('Location-EN.png')
    
    # In[ ]:





    # In[ ]:

    freqs = {
        'Message': 'Some msg',
        'Result': "ksu"
    }
    return freqs



