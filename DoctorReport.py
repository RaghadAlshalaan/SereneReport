    #!/usr/bin/env python
    # coding: utf-8

    # In[1]:
def reportDoc(userid, doctorid):

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
    import matplotlib.patches as mpatches


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


    # In[6]:


    today = datetime.datetime.now()
    timestamp = today.strftime("%Y-%m-%d")
    bucket = storage.bucket(app=app)


    # ## Get data from storage and get list of dates (2 weeks)

    # In[7]:


    #get a a list of date between start and end date 
    userID = userid
    doctorID = doctorid
    duration = 15 # two weeks
    dates =[]
    for x in range(0 ,duration):
        today=date.today() #revert to original
        #yesterday = today - datetime.timedelta(days=1)
        start_date = (today-timedelta(days=duration-x)).isoformat()
        dates.append(start_date)


    # In[8]:


    df= pd.DataFrame()
    notAvailableDates = []
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
            notAvailableDates.append(dates[x])
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
            heart_rate = df_heartrate.groupby(["hour"], as_index=False).max()
            heart_rate['sleepMin'] = sleepMinutes
            heart_rate['TotalSteps'] = steps
            heart_rate['date'] = dates[x]
            heart_rate = heart_rate.astype({"hour": int})  
        except:
            notAvailableDates.append(dates[x])
            pass

        # append dataframe
        df = df.append(heart_rate, ignore_index = True)


    # In[9]:


    notAvailableDates
    notSyncedDates = pd.DataFrame()
    notSyncedDates ['date'] = notAvailableDates 


    # In[10]:


    notSyncedDates = notSyncedDates.drop_duplicates()


    # In[11]:


    # ### Get user location

    # In[12]:


    # get location from database
    loc_df = pd.DataFrame()
    locID = []
    locations = db.collection(u'PatientLocations').where(u'patientID', u'==', userID ).stream()

    for location in locations:
        loc1 = location.to_dict()
        locID.append(location.id)
        loc_df = loc_df.append(pd.DataFrame(loc1,index=[0]),ignore_index=True)

    loc_df['id'] = locID



    # In[13]:


    loc_df.drop(['anxietyLevel', 'lat','lng', 'patientID'  ], axis=1, inplace = True)


    # In[14]:


    loc_df.time.apply(str)
    loc_df['time'] = pd.to_datetime(loc_df['time'])
    loc_df['date'] = pd.to_datetime(loc_df['time'], format='%Y:%M:%D').dt.date
    loc_df['hour'] = loc_df['time'].apply(lambda time: time.strftime('%H'))
    loc_df.drop(['time'], axis=1, inplace = True)
    loc_df.hour = loc_df.hour.astype(int) 
    loc_df.date = loc_df.date.astype(str)
    df.date = df.date.astype(str)


    # In[15]:


    dfinal = pd.merge(left=df, 
                      right = loc_df,
                      how = 'left',
                      left_on=['hour','date'],
                      right_on=['hour','date']).ffill()


    # ### Test data into model

    # In[16]:


    #test model 
    train_df = dfinal.rename(columns={'value': 'Heartrate'})


    # In[17]:


    Labeled_df = pd.DataFrame()
    Labeled_df = trainData(train_df)


    # In[18]:


    Labeled_df.drop(['lon'],axis=1, inplace = True)


    # In[19]:


    Labeled_df['name'].fillna("Not given", inplace=True)
    Labeled_df['id'].fillna("Not given", inplace=True)
    Labeled_df['anxiety_assigned'].fillna('Not given', inplace = True)


    # In[20]:


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

    # In[21]:


    # Show the highest level 
    df_high = pd.DataFrame()
    df_high = Labeled_df[Labeled_df.Label == 'High']


    # # Get patient information

    # In[24]:


    docDf = pd.DataFrame()
    doc_ref = db.collection(u'Patient').document(userID)
    doc = doc_ref.get().to_dict()
    docDf = docDf.append(pd.DataFrame(doc,index=[0]),ignore_index=True)


    # In[25]:


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


    # ## Storage intiliazation

    # In[26]:


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


    # # HR

    # In[27]:


    sns.set( rc={'axes.facecolor': '#fcfeff'})


    # In[28]:


    # Take the highest heartrate in a day
    dfhr = pd.DataFrame()
    hr = []
    hrr = 0
    rowCount = 1
    for x in range(0 ,len(dates)):
        rowCount = 0
        for row in df.itertuples():
            if (row.date == dates[x]):
                if(row.value > rowCount):
                    rowCount = row.value
        hr.append(rowCount)


    dfhr['date'] = dates
    dfhr['hr'] = hr


    # In[29]:


    plt.figure(figsize=(20, 7))

    plt.plot(dfhr['date'], dfhr['hr'], color="#a03e3e", linewidth=2)
    plt.xlabel("Date", fontsize=16)
    plt.ylabel("Amplitude (bpm)")
    plt.tick_params(axis='x', rotation=70)
    plt.tight_layout()


    plt.savefig('hr.png', dpi=None)
    plt.show()
    plt.draw()


    # # Steps

    # In[30]:


    dfstep = pd.DataFrame()
    avgSteps = []
    totalsteps = 0
    rowCount = 1
    for x in range(0 ,len(dates)):
        for row in Labeled_df.itertuples():
            if (row.date == dates[x]):
                rowCount += 1
                totalsteps += row.TotalSteps
        avgSteps.append(totalsteps/rowCount)


    dfstep['date'] = dates
    dfstep['Steps'] = avgSteps


    # In[31]:


    # Plot Steps


    plt.figure(figsize=(20,7))
    plt.fill_between(dfstep['date'], dfstep['Steps'], color="#ffd6b0", linewidth=1)
    plt.plot(dfstep['date'], dfstep['Steps'], color="#ff5900", linewidth=1)
    plt.xlabel("Date", fontsize=16)
    plt.ylabel("Total steps")
    plt.tick_params(axis='x', rotation=70)
    plt.tight_layout()

    plt.savefig('steps.png', dpi=None)
    plt.show()
    plt.draw()


    # # Sleep

    # In[32]:


    dfsleep = pd.DataFrame()
    sleeps = []
    totalsleep = 0
    rowCount = 1
    for x in range(0 ,len(dates)):
        for row in df.itertuples():
            if (row.date == dates[x]):
                totalsleep = row.sleepMin
        sleeps.append(totalsleep/60)


    dfsleep['date'] = dates
    dfsleep['sleep'] =  sleeps


    # In[33]:


    figs = dfsleep.plot.bar(x = 'date', y = 'sleep', rot = 70, color= '#3629a6', capstyle = 'round').get_figure()
    figs.set_size_inches(20, 10)
    plt.xlabel('Date')
    plt.ylabel('Sleep (hr)')
    plt.show()
    plt.draw()
    plt.tight_layout()

    figs.savefig('sleep.png', dpi=None)


    # # AL

    # In[34]:


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


    # In[35]:


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


    # In[36]:


    fig, ax = plt.subplots()
    # Draw the stem and circle
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








    fig.savefig('AL.png', dpi = None)


    # # Location Analysis

    # In[37]:


    # get location from database
    new_loc = pd.DataFrame()
    locID = []
    locations = db.collection(u'PatientLocations').where(u'patientID', u'==', userID ).stream()

    for location in locations:
        loc = location.to_dict()
        locID.append(location.id)
        new_loc = new_loc.append(pd.DataFrame(loc,index=[0]),ignore_index=True)

    new_loc['id'] = locID


    # In[38]:


    new_loc.time.apply(str)
    new_loc['time'] = pd.to_datetime(new_loc['time'])
    new_loc['date'] = pd.to_datetime(new_loc['time'], format='%Y:%M:%D').dt.date
    new_loc.drop(['time'], axis=1, inplace = True)
    new_loc.date = new_loc.date.astype(str)


    # In[39]:


    new_loc = new_loc[(new_loc.date >= dates[0]) & (new_loc.date <= dates[len(dates)-1])]


    # In[40]:


    names = []
    Name =""
    for row in new_loc.itertuples():
        Name  = row.nearestLoc         
        names.append(Name)


    # In[41]:


    new_name =pd.DataFrame()
    new_name ['name']= names


    # In[42]:


    new_name = new_name.drop_duplicates()
    new_name.dropna()


    # In[44]:


    fnames = []
    fName =""
    for row in new_name.itertuples():
        fName  = row.name
        fnames.append(fName)


    # In[45]:


    analysis = pd.DataFrame()
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

    analysis ['Location'] = locationNames
    analysis ['Frequency'] = counts
    analysis ['Anxiety Level'] = labels
    analysis ['Nearest Location'] = nearLocs


    # In[46]:


    newA = analysis.drop(analysis[analysis['Frequency'] == 0].index, inplace= True)


    # In[47]:


    import six
    import arabic_reshaper
    from bidi.algorithm import get_display
    import unicodedata

    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.lib import colors


    # In[48]:


    def render_mpl_table(data, col_width=5.0, row_height=0.625, font_size=14,
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

        fig.savefig('Location.png', dpi = 100)
        return ax


    # In[49]:


    if(len(analysis) > 0):
        for ind,row in analysis.iterrows():
            analysis.loc[ind,'Nearest Location']=get_display(arabic_reshaper.reshape(analysis.loc[ind,'Nearest Location']))


    # In[50]:


    if(len(analysis) > 0):
        render_mpl_table(analysis, header_columns=0, col_width=5)
    
    
    # ## Improvements
    
    
    
    # get yesterday improvement 
    today_al = float("{:.2f}".format(plot_df['Anxiety'].mean()))
    improvement = -1
    
    try:
        ID = "Doctor"+userID
        doc_ref = db.collection(u'DoctorReports').document(ID)
        doc = doc_ref.get().to_dict()
        prev = float("{:.2f}".format(doc['anxiety_level']))
        # calculate the improvement
        improvement= float("{:.2f}".format(((prev - today_al)/3)*100 ))     

    
    except:
        improvement = -1
    
    # ## Generate doctor report pdf and store it in database
    # 

    # In[51]:


    pdf = canvas.Canvas('Doctor.pdf')
    pdf.setTitle('Patient report')



    pdf.drawImage("serene .png", 150, 730, width=300,height=130, mask= 'auto')


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

    pdf.drawString(115,330, "Report Duration From: (" + dates[0] +" To: "+ dates[len(dates)-1]+ ")")

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(250,300, "Improvements: ")

    pdf.setFont("Helvetica-Bold", 20)
    if(improvement == -1):
        pdf.drawString(290,260, "--")

    else:
        pdf.drawString(280,260, improvement +'%')


    pdf.showPage()



    pdf.drawImage("serene .png", 150, 730, width=300,height=130, mask= 'auto')



    pdf.setFont("Helvetica-Bold", 20)
    pdf.setFillColor(colors.HexColor('#808080'))

    pdf.drawString(100,650, "Anxiety Level")
    pdf.drawImage("AL.png", 57, 400, width=485,height=200)


    pdf.drawString(100,330, "Heart Rate")
    pdf.drawImage("hr.png", 57, 100, width=485,height=200)

    pdf.showPage()

    pdf.drawImage("serene .png", 150, 730, width=300,height=130, mask= 'auto')

    pdf.setFont("Helvetica-Bold", 20)
    pdf.setFillColor(colors.HexColor('#808080'))

    pdf.drawString(100,650, "Activity Rate")
    pdf.drawImage("steps.png", 57, 400, width=485,height=200)

    pdf.drawString(100,350, "Sleep Analysis")
    pdf.drawImage("sleep.png", 57, 100, width=485,height=200)

    pdf.showPage()

    pdf.drawImage("serene .png", 150, 730, width=300,height=130, mask= 'auto')

    pdf.setFont("Helvetica-Bold", 20)
    pdf.setFillColor(colors.HexColor('#808080'))

    pdf.drawString(100,650, "Location Analysis")

    if(len(analysis) > 0):
        pdf.drawImage("Location.png", 30, 200, width=570,height=100)

    else:
        pdf.setFont("Helvetica", 15)
        pdf.setFillColor(colors.HexColor('#23495f'))
        t = pdf.beginText(130,550)
        text = [
        name +" condition was stable through this period,", 
        "no locations with high anxiety level were detected." ]
        for line in text:
            t.textLine(line)

        pdf.drawText(t)


    if(len(notSyncedDates) != 0):  
        pdf.setFont("Helvetica", 12)
        pdf.setFillColor(colors.HexColor('#d40027'))
        pdf.drawString(75,100, "Note: Below dates are missing, because they were not synced correctly:")
        i = 70
        for row in notSyncedDates.itertuples():
            pdf.drawString(85,i, '- '+ row.date)
            i = i-20



    pdf.save()



    # In[52]:


    #new method
    doct = storage.child(userID+"/DoctorReport/doctorReport").put('Doctor.pdf')


    # In[53]:


    linkDF = pd.DataFrame()
    linkDF = linkDF.append(pd.DataFrame(doct,index=[0]),ignore_index=True)


    # In[54]:


    token1 = linkDF['downloadTokens'].values
    token = token1[0]
    link = storage.child(userID+"/DoctorReport/doctorReport").get_url(token)


    # In[55]:


    from datetime import  datetime


    # In[56]:


    date = datetime.now()


    


    # get before yesterday to calculate the improvement
    
    
    ID = "Doctor"+userID
    doc_rec = db.collection(u'DoctorReports').document(str(ID))
    doc_rec.set({
            u'doctorId': doctorID ,
            u'emailsent':True,
            u'patientId': userID,
            u'reportTime': date,
            u'reportUrl': link,
            u'improvement': improvement,
            u'anxiety_level': today_al

        })



    # In[59]:


    os.remove("hr.png")
    os.remove("sleep.png")
    os.remove("AL.png")
    os.remove("steps.png")
    os.remove("Doctor.pdf")
    if(len(analysis)>0):
        os.remove("Location.png")


    # In[62]:


    freqs = {
        'Doctor': 'YAAAY'
    }
    return freqs


    # In[ ]:

