import streamlit as st
import pandas as pd
from prophet import Prophet
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
from sklearn.metrics import mean_squared_error, mean_absolute_error
#st.cache(suppress_st_warning=True) 
#is a caching mechanism that allows your app to stay performant even when loading data from the web, manipulating large datasets, or performing expensive computations.

#For a cleaner, more readable code below and easier maintenance later I define all larger texts here and store them in a variable
Home_AboutHelvetas = " **Helvetas is a Swiss based INGO in development cooperation** Together with partners Helvetas tackles the global challenges at various levels: with projects on the ground, with expert advice and by advocating for conducive framework conditions benefiting the poor. This triple commitment is empowering people and transforming lives. Helvetas follows a multi-stakeholder approach by linking civil society actors, governments and private sector. Helvetas is active in the following areas: water, food and climate, education, jobs and private sector development, governance, gender and social equity. Helvetas engages in emergency relief, reconstruction and rehabilitation. In addition to rural areas, Helvetas is increasingly involved in urban development and is focusing its work on young women and men. (https://www.helvetas.org/en/switzerland/who-we-are/vision-mission)"
Home_AboutSDG = "The Sustainable Development Goals (SDGs), also known as the Global Goals, were adopted by the United Nations in 2015 as a universal call to action to end poverty, protect the planet, and ensure that by 2030 all people enjoy peace and prosperity. The 17 SDGs are integrated—they recognize that action in one area will affect outcomes in others, and that development must balance social, economic and environmental sustainability. Countries have committed to prioritize progress for those who're furthest behind. The SDGs are designed to end poverty, hunger, AIDS, and discrimination against women and girls. The creativity, knowhow, technology and financial resources from all of society is necessary to achieve the SDGs in every context. (https://www.undp.org/sustainable-development-goals)"


#Function to create a grid for easier Formating the Output
def make_grid(rows,cols):
    grid = [0]*rows
    for i in range(rows):
        with st.container():
            grid[i] = st.columns(cols)
    return grid


#Function for Prediction
@st.cache
def make_sdg_prediction(dataset, country, goal):
    periods = 3
    df =  dataset
    country = country
    goal = goal

    #prepare dataset
    df_p = df[df['Country'] == country].copy()
    df_p['Year'] = pd.to_datetime(df_p['Year'].map(str))
    df_p['ds'] = pd.to_datetime(df_p['Year'])
    df_p['y'] = df_p[goal].copy()
    #define an train the model
    m = Prophet()
    m.fit(df_p)
    
    #Create Yearly Future Dataset
    future = m.make_future_dataframe(freq = "YS", periods=periods)
    forecast = m.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], df_p #return predictiom and actuall data


@st.cache
def prepare_dataframe(file,tab, helvetas_countries):
    #This function loads and transforms the dataset, it only need to be called once. Therfore as sperate function so that st.cache can be used
    df = pd.read_excel(file, tab)
    #Helvetas Dataframe
    df_hel = df.loc[:, :'Goal 17 Score'].copy() #only goals
    goal_list =  df_hel.loc[:, 'Goal 1 Score':'Goal 17 Score'].columns
    #Create a ranking for Each country and each goals (Overall)
    for y in df_hel['Year'].unique():#df_ranked[df_ranked['Year']==y]
        df_hel.loc[df_hel['Year']==y, 'SDG_Index_Rank'] = df_hel[df_hel['Year']==y]['SDG Index Score'].rank(ascending = False)
        rank_col_names = ['Goal '+str(i)+' Rank' for i in range(1,len(goal_list)+1)]
        for y in df_hel['Year'].unique():
            for goal_column in goal_list:
                df_hel.loc[df_hel['Year']==y, goal_column+'_rank'] = df_hel[df_hel['Year']==y][goal_column].rank(ascending = False)
        ####
    df_hel =  df_hel[df_hel['Country'].isin(helvetas_countries)] #keep only Helvetas countries
    return df_hel


######################################
#Plot Function
######################################

def plot_predictions(predicted_values_df, actual_values_df):
    import matplotlib.pyplot as plt
    plt.plot(predicted_values_df['ds'][-4:],predicted_values_df['yhat'][-4:], label ='forcasted_values')
    plt.plot(actual_values_df['ds'],actual_values_df['y'], label ='actual_values')
    #plt.plot(x, y2, '-.', label ='y2')
    plt.xlabel("X-axis data")
    plt.ylabel("Y-axis data")
    plt.legend()
    plt.title('multiple plots')
    plt.show()

#Helper function to create a profile for each Country
def country_profile(country):
    #df_hel[(df_hel['Country'] == 'Benin') & (df_hel['Year'] == 2021)]
    yes = df_hel[(df_hel['Country'] == country) & (df_hel['Year'] == 2021)][['Year']].iloc[0,0]
    pos = df_hel[(df_hel['Country'] == country) & (df_hel['Year'] == 2021)][['Population']].iloc[0,0].round(0)
    res = df_hel[(df_hel['Country'] == country) & (df_hel['Year'] == 2021)][['Region']].iloc[0,0]
    ins = df_hel[(df_hel['Country'] == country) & (df_hel['Year'] == 2021)][['Income Group']].iloc[0,0]
    return yes, pos, res, ins


#Load and Prepare the Dataset
with st.spinner("Loading an preparing data..."):
    #df = pd.read_excel('SDR-2022-database.xlsx', 'Backdated SDG Index')

    #Create a list of Helvetas Countries for each region    
    africa = ['Benin','Burkina Faso','Ethiopia','Madagascar','Mali','Mozambique','Niger','Tanzania']
    asia = ['Bangladesh','Bhutan','Kyrgyz Republic','Lao PDR','Myanmar','Nepal','Pakistan','Sri Lanka','Tajikistan','Vietnam']
    easter_europe = ['Albania','Bosnia and Herzegovina','Moldova','North Macedonia','Serbia']
    latin_america = ['Bolivia','Guatemala','Haiti','Honduras','Peru']

    #combine the Regional Lists to a complete list
    helvetas_countries = africa + asia + easter_europe + latin_america
    #Load and Prepare the dataset, only relevant countries, and columns
    df_hel = prepare_dataframe('SDR-2022-database.xlsx','Backdated SDG Index', helvetas_countries)

    
    st.success("Ready to go!")

    ###############
    
st.sidebar.header("SDG Indexes Prediction App")
app_mode = st.sidebar.selectbox('Select Page',['Home','Predictions']) #two pages
# Titel / Text Information
#text / title
### Navigation and outut

#SDG Nanme liat
#Create a Dataframe with all Goal Infos and the link to the choice

#List of more explanatory goal names
sdg_goal_names = ['SDG1: No Poverty', 'SDG2: No Hunger','SDG3: Good Health and Well-Being','SDG4: Quality Education','SDG5: Gender Equality','SDG6: Clean Water and Sanitation','SDG7: Affordable and Clean Energy','SDG8: Decent Work and Economic Growth','SDG9: Industry, Innovation and Infrastructure','SDG10: Reduced Inequalities','SDG11: Sustainable Cities and Communities','SDG12: Responsible Consumption and Production','SDG13: Climate Action','SDG14: Life Below Water','SDG15: Life on Land','SDG16: Peace, Justice and Strong Institutions','SDG17: Partnerships for the Goals']
goal_list =  df_hel.loc[:, 'Goal 1 Score':'Goal 17 Score'].columns

#Create a Dataframe that contains all Names and Links to a Wikipedia Page with aditional information for the SDGs
linklist = [ "https://en.wikipedia.org/wiki/Sustainable_Development_Goal_"+str(i) for i in range(1,18)] #Wikipedia has an entry for each SDG, so we can use the basic link and only vary the number
goal_info = pd.DataFrame()
goal_info['idx'] = goal_list
goal_info['name'] = sdg_goal_names
goal_info['link'] = linklist

#Make a dictionary with more explanatory names and the column names of the dataframe
goal_dict = {sdg_goal_names[i]: goal_list[i] for i in range(len(sdg_goal_names))}

########################################################    
#Home Screen
########################################################
if app_mode=='Home':
    logo_grid = make_grid(1,2)
    logo_grid[0][0].image('helvetas_logo1.gif', width = 200)
    logo_grid[0][1].image('icons/E_SDG_logo_UN_emblem_horizontal_WEB.jpg')
    st.header("Predic SDG Goal Indexes for Helvetas Partner Countries")
    st.markdown(" This app let you **predict the SDG Indexes** for Helvetas Partner Countries using  **Machine learning** Algorithms.")
    st.subheader("About Helvetas")
    st.markdown(Home_AboutHelvetas)
    st.subheader("About SDG")
    st.markdown(Home_AboutSDG)
    
    st.subheader("About this App")
    st.markdown("This app has been created by Patric Masar on December 2022 as part of the Predictive Analytics Courses of the EDHC MSc in Data Science and Business Analytics")

########################################################    
#Prediction Screen
########################################################


elif app_mode == 'Predictions':
    st.image('helvetas_logo1.gif')
    img_list=[['E-WEB-Goal-01.png','E-WEB-Goal-02.png','E-WEB-Goal-03.png','E-WEB-Goal-04.png','E-WEB-Goal-05.png','E-WEB-Goal-06.png'],['E-WEB-Goal-07.png','E-WEB-Goal-08.png','E-WEB-Goal-09.png','E-WEB-Goal-10.png','E-WEB-Goal-11.png','E-WEB-Goal-12.png'],['E-WEB-Goal-13.png','E-WEB-Goal-14.png','E-WEB-Goal-15.png','E-WEB-Goal-16.png','E-WEB-Goal-17.png']]
    
    img_list_flat = [img for cont_list in img_list for img in cont_list]
    #display all single image in a grid
    #goal_grid = make_grid(3,6)
  
    #for i in range(len(img_list)):
    #    for z in range(len(img_list[i])):
    #        print(i,' ',z, img_list[i][z])
    #        goal_grid[i][z].image('icons/'+img_list[i][z], width=80)
    

    st.image('sdg_poster.png')
    st.sidebar.header("Choose a Country and a goal :")
    
    #Select Goal to predict
    goal_s = st.sidebar.selectbox("Goal", sdg_goal_names)
    goal = goal_dict[goal_s]
    
    #Select Country to Predict
    p_selection =[]
    p_country_af = st.sidebar.multiselect("Africa", africa)
    p_country_as = st.sidebar.multiselect("Asia", asia)
    p_country_ee = st.sidebar.multiselect("Eastern Europe", easter_europe)
    p_country_la = st.sidebar.multiselect("Latin America", latin_america)
    p_selection = p_country_af+p_country_as+p_country_ee+p_country_la
    
    
    
    st.write(goal_info[goal_info['idx'] == goal][['name']].iloc[0,0])
    #plot the icon of the selected goal
    
    st.image('icons/'+img_list_flat[sdg_goal_names.index(goal_s)], width = 80)
    st.write('Click on the Link below to learn more about your selected SDG Goal')
    st.write(goal_info[goal_info['idx'] == goal][['link']].iloc[0,0])
    
    #Prepare Plot Dataframe with the additional, to predict periods
    df_plot = pd.DataFrame()
    periods = 3
    pd.date_range(start = pd.to_datetime(str(min(df_hel['Year']))), end = pd.to_datetime(str(str(max(df_hel['Year'])+periods+1))), freq='YS')
    df_plot['Year'] = pd.date_range(start = pd.to_datetime(str(min(df_hel['Year']))), end = pd.to_datetime(str(max(df_hel['Year'])+periods)), freq='YS')
    
    #Make Prediction for each selected country and store the in the dF_Plot datafarme formated for plotting
    if p_selection == []:
        p_selection = ['Mali'] #set  default value
    for country in p_selection:
        pred, act = make_sdg_prediction(df_hel,country,goal)

        df_temp = pred.iloc[-3:][['ds','yhat']].rename(columns = {'ds':'Year', 'yhat':country+'Yhat:'})
        df_plot = pd.merge(df_plot, act[['Year',goal]], how = "outer").rename(columns = {goal:country})
        df_plot = pd.merge(df_plot, df_temp, how = "outer")
 
    

    #plot_predictions(pred, act)
    st.line_chart(df_plot, x='Year' )
    
    
    #print output about countries in 2 Columns: On Left Text Information, on the Right a graph ploting the development of the Overall Ranking
    pred_grid = make_grid(len(p_selection),2)
    count = 0
    for country in p_selection:
        
        #Prepare text for country profile because streamlite does not allow combin text and variable in this situation (its only possible whe using directly st.write()
        #Call country_profile function to
        year, pop, reg, inc = country_profile(country)
        prof = '**Country Infos for '+country+'**'
        year = 'Date for: '+str(year)
        pop = 'Population: '+str(pop)
        reg = 'Region: '+reg
        inc = ' Income Group: '+inc                     
        
        p_selection_Yhat = country+'Yhat:' #Column name in df_plot
        
        y_max = int(df_plot['Year'].dt.strftime('%Y').max())
        predl = df_plot[-3:][p_selection_Yhat].round(2).values
    
        #pred_txt = ''
        #for i in range(0, len(predl)):
        #    pred_txt = pred_txt+str(y_max-2 + i) +' : '+ str(predl[i])+'\n'
        #pred_txt = 'Prediction for the next Years:'
        
        #Create lsit with the Year and the Predictions for easier print 
        pred_next_years = [str(y_max-2 + i) +' : '+ str(predl[i]) for i in range(0, len(predl))] #The three last years in the Dataframe are the predicted ones, in ascending order
        #print country Information
        pred_grid[count][0].markdown(prof)
        pred_grid[count][0].write(year)
        pred_grid[count][0].write(pop)
        pred_grid[count][0].write(reg)
        pred_grid[count][0].write(inc)
        pred_grid[count][0].write("Prediction for the next Years:")
        pred_grid[count][0].write(pred_next_years[0])
        pred_grid[count][0].write(pred_next_years[1])
        pred_grid[count][0].write(pred_next_years[2])
        
       
        line_chart_rank_1 = alt.Chart(df_hel[df_hel['Country'] == country][['Year','SDG_Index_Rank','Goal 3 Score_rank']]).mark_line().encode(
        y= 'SDG_Index_Rank',
        x='Year')
        
        line_chart_rank_2 = alt.Chart(df_hel[df_hel['Country'] == country][['Year','SDG_Index_Rank','Goal 1 Score_rank']]).mark_line().encode(
        y= 'Goal 1 Score_rank',
        x='Year')
        cl = alt.layer(line_chart_rank_1,line_chart_rank_2)
        
        pred_grid[count][1].write('Overall Ranking for tha country in total and selected SDG')
        pred_grid[count][1].altair_chart(cl, use_container_width=True)
        #st.altair_chart(cl, use_container_width=True)
        
        count = count +1
        #st.altair_chart(line_chart_rank, use_container_width=True)
    
    #Plot Population this should give just some additional information for the user.
    #Create a dataset to plot Country Population
    df_population = pd.DataFrame()
    df_population['Year'] = pd.date_range(start = pd.to_datetime(str(min(df_hel['Year']))), end = pd.to_datetime(str(max(df_hel['Year']))), freq='YS').year
    for country in p_selection:
        df_population = pd.merge(df_population, df_hel[df_hel['Country']==country][['Year','Population']], how="right", on=["Year"]).rename(columns = {'Population':country})

    st.write('To give you a bit of a context and se how many people are affected by this goal, see the pupulation development for the selected countries')
    st.line_chart(df_population, x='Year' )


    
    
    
    
    