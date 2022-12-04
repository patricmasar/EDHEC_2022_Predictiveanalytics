###############################################
# Patric Masar
# Dec 04 2022
# EDHEC MSc Data Science and Business Analytics
# Final Assignment Course: Predictive Analytics
#
# An App to predict the SDG Indexes for the next 3 yeasr for Helvetas Partner Countries.
#
###############################################

import streamlit as st
import pandas as pd
from prophet import Prophet
import numpy as np
import matplotlib.pyplot as plt
import altair as alt
from sklearn.metrics import mean_squared_error, mean_absolute_error


#For a cleaner, more readable code below and easier maintenance later I define all larger texts here and store them in a variable

AppGoal ="This app lets you **predict the SDG Indexes** for for the upcoming 3 years for Helvetas partner countries using  **Machine learning**.  The index values are in the range of 1-100 while 100 is the best value that can be achive"

GraphInfo = "The graph below gives you a first overview by showing a consolidated index value of all goals per country. You can select a specific year using the slider."

WhyPredictSDG = "All **projects of Helvetas contribute to achieving the SDGs**. Having an overview of the development of the SDGs in the countries where Helvetas is active is relevant for the board of directors, the management board,  and country directors to make operational and strategic decisions. The forecast for future development, together with other information, supports planning. It allows conclusions to be drawn about where to put the focus and where to invest in the coming years. The prediction is limited a **strategy period of Helvetas (3 Years)**. The development is very dependent on external influences (political events, climate, global economy, environmental catastrophes, etc.) which can greatly change the development of the target within a short period of time. A forecast further into the future would therefore be very inaccurate."

Home_AboutHelvetas = " **Helvetas is a Swiss based INGO in development cooperation** Together with partners Helvetas tackles the global challenges at various levels: with projects on the ground, with expert advice and by advocating for conducive framework conditions benefiting the poor. This triple commitment is empowering people and transforming lives. Helvetas follows a multi-stakeholder approach by linking civil society actors, governments and private sector. Helvetas is active in the following areas: water, food and climate, education, jobs and private sector development, governance, gender and social equity. Helvetas engages in emergency relief, reconstruction and rehabilitation. In addition to rural areas, Helvetas is increasingly involved in urban development and is focusing its work on young women and men. (https://www.helvetas.org/en/switzerland/who-we-are/vision-mission)"
Home_AboutSDG = "The Sustainable Development Goals (SDGs), also known as the Global Goals, were adopted by the United Nations in 2015 as a universal call to action to end poverty, protect the planet, and ensure that by 2030 all people enjoy peace and prosperity. The 17 SDGs are integrated—they recognize that action in one area will affect outcomes in others, and that development must balance social, economic and environmental sustainability. Countries have committed to prioritize progress for those who're furthest behind. The SDGs are designed to end poverty, hunger, AIDS, and discrimination against women and girls. The creativity, knowhow, technology and financial resources from all of society is necessary to achieve the SDGs in every context. (https://www.undp.org/sustainable-development-goals)"
AboutThisApp = "This app has been created by Patric Masar (patric.masar@edhec.com) in December 2022 as part of the course in 'Predictive Analytics' in  the **EDHEC MSc in Data Science and Business Analytics**"

sources = "- Datasource SDG Index Data: Sustainable Development Report:  https://www.sdgindex.org/ Direct link to dataset: https://dashboards.sdgindex.org/static/downloads/files/SDR-2022-Database.xlsx \n - SDG Icons and Logos: UN Sustainable Development Goals Website: https://www.un.org/sustainabledevelopment/news/communications-material/ \n- Information about Helvetas partner countries: Helvetas Website https://www.helvetas.org/en/switzerland/what-we-do/where-we-work/partner-countries \n- Helvetas Logo: Helvetas Marketing and Communication Team: https://www.helvetas.org/en/switzerland/how-you-can-help/follow-us/media \n- Additional Detail about Income class: https://blogs.worldbank.org/opendata/new-world-bank-country-classifications-income-level-2022-2023"


#Function to create a grid for easier Formating the Output
def make_grid(rows,cols):
    """ Create a grid that to position streamlit elements.
    Create a grid that gives the oportunity to display an element on a specific place in streamlit.

    
    Parameters
    ----------
    arg1: int
    Number of rows for the grid
    
    arg2: int
    Number of columns for the grid 
    
    Example:
    --------
    demo_grid = make_grid(2,2)
    demo_grid[0][0].write('This is the upper left cell of a 2 by 2 grid)
    demo_grid[1][1].write('This is the lower right cell of a 2 by 2 grid') """
    
    grid = [0]*rows
    for i in range(rows):
        with st.container():
            grid[i] = st.columns(cols)
    return grid

#@st.cache is a caching mechanism that allows your app to stay performant even when loading data from the web, manipulating large datasets, or performing expensive computations.
#Function for Prediction
@st.cache
def make_sdg_prediction(dataset, country, goal):
    """ Make a prediction for the futer SDG Index value of a given country and goal.
    This function calculates the next 3 years of the SDG index for a given country and a given goal

    
    Parameters
    ----------
    arg1: DataFrame
    A Dataframe that contains the relevant history of the SDG to predict. It has to have the format that is createt by the function (prepare_dataframe()) 
    and needs to have at least the follwoing structure:
    #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
    Year          616 non-null    int64  
    Country       616 non-null    object 
    Goal X Score  616 non-null    float64  (the X means the number of the goal and is in range 1 to 17. 
    The column could be named different for this function, but the rest of the App relays on this naming conventions.
    
    
    arg2: str
    Name of the Country to make the prediction written the same way as it is in the Country column of the passed Dataframe
    
    arg2: str
    Name of the Column that contains the time series of the goal to predict, ('Goal 1 Score' ... 'Goal 17 Score'
    
    Returns:
    --------
    1: Dataframe with the predicted values in Format: 'ds', 'yhat', 'yhat_lower', 'yhat_upper'
    2: Datframe with the Actual Values in Format: 'ds', 'y', 'Year'
    
    Example:
    --------
    forcast, actual = make_sdg_prediction(df, 'Mali', 'Goal 4 Score' ) """
    
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

#This function loads and transforms the dataset, it only need to be called once. Therfore as sperate function so that st.cache can be used
@st.cache
def prepare_dataframe(file,tab, helvetas_countries):

    """ Reads the data and prepares the dataframe.
    Reads the data and converts it in the way we need it for further processing
    It does the followong transformation
    - For each goal it adds a an additional column that contains the Ranking of a country in that goal
    - it removes unused columns
    - it removes rows with countries that are not Helvetas countries

    
    Parameters
    ----------
    arg1: str
    Filename of the datasource. Datasource needs to be in the exact format as it can be downloaded form the original source:
    https://dashboards.sdgindex.org/static/downloads/files/SDR-2022-Database.xlsx
    
    arg2: str
    Name of the Excel sheet that contains the relevant data, original name is 'Backdated SDG Index'
    
    arg3: list of strings
    A list of all Helvetas Partner Countries
    
    Returns:
    --------
    A Datafrem that contains all the relevant columns for further processing in this app in the following format:
    #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
    Country Code ISO3   616 non-null    object 
    Country             616 non-null    object 
    Year                616 non-null    int64  
    Population          616 non-null    float64
    Region              616 non-null    object 
    Income Group        616 non-null    object 
    SDG Index Score     616 non-null    float64
    Goal 1 Score        616 non-null    float64
    Goal 2 Score        616 non-null    float64
    ........
    Goal 17 Score       616 non-null    float6
    SDG_Index_Rank      616 non-null    float64
    Goal 1 Score_rank   616 non-null    float64
    .....
    Goal 17 Score_rank   616 non-null    float64
    
    Example:
    --------
    df_prepared = prepare_dataframe('SDR-2022-database.xlsx','Backdated SDG Index', helvetas_countries)
    
    """
    df = pd.read_excel(file, tab)
    df_hel = df.loc[:, :'Goal 17 Score'].copy() #only goals and basic infos
    goal_list =  df_hel.loc[:, 'Goal 1 Score':'Goal 17 Score'].columns  #create a list of all column names to easer lop through
    #Create a ranking for Each country and each goals (Overall)
    for y in df_hel['Year'].unique(): #Create a ranking for each Year
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
    """ Helper function that returns avluable about a country.
    Extracts and returns some information fro the source daata that are displayed later

    
    Parameters
    ----------
    arg1: str
    Country of interest
    
    Returns
    1: int: lates year in the dataframe, because the data is extracted for the mos recent year
    2: float Population of the country in the most recent year
    3: str Region where the country is located
    4: str Income Class defined by Worldbank, see here (https://blogs.worldbank.org/opendata/new-world-bank-country-classifications-income-level-2022-2023)
    
    Example:
    --------
    country_profile('Mali') 
    Return: (2021, 20855724.0, 'Africa', 'LIC')
    """
    max_year = df_hel['Year'].max()
    yes = df_hel[(df_hel['Country'] == country) & (df_hel['Year'] == max_year)][['Year']].iloc[0,0]
    pos = df_hel[(df_hel['Country'] == country) & (df_hel['Year'] == max_year)][['Population']].iloc[0,0].round(0)
    res = df_hel[(df_hel['Country'] == country) & (df_hel['Year'] == max_year)][['Region']].iloc[0,0]
    ins = df_hel[(df_hel['Country'] == country) & (df_hel['Year'] == max_year)][['Income Group']].iloc[0,0]
    return yes, pos, res, ins



#Load and Prepare the Dataset
with st.spinner("Loading an preparing data..."):

    #Create a list of Helvetas Countries for each region    
    africa = ['Benin','Burkina Faso','Ethiopia','Madagascar','Mali','Mozambique','Niger','Tanzania']
    asia = ['Bangladesh','Bhutan','Kyrgyz Republic','Lao PDR','Myanmar','Nepal','Pakistan','Sri Lanka','Tajikistan','Vietnam']
    easter_europe = ['Albania','Bosnia and Herzegovina','Moldova','North Macedonia','Serbia']
    latin_america = ['Bolivia','Guatemala','Haiti','Honduras','Peru']

    #combine the Regional Lists to a complete list
    helvetas_countries = africa + asia + easter_europe + latin_america
    
    #Load and Prepare the dataset, only relevant countries, and columns are kept
    df_hel = prepare_dataframe('SDR-2022-database.xlsx','Backdated SDG Index', helvetas_countries)
    
    st.success("Ready to go!")

    ###############
    
st.sidebar.header("SDG indexes Prediction App")
with st.sidebar:
        st.write("Select Home to get general information about the app or Prediction  to get more information about the countries and their SDG indexes")
app_mode = st.sidebar.selectbox('Change the page',['Home','Predictions']) #two pages



#List of more explanatory goal names used for the the selection in the gui
sdg_goal_names = ['SDG1: No Poverty', 'SDG2: No Hunger','SDG3: Good Health and Well-Being','SDG4: Quality Education','SDG5: Gender Equality','SDG6: Clean Water and Sanitation','SDG7: Affordable and Clean Energy','SDG8: Decent Work and Economic Growth','SDG9: Industry, Innovation and Infrastructure','SDG10: Reduced Inequalities','SDG11: Sustainable Cities and Communities','SDG12: Responsible Consumption and Production','SDG13: Climate Action','SDG14: Life Below Water','SDG15: Life on Land','SDG16: Peace, Justice and Strong Institutions','SDG17: Partnerships for the Goals']

#Create a list of all columns names that contains index values for the SDG
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
    
    #Show the Helvetas and SDG Logo on top
    logo_grid = make_grid(1,2)
    logo_grid[0][0].image('helvetas_logo1.gif', width = 200)
    logo_grid[0][1].image('icons/E_SDG_logo_UN_emblem_horizontal_WEB.jpg')
    
    #print information for the suer
    
    st.header("Predic SDG indexes for Helvetas partner countries for the next 3 years")
    st.markdown(AppGoal)
    st.markdown(GraphInfo)
    st.markdown("For details and the predictions go to the prection page, by selecting 'Prediction' on the left.")
    

    # A Slider to sleect the year we want to see in the Bar charts
    s_year = st.slider(" Select a year:", 2000,2021)

    # Create a sorted dataframe to with the relevant information for the bar charts
    df_x = df_hel[df_hel['Year']==s_year][['Country','SDG Index Score']].sort_values(by='SDG Index Score',ascending=False)
    df_x[['Country','SDG Index Score']] = df_x[['Country','SDG Index Score']].round(decimals = 2)
    
    #Plot a bar chart that shows the total SDG Index Score for all countries in the selected year
    st.write(alt.Chart(df_x).mark_bar().encode(
        x=alt.X('Country', sort=None),
        y='SDG Index Score',
        #color ='b' 
    ).configure_mark(
        opacity=1,
        color='blue'
    ))
    
    #More information for the user
    st.subheader("Why do we predict the future development of the SDG ?")
    st.markdown(WhyPredictSDG)
    
    st.subheader("About Helvetas")
    st.markdown(Home_AboutHelvetas)
    st.subheader("About SDG")
    st.markdown(Home_AboutSDG)
    
    st.subheader("About this App")
    st.markdown(AboutThisApp)
    
    st.subheader("Data and media Sources")
    st.markdown(sources)

########################################################    
#Prediction Screen
########################################################


elif app_mode == 'Predictions':
    st.image('helvetas_logo1.gif')
    
    #Store all Image names for the SDG Icons in a sortet list for accessing the files later more easily. List has been nested so that we can print each icon separtly in a 6*3 grid, which is the
    # Offical UN way to display all Icons
    img_list=[['E-WEB-Goal-01.png','E-WEB-Goal-02.png','E-WEB-Goal-03.png','E-WEB-Goal-04.png','E-WEB-Goal-05.png','E-WEB-Goal-06.png'],['E-WEB-Goal-07.png','E-WEB-Goal-08.png','E-WEB-Goal-09.png','E-WEB-Goal-10.png','E-WEB-Goal-11.png','E-WEB-Goal-12.png'],['E-WEB-Goal-13.png','E-WEB-Goal-14.png','E-WEB-Goal-15.png','E-WEB-Goal-16.png','E-WEB-Goal-17.png']]
    
    #for showing the selected Icon ist easier to acess it from a flat list, tahts why we flatten the list
    img_list_flat = [img for cont_list in img_list for img in cont_list]
    
    # User Info and SDG Poster
    st.image('sdg_poster.png')
    st.sidebar.header("Choose one goal to predict:")
    
    #Select Goal to predict
    goal_s = st.sidebar.selectbox("Goal", sdg_goal_names)
    goal = goal_dict[goal_s] #lookup the column name of the selected goal in the dictionary and stor it in the variable goal
    
    st.sidebar.header("Choose one or several countries:")
    #Select Countries to Predict. Whily on one goal can be selected at a time ist possible to select more than one country
    p_selection =[]
    p_country_af = st.sidebar.multiselect("Africa", africa)
    p_country_as = st.sidebar.multiselect("Asia", asia)
    p_country_ee = st.sidebar.multiselect("Eastern Europe", easter_europe)
    p_country_la = st.sidebar.multiselect("Latin America", latin_america)
    p_selection = p_country_af+p_country_as+p_country_ee+p_country_la
    
    
    
    st.write(goal_info[goal_info['idx'] == goal][['name']].iloc[0,0])
    
    #plot the icon of the selected goal by accessing the filename via index in img_list_flat
    st.image('icons/'+img_list_flat[sdg_goal_names.index(goal_s)], width = 80)
    st.write('Click on the Link below to learn more about your selected SDG Goal')
    st.write(goal_info[goal_info['idx'] == goal][['link']].iloc[0,0])  #Plot Link to wikipedia Page of the goal
    
    #Prepare Plot Dataframe with the additional, to predict periods
    df_plot = pd.DataFrame()
    periods = 3
    pd.date_range(start = pd.to_datetime(str(min(df_hel['Year']))), end = pd.to_datetime(str(str(max(df_hel['Year'])+periods+1))), freq='YS')
    df_plot['Year'] = pd.date_range(start = pd.to_datetime(str(min(df_hel['Year']))), end = pd.to_datetime(str(max(df_hel['Year'])+periods)), freq='YS')
    
    #Make Prediction for each selected country and store the in the df_Plot datafarme formated for plotting
    if p_selection == []:
        p_selection = ['Mali'] #set  default value
    for country in p_selection:
        pred, act = make_sdg_prediction(df_hel,country,goal)

        df_temp = pred.iloc[-3:][['ds','yhat']].rename(columns = {'ds':'Year', 'yhat':country+'Yhat:'})
        df_plot = pd.merge(df_plot, act[['Year',goal]], how = "outer").rename(columns = {goal:country})
        df_plot = pd.merge(df_plot, df_temp, how = "outer")
        
    

    #Plot the predictions in a line graph
    st.markdown("See below the **past development** and the **predcition for the next 3 years** of the selected SDG and the selected countries, remember the scale of the indexes goes from 1 to 100, the visible scale is adjusted to the actual data to see more details. So please check the Y axis range")
    st.line_chart(df_plot, x='Year' )
    
    
    #print output about countries in 2 Columns: On left Text Information, on the right a graph ploting the development of the overall ranking
    pred_grid = make_grid(len(p_selection),2)
    count = 0
    #loop through all selected countries
    for country in p_selection:
        
        #Prepare text for country profile because streamlite does not allow combin text and variable in this situation (its only possible whe using directly st.write()
        #Call country_profile function to
        year, pop, reg, inc = country_profile(country)
        prof = '#### Country Infos for '+country+''
        year = '**Data is for:** '+str(year)
        pop = '**Population:** '+"{:.2f}".format(pop/1000000)+' Mio'
        reg = '**Region:** '+reg
        inc = '**Income Group:** '+inc                     
        
        p_selection_Yhat = country+'Yhat:' #Column name in df_plot
        
        y_max = int(df_plot['Year'].dt.strftime('%Y').max())
        predl = df_plot[-3:][p_selection_Yhat].round(2).values
    
        
        #Create list with the Year and the Predictions for easier print 
        pred_next_years = [str(y_max-2 + i) +' : '+ str(predl[i]) for i in range(0, len(predl))] #The three last years in the Dataframe are the predicted ones, in ascending order
        #print country Information
        pred_grid[count][0].markdown(prof)
        pred_grid[count][0].markdown(year)
        pred_grid[count][0].markdown(pop)
        pred_grid[count][0].markdown(reg)
        pred_grid[count][0].markdown(inc)
        pred_grid[count][0].markdown("##### Prediction for the next Years:")
        pred_grid[count][0].markdown(pred_next_years[0])
        pred_grid[count][0].markdown(pred_next_years[1])
        pred_grid[count][0].markdown(pred_next_years[2])
        
       #prepare a temp dataframe and plot a grpah that contains a timeline for the 'SDG_Index_Rank' and teh Selected SDG
    
        df_rank_plot = df_hel[df_hel['Country'] == country][['Year','SDG_Index_Rank', goal+'_rank']].melt('Year', var_name = 'Rankings', value_name = 'Rank')
        cl = alt.Chart(df_rank_plot).mark_line().encode(
            x='Year',
            #y='Rank:Q',
            y= alt.Y('Rank:Q', scale=alt.Scale(reverse=True),title = 'Rank'),
            color = alt.Color('Rankings:N',
                    legend = alt.Legend(orient='none', legendX=5, legendY=5))
        ) 
        
        
        pred_grid[count][1].markdown('**Worldwide rank** within 177 countries (overall and for the selected SDG), inversed Y axis because rank #1 is the best')
        pred_grid[count][1].altair_chart(cl, use_container_width=True)
        count = count +1
    
    #Plot Population this should give just some additional information for the user.
    #Create a dataset to plot Country Population
    df_population = pd.DataFrame()
    df_population['Year'] = pd.date_range(start = pd.to_datetime(str(min(df_hel['Year']))), end = pd.to_datetime(str(max(df_hel['Year']))), freq='YS').year
    for country in p_selection:
        df_population = pd.merge(df_population, df_hel[df_hel['Country']==country][['Year','Population']], how="right", on=["Year"]).rename(columns = {'Population':country})

    st.write('To give you a bit of a context and show how many people are affected, see the population development for the selected countries')
    st.line_chart(df_population, x='Year' )

   
    

    
    
    
    
    