#Import required libraries
import os 

import streamlit as st
import pandas as pd

from langchain_openai import OpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

#OpenAIKey
os.environ['OPENAI_API_KEY'] = api_key=st.secrets["openai_api_key"].key

#Title
st.title('AI Assistant for Data Science 🤖')

#Welcoming message
st.write("Hello, 👋 I am your AI Assistant and I am here to help you with your data science projects.")


#Initialise the key in session state
if 'clicked' not in st.session_state:
    st.session_state.clicked ={1:False}

#Function to udpate the value in session state
def clicked(button):
    st.session_state.clicked[button]= True
st.button("Let's get started", on_click = clicked, args=[1])
if st.session_state.clicked[1]:
    user_csv = st.file_uploader("Upload your file here", type="csv")
    if user_csv is not None:
        user_csv.seek(0)
        df = pd.read_csv(user_csv, low_memory=False)

        #llm model
        llm = OpenAI(temperature = 0)

        #Function sidebar
        @st.cache_data
        def steps_eda():
            steps_eda = llm('What are the steps of EDA')
            return steps_eda

        #Pandas agent
        pandas_agent = create_pandas_dataframe_agent(llm, df, verbose = True)

        #Functions main
        @st.cache_data
        def function_agent():
            st.write("**Data Overview**")
            st.write("The first rows of your dataset look like this:")
            st.write(df.head())
            st.write("**Data Cleaning**")
            columns_df = pandas_agent.run("What are the meaning of the columns?")
            st.write(columns_df)
            missing_values = pandas_agent.run("How many missing values does this dataframe have? Start the answer with 'There are'")
            st.write(missing_values)
            duplicates = pandas_agent.run("Are there any duplicate values and if so where?")
            st.write(duplicates)
            st.write("**Data Summarisation**")
            st.write(df.describe())
            correlation_analysis = pandas_agent.run("Calculate correlations between numerical variables to identify potential relationships.")
            st.write(correlation_analysis)
            outliers = pandas_agent.run("Identify outliers in the data that may be erroneous or that may have a significant impact on the analysis.")
            st.write(outliers)
            new_features = pandas_agent.run("What new features would be interesting to create?.")
            st.write(new_features)
            return

        @st.cache_data
        def function_question_variable():
            st.line_chart(df, y =[selected_variable])
            summary_statistics = pandas_agent.run(f"Give me a summary of the statistics of {selected_variable}")
            st.write(summary_statistics)
            normality = pandas_agent.run(f"Check for normality or specific distribution shapes of {selected_variable}")
            st.write(normality)
            outliers = pandas_agent.run(f"Assess the presence of outliers of {selected_variable}")
            st.write(outliers)
            trends = pandas_agent.run(f"Analyse trends, seasonality, and cyclic patterns of {selected_variable}")
            st.write(trends)
            missing_values = pandas_agent.run(f"Determine the extent of missing values of {selected_variable}")
            st.write(missing_values)
            return
        
        @st.cache_data
        def function_question_dataframe():
            dataframe_info = pandas_agent.run(user_question_dataframe)
            st.write(dataframe_info)
            return

        #Main

        st.header('Exploratory data analysis')
        st.subheader('General information about the dataset')

        function_agent()

        st.subheader('Variable of study')

        non_date_named_columns = [
            col for col in df.columns if "Date" not in col
        ]

        options = [None] + non_date_named_columns

        selected_variable = st.selectbox(
            "🤖 Choose a variable so I can analyze it!",
            options=options,
            placeholder="Select a variable",
            index=0,
        )

        if selected_variable is not None and selected_variable !="":
            function_question_variable()

            st.subheader('Further study')

        if selected_variable:
            user_question_dataframe = st.text_input( "Is there anything else you would like to know about your dataframe?")
            if user_question_dataframe is not None and user_question_dataframe not in ("","no","No"):
                function_question_dataframe()
            if user_question_dataframe in ("no", "No"):
                st.write("")

