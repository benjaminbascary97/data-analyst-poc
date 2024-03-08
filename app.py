# Import required libraries
import os
import time
import streamlit as st
import pandas as pd

from langchain_openai import OpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_community.callbacks import StreamlitCallbackHandler


def response_generator(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


# OpenAIKey
os.environ["OPENAI_API_KEY"] = api_key = st.secrets["openai_api_key"].key

# Title
st.title("AI Assistant for Data Science 🤖")

# Welcoming message
st.write(
    "Hello, 👋 I am your AI Assistant and I am here to help you analyze data and generate new insights."
)


# Initialise the key in session state
if "clicked" not in st.session_state:
    st.session_state.clicked = {1: False}


# Function to udpate the value in session state
def clicked(button):
    st.session_state.clicked[button] = True


st.button("Let's get started", on_click=clicked, args=[1])
if st.session_state.clicked[1]:
    user_csv = st.file_uploader("Upload your file here", type="csv")
    if user_csv is not None:
        user_csv.seek(0)
        df = pd.read_csv(user_csv, low_memory=False)

        # llm model
        llm = OpenAI(temperature=0, model="gpt-4-0125-preview", max_tokens=4096)

        # Function sidebar
        @st.cache_data
        def steps_eda():
            steps_eda = llm("What are the steps of EDA")
            return steps_eda

        # Pandas agent
        pandas_agent = create_pandas_dataframe_agent(llm, df)

        # Functions main
        @st.cache_data
        def function_agent():
            st.write_stream(response_generator("**Data Overview**"))
            st.write_stream(
                response_generator("The first rows of your dataset look like this:")
            )
            st.write(df.head())
            st.subheader("📓 **Data Overview**")
            columns_df = pandas_agent.run("What are the meaning of the columns?")
            st.write_stream(response_generator(columns_df))

            duplicates = pandas_agent.run(
                "Are there any duplicate values and if so where?"
            )
            st.write_stream(response_generator(duplicates))
            correlation_analysis = pandas_agent.run(
                "Calculate correlations between numerical variables to identify potential relationships."
            )
            st.write_stream(response_generator(correlation_analysis))

            st.write_stream(response_generator("✅ AI Data Overview completed!"))
            st.write("**Data Summarisation**")
            st.write(df.describe())
            return

        @st.cache_data
        def generating_variable_analysis(selected_variable):
            st.line_chart(df, y=[selected_variable])
            summary_statistics = pandas_agent.run(
                f"Give me a summary of the statistics of {selected_variable}. Make it human readable and in markdown please."
            )
            st.write_stream(response_generator(summary_statistics))
            normality = pandas_agent.run(
                f"Check for normality or specific distribution shapes of {selected_variable}"
            )
            st.write_stream(response_generator(normality))
            outliers = pandas_agent.run(
                f"Indicate the outliers of {selected_variable}"
            )
            st.write_stream(response_generator(outliers))
            trends = pandas_agent.run(
                f"Analyse trends, and cyclic patterns of {selected_variable}"
            )
            st.write_stream(response_generator(trends))
            return

        # Main

        st.header("Exploratory data analysis")
        st.subheader("General information about the dataset")

        function_agent()

        st.subheader("Variable of study")

        non_date_named_columns = [col for col in df.columns if "Date" not in col]

        options = [None] + non_date_named_columns

        selected_variable = st.selectbox(
            "🤖 Choose a variable so I can analyze it!",
            options=options,
            placeholder="Select a variable",
            index=0,
        )

        if selected_variable is not None and selected_variable != "":
            generating_variable_analysis(selected_variable=selected_variable)

        if selected_variable is not None:
            st.subheader("Would you like to ask the AI a question about your data?")
            if prompt := st.chat_input(max_chars=150):

                st.chat_message("user").write(prompt)
                with st.chat_message("assistant"):
                    st_callback = StreamlitCallbackHandler(st.container())
                    response = pandas_agent.invoke(
                        {"input": prompt}, {"callbacks": [st_callback]}
                    )
                    st.write(response["output"])
