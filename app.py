import os
import time
import streamlit as st
import pandas as pd

from langchain_openai import OpenAI, ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.agents import AgentType


def response_generator(response):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)


# OpenAIKey
os.environ["OPENAI_API_KEY"] = api_key = st.secrets["openai_api_key"].key

# Define wide and config
st.set_page_config(
    layout="wide",
    page_title="AI Assistant for Data Analysis and Maintenance 🛠️",
    page_icon="🛠️",
)

# Title
st.title("AI Assistant for Data Analysis and Maintenance 🛠️")

# Welcoming message
st.write(
    "Hello, I am your AI Assistant and I am here to help you analyze data and generate new insights."
)


# Initialise the key in session state
if "clicked" not in st.session_state:
    st.session_state.clicked = {1: False}


# Function to udpate the value in session state
def clicked(button):
    st.session_state.clicked[button] = True


if "file_uploader_key" not in st.session_state:
    st.session_state["file_uploader_key"] = 0

if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []

st.button("Let's get started", on_click=clicked, args=[1])
if st.session_state.clicked[1]:
    user_csv = st.file_uploader(
        "Upload your file here",
        type="csv",
        accept_multiple_files=True,
        key=st.session_state["file_uploader_key"],
    )

    dataframes = []
    if user_csv is not None and len(user_csv) > 0:
        st.session_state["uploaded_files"] = user_csv

        if st.button("Clear uploaded files"):
            st.session_state["file_uploader_key"] += 1
            st.experimental_rerun()

        for csv in user_csv:
            csv.seek(0)
            df = pd.read_csv(csv, low_memory=False)
            dataframes.append(df)

    dataframe_names = ""
    for index, df in enumerate(dataframes):
        dataframe_names += f"df{index + 1}\n"

    print(dataframes)
    PROMPT_PREFIX = f"""
    You are an expert pandas agent that works for an oil & gas company. You main job is
    to use your PythonAstREPLTool to analyze the data and execute code to find an answer
    to user question. Think analytically and keep in mind you have all the information
    provided in the dataframes. If there are more than one dataframe, it is very probable
    that the user wants to join them and link variables of both dataframes.

    The dataframes names are:

    {dataframe_names}

    One last detail: make your responses the more human readable possible. If someone asks
    for time, give days, hours, minutes, seconds. If someone asks for a number, give the
    number in the most human readable format.

    Begin!
    """

    print(PROMPT_PREFIX)

    if len(dataframes) > 0:
        llm = OpenAI(temperature=0, max_tokens=-1, streaming=True)
        advanced_llm = ChatOpenAI(temperature=0, model="gpt-4")

        # Function sidebar
        @st.cache_data(show_spinner=True)
        def steps_eda(user_csv):
            steps_eda = llm.stream(
                input="What are the steps of EDA. Use a few words please and only 5 steps."
            )
            return st.write_stream(steps_eda)

        # Pandas agent
        pandas_agent = create_pandas_dataframe_agent(
            llm,
            dataframes,
            max_execution_time=250,
            max_iterations=10,
            number_of_head_rows=5,
            # agent_type=AgentType.OPENAI_FUNCTIONS,
            prefix=PROMPT_PREFIX,
        )
        gpt4_pandas_agent = create_pandas_dataframe_agent(
            advanced_llm,
            dataframes,
            number_of_head_rows=5,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            max_execution_time=250,
            max_iterations=10,
            prefix=PROMPT_PREFIX,
        )
        pandas_agent.handle_parsing_errors = True
        gpt4_pandas_agent.handle_parsing_errors = True

        st.write("📊 **Steps of Exploratory Data Analysis**")
        steps_eda(user_csv=user_csv)

        # Functions main
        @st.cache_data
        def function_agent(user_csv):

            st.write_stream(response_generator("**Data Overview**"))

            if len(dataframes) > 1:
                st.write_stream(
                    response_generator(
                        "You have uploaded multiple dataframes. I will provide an overview of each of them. The columns of each dataframe are as follows:"
                    )
                )
            else:
                st.write_stream(
                    response_generator(
                        "You have uploaded a single dataframe. I will provide an overview of it. The columns of the dataframe are as follows:"
                    )
                )
            for df in dataframes:
                st.write(df.head())
            st.subheader("📓 **Data Overview**")
            columns_df = pandas_agent.run(
                """Answer a super simple EDA analysis of the datasets"""
            )
            st.write_stream(response_generator(columns_df))
            return

        # Main

        st.header("Exploratory data analysis")
        st.subheader("General information about the dataset")

        function_agent(user_csv=user_csv)

        # if selected_variable is not None:
        st.subheader("Would you like to ask the AI a question about your data?")
        if prompt := st.chat_input():

            st.chat_message("user").write(prompt)
            with st.chat_message("assistant"):
                st_callback = StreamlitCallbackHandler(
                    st.container(),
                    collapse_completed_thoughts=True,
                    expand_new_thoughts=True,
                )
                response = gpt4_pandas_agent.invoke(
                    {"input": prompt}, {"callbacks": [st_callback]}
                )
                st.write(response["output"])
