# Import required libraries
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

# Title
st.title("🤖 AI Assistant for Data Analysis and Maintenance 🛠️")

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
        llm = OpenAI(temperature=0, max_tokens=-1)
        advanced_llm = ChatOpenAI(temperature=0, model="gpt-4")

        # Function sidebar
        @st.cache_data
        def steps_eda():
            steps_eda = llm("What are the steps of EDA")
            return steps_eda

        # Pandas agent
        pandas_agent = create_pandas_dataframe_agent(llm, df)
        gpt4_pandas_agent = create_pandas_dataframe_agent(advanced_llm, df, agent_type=AgentType.OPENAI_FUNCTIONS)

        # Functions main
        @st.cache_data
        def function_agent():
            st.write_stream(response_generator("**Data Overview**"))
            st.write_stream(
                response_generator("The first rows of your dataset look like this:")
            )
            st.write(df.head())
            st.subheader("📓 **Data Overview**")
            columns_df = pandas_agent.run("""Answer the following questions as a list of responses separated by bullet points:
                                          What are the meaning of the columns?
                                          Are there any duplicate values and if so where?
                                          Calculate correlations between numerical variables to identify potential relationships.
                                          """)
            st.write_stream(response_generator(columns_df))
            return

        # Main

        st.header("Exploratory data analysis")
        st.subheader("General information about the dataset")

        function_agent()

        # if selected_variable is not None:
        st.subheader("Would you like to ask the AI a question about your data?")
        if prompt := st.chat_input():

            st.chat_message("user").write(prompt)
            with st.chat_message("assistant"):
                st_callback = StreamlitCallbackHandler(st.container(), collapse_completed_thoughts=False)
                response = gpt4_pandas_agent.invoke(
                    {"input": prompt}, {"callbacks": [st_callback]}
                )
                st.write(response["output"])
