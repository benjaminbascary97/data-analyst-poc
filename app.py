import io
import streamlit as st
import pandas as pd
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain_community.callbacks import StreamlitCallbackHandler

llm = ChatOpenAI(
    api_key=st.secrets["openai_api_key"].key, temperature=0, streaming=True
)

st.title("AI Assistant for Data Analysis")
st.header("Exploratory Data Analysis part")

if "clicked" not in st.session_state:
    st.session_state.clicked = {1: False}

# function to display the dataset


def clicked(button):
    st.session_state.clicked[button] = True


st_callback = StreamlitCallbackHandler(
    st.container(border=True), expand_new_thoughts=True
)


st.button("Let's get started!", on_click=clicked, args=[1])
if st.session_state.clicked[1]:
    st.write("Upload your dataset")
    file = st.file_uploader("Upload your dataset", type=["csv", "xlsx"])
    if file is not None:
        st.write(f"✅ File *{file.name}* uploaded successfully!")
        loaded_dataframe = pd.read_csv(file, low_memory=False)

        pandas_agent = create_pandas_dataframe_agent(
            df=loaded_dataframe,
            llm=llm,
            agent_type=AgentType.OPENAI_FUNCTIONS,
        )

        @st.cache_data
        def run_agent():
            st.subheader("**AI Data Overview**")

            answer = pandas_agent.run(
                "Describe me the columns please!"
            )

            st.write(answer)

            st.subheader("📄 Brief description:")
            st.write(loaded_dataframe.describe())

        run_agent()


        non_date_named_columns = [
            col for col in loaded_dataframe.columns if "Date" not in col
        ]

        options = [None] + non_date_named_columns

        variable = st.selectbox(
            "🤖 Choose a variable so I can analyze it!",
            options=options,
            placeholder="Select a variable",
            index=0,
        )

        def show_chart_and_analysis():
            answer = pandas_agent.run(
                f"Describe me following variable and correlations with other variables: {variable}"
            )

            st.subheader(f"📊 {variable} variable summary")

            st.markdown(answer)

            st.subheader(f"📈 {variable} variable distribution")
            st.line_chart(loaded_dataframe, y=[variable])

            return answer

        if variable is not None:
            show_chart_and_analysis()

        if variable is not None:
            if prompt := st.chat_input():
                st.chat_message("user").write(prompt)
                with st.chat_message("assistant"):
                    st_callback = StreamlitCallbackHandler(st.container())
                    response = pandas_agent.invoke(
                        {"input": prompt}, {"callbacks": [st_callback]}
                    )
                    st.write(response["output"])
