import streamlit as st
from langchain_openai.chat_models import ChatOpenAI
import os, pymssql
from dotenv import load_dotenv

load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri('mssql+pymssql://SA:Password1234@localhost:1433/master', schema="dbo")

st.title("🦜🔗 Northwind RAG App")

def generate_response(input_text):

    llm = ChatOpenAI(model="gpt-4o-mini")

    examples = [{"input": "Find the top selling product per country", "query": """with product_sales as (
                    select 
                        t1.ShipCountry
                    ,   t3.ProductName
                    ,   sum(t2.Quantity) as total_sales
                    ,   rank() over (partition by t1.ShipCountry  order by sum(t2.Quantity) desc) as position
                    from 
                        [dbo].[Orders] T1 
                    inner join 
                        [dbo].[Order Details] T2 
                    on 
                        t1.OrderID = t2.OrderID
                    inner join 
                        [dbo].Products T3 
                    on 
                        t2.ProductID = t3.ProductID
                    group by 
                        t1.ShipCountry, t3.ProductName
                    ) 

                    select *
                    from product_sales
                    where position = 1;"""},
                    {"input": "what was the first order shipped?",
                     "query":"SELECT TOP 1 OrderID, ShippedDate FROM [dbo].[Orders] where shippeddate is not null ORDER BY ShippedDate ASC;"}
    ]

    from langchain_community.vectorstores import FAISS
    from langchain_core.example_selectors import SemanticSimilarityExampleSelector
    from langchain_openai import OpenAIEmbeddings

    example_selector = SemanticSimilarityExampleSelector.from_examples(
        examples,
        OpenAIEmbeddings(),
        FAISS,
        k=5,
        input_keys=["input"],
    )

    from langchain_core.prompts import (
        ChatPromptTemplate,
        FewShotPromptTemplate,
        MessagesPlaceholder,
        PromptTemplate,
        SystemMessagePromptTemplate,
    )
    from langchain_community.agent_toolkits import create_sql_agent

    system_prefix = """You are an agent designed to interact with a SQL database.
    Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
    Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
    You can order the results by a relevant column to return the most interesting examples in the database.
    Never query for all the columns from a specific table, only ask for the relevant columns given the question.
    You have access to tools for interacting with the database.
    Only use the given tools. Only use the information returned by the tools to construct your final answer.
    You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

    DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

    If the question does not seem related to the database, just return "I don't know" as the answer.
    
    After the answer return the SQL you used to generate the query

    Here are some examples of user inputs and their corresponding SQL queries:"""

    few_shot_prompt = FewShotPromptTemplate(
        example_selector=example_selector,
        example_prompt=PromptTemplate.from_template(
            "User input: {input}\nSQL query: {query}"
        ),
        input_variables=["input", "dialect", "top_k"],
        prefix=system_prefix,
        suffix="",
    )

    full_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate(prompt=few_shot_prompt),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    prompt_val = full_prompt.invoke(
        {
            "input": "What is the top selling product per country?",
            "top_k": 5,
            "dialect": "MS SQL",
            "agent_scratchpad": [],
        }
    )

    agent = create_sql_agent(
        llm=llm,
        db=db,
        prompt=full_prompt,
        verbose=True,
        agent_type="openai-tools",
    )
    response = agent.invoke({"input": input_text})['output']
    st.session_state.messages.append({"role": "assistant", "content": response})

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if prompt := st.chat_input("Ask a question about the database?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    generate_response(prompt)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
