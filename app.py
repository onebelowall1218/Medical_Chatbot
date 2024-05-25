from langchain.chains import RetrievalQA
from langchain.output_parsers import OutputFixingParser
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
import os
import chainlit as cl

from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms.ollama import Ollama

from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

import PyPDF2

from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.chains import (
    ConversationalRetrievalChain,
)

from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import pandas as pd

import asyncio

# Configure LLM
llm = Ollama(
    base_url="http://localhost:11434",
    model="gemma:2b",
    callback_manager=CallbackManager(
        [StreamingStdOutCallbackHandler()],
    ),
    temperature=0.0,
)

embedding_function = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2")  # Different embedding function could be used


async def llm_response(message, display=True):
    if display:
        msg = cl.Message(content="")
        await msg.send()
        response = llm.astream(message)
        async for content in response:
            if content:
                await msg.stream_token(content)
        await msg.update()
    final_response = msg.content
    return final_response


async def ask_patient_info(user_response):
    # Define response schemas
    patient_name_schema = ResponseSchema(name="patient name",
                                         description="This is the name of the patient",
                                         )
    patient_age_schema = ResponseSchema(name="age",
                                        description="This is the age of the patient",
                                        )
    patient_gender_schema = ResponseSchema(name="gender",
                                           description="This is the gender (Male or Female) of the patient",
                                           )
    patient_height_schema = ResponseSchema(name="height",
                                           description="This is the height of the patient",
                                           )
    patient_weight_schema = ResponseSchema(name="weight",
                                           description="This is the weight of the patient",
                                           )
    patient_symptoms_schema = ResponseSchema(name="symptoms",
                                             description="Descriptions of medical condition that the patient is facing.",
                                             )
    patient_allergies_schema = ResponseSchema(name="allergies",
                                              description="This is the allergies that the patient have.",
                                              )
    response_schemas = [patient_name_schema,
                        patient_age_schema,
                        patient_gender_schema,
                        patient_height_schema,
                        patient_weight_schema,
                        patient_symptoms_schema,
                        patient_allergies_schema,
                        ]

    # Create structured output parser
    output_parser = StructuredOutputParser.from_response_schemas(
        response_schemas)

    # Get format instructions
    format_instructions = output_parser.get_format_instructions()

    # Define the template string
    template_string = """You are an automatic form-filling system designed to collect patient information.
    Fill out the following details based on the user's response below. Write "None" if a particular information is missing.:

    user response: ```{user_response}```

    {format_instructions}
    """

    # Create chat prompt template
    prompt = ChatPromptTemplate.from_template(template=template_string)

    # Formate messages
    # # Example 1:
    # user_response = """Hi, I am Anjum Ul Azim. I am 24 years old. I am a male of course. I have back pain and fever. Also I have headache and mild fever. I don't have
    # any kind of allergies. My height is 169 cm. I weigh 54.5 kg.
    # """
    # # Example 2:
    # user_response = """Hi, I am Naman Sethiya. I am 20 years old. I am a male of course. I have knee pain. I am allergic to peanuts.
    # My height is 171 cm. I weigh 65.5 kg.
    # """
    # # Example 3:
    # user_response = """Hi, I am Adarsh Sharma. I am 20 years old. I am a male of course. I am having low-grade fevers, malaise, weight loss, myalgias, and arthralgias. I am allergic to meat.
    # My weight is 50 kg. I am 169 cm tall and I have no allergies.
    # """

    # User input
    # user_response = input("User : ")
    user_response = user_response

    messages = prompt.format_messages(user_response=user_response,
                                      format_instructions=format_instructions,
                                      )

    # Invoke the language model
    response = llm.invoke(messages)
    # Optional, for checking out what is going on.
    await cl.Message(content=response).send()
    # Extract the structured output
    try:
        structured_output = output_parser.parse(response)
    except:
        new_parser = OutputFixingParser.from_llm(parser=output_parser,
                                                 llm=llm,
                                                 )
        structured_output = new_parser.parse(response)
    return structured_output


# Find diseases
async def find_diseases(patient_symptoms, vectordb):
    question_template = """List down most possible diseases or disorders based on the patients symptoms given below :

    Patient Symptoms : {patient_symptoms}
    """

    question = question_template.format(patient_symptoms=patient_symptoms)

    # # This method consumes too much resources
    # qa_chain_mmr = RetrievalQA.from_chain_type(
    #     llm,
    #     retriever = vectordb.as_retriever(),
    #     chain_type="map_reduce"
    # )

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(),
    )
    result = qa_chain.invoke({"query": question})

    # Format Disease List
    disease_list = result["result"]
    # Define response schemas
    possible_diseases_schema = ResponseSchema(name="possible diseases",
                                              description="Comma separated top ten most probable list of diseases",
                                              )
    response_schemas = [possible_diseases_schema]

    # Create structured output parser
    output_parser = StructuredOutputParser.from_response_schemas(
        response_schemas)

    # Get format instructions
    format_instructions = output_parser.get_format_instructions()

    # Define the template string
    template_string = """Convert the list of diseases given below to the format as needed.

    Disease List :
    {disease_list}

    {format_instructions}
    """

    # Create chat prompt template
    prompt = ChatPromptTemplate.from_template(template=template_string)

    # # Formate messages
    # # Example 1:
    # disease_list = """Sure, here are the possible diseases or disorders based on the patient's symptoms:

    # 1. Leukocytoblastic vasculitis
    # 2. Renal disease
    # 3. Polyarteritis nodosa
    # 4. Systemic lupus erythematosus
    # 5. Rheumatoid arthritis
    # 6. Scleroderma
    # 7. Polymyositis
    # 8. Limited Wegener's disease"""

    messages = prompt.format_messages(disease_list=disease_list,
                                      format_instructions=format_instructions,
                                      )

    # Invoke the language model
    response = llm.invoke(messages)
    await cl.Message(content=response).send()

    # Extract the structured output
    try:
        structured_output = output_parser.parse(response)

    except:
        new_parser = OutputFixingParser.from_llm(parser=output_parser,
                                                 llm=llm,
                                                 )
        structured_output = new_parser.parse(response)

    disease_list = structured_output["possible diseases"].split(",")
    return disease_list


async def get_treatment(disease):
    template_string = f"What is the treatment for {disease}?"
    result = await asyncio.to_thread(llm.invoke, template_string)
    await cl.Message(content=result).send()
    return result


async def get_more_treatments(disease_list, size=1):
    tasks = [get_treatment(disease) for disease in disease_list[:size]]
    results = await asyncio.gather(*tasks)
    return results


async def format_info(structured_output, disease_list, treatments):
    df1 = pd.DataFrame([structured_output])
    temp_dict = {}
    for i in range(3):
        temp_str = "***Disease : "+disease_list[i]+"\n\n"+treatments[i]
        temp_dict[f"disease{i+1}"] = temp_str
    df2 = pd.DataFrame([temp_dict])
    df = pd.concat([df1, df2], axis=1)
    return df


async def add_info(structured_output, disease_list, treatments, user_database, database_path):
    df = await format_info(structured_output, disease_list, treatments)
    updated_user_database = pd.concat(
        [df, user_database], axis=0, ignore_index=True)
    updated_user_database.to_csv(database_path, index=False)
    return updated_user_database


@cl.on_chat_start
async def start():
    # Load Vector Store
    persist_directory = "database/"
    vectordb = Chroma(
        embedding_function=embedding_function,
        persist_directory=persist_directory,
    )
    retriever = vectordb.as_retriever(
        search_type="similarity", search_kwargs={"k": 5})

    # ask patient's information:
    await cl.Message(content="Tell me the medical problem you are facing...").send()
    cl.user_session.set("vectordb", vectordb)


@cl.on_message
async def on_message(message: cl.Message):
    # Get the user database:
    database_path = "example_user_database.csv"
    user_database = pd.read_csv(database_path)
    vectordb = cl.user_session.get("vectordb")
    # Ask patient's information:
    structured_output = await ask_patient_info(message.content)
    print("\n\nStructured data of the user extracted...\n\n")
    # Find possible diseases:
    patient_symptoms = structured_output["symptoms"]
    patient_allergies = structured_output["allergies"]
    disease_list = await find_diseases(patient_symptoms, vectordb)
    print("\n\nFound possible diseases.\n\n")
    # Get treatments
    treatments = await get_more_treatments(disease_list, size=3)
    print("\n\nFound treatments for most possible diseases\n\n")
    print("\n\nAll Done.\n\n")
    user_database = await add_info(structured_output, disease_list, treatments, user_database, database_path)
    await cl.Message(content='Your information are saved successfully.').send()
