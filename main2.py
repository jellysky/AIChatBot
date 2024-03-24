import langchain
import langchain_openai
import langchain_community
import langchain_core
import chromadb
import tiktoken
import PyPDF2

OPENAI_API_KEY='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'

from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.output_parsers.pydantic import PydanticOutputParser
import os
import io
import chainlit as cl
import PyPDF2
from io import BytesIO
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.memory import ConversationSummaryBufferMemory, ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

#from dotenv import load_dotenv

# Load environment variables from .env file (Optional)
#load_dotenv()

#OPENAI_API_KEY = os.getenv('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')
os.environ['OPENAI_API_KEY'] = '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'
system_template = """Use the following pieces of context to answer the users question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should be a reference to the source of the document from which you got your answer.

Example of your response should be:

```
The answer is foo
SOURCES: xyz
```

Begin!
----------------
{summaries}"""


messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

@cl.on_chat_start
async def on_chat_start():

    # Sending an image with the local file path
    elements = [cl.Image(name="image1", display="inline", path=".im/robot.jpg")]
    await cl.Message(content="Reading in data to AskAnyQuery!", elements=elements).send()

    path_for_directory_of_pdf = 'MangrovePDFs'

    pdf_loader = DirectoryLoader(path_for_directory_of_pdf, glob="**/*.pdf")
    loaders = [pdf_loader]
    documents = []

    for loader in loaders:
        documents.extend(loader.load())

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(
        documents, embeddings, persist_directory=f"{path_for_directory_of_pdf}db")

    vectorstore.persist()
    vectorstore = None
    vectorstore = Chroma(persist_directory=f'{path_for_directory_of_pdf}db', embedding_function=embeddings)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k")
    memory = ConversationBufferMemory(llm=llm, memory_key="chat_history")
    qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), memory=memory,
                                               get_chat_history=lambda h: h)


    # Read the PDF file
#    pdf_stream = BytesIO(documents)
#    pdf = PyPDF2.PdfReader(pdf_stream)
#    pdf_text = ""
#    for page in pdf.pages:
#        pdf_text += page.extract_text()

    # Let the user know that the system is ready
    msg = cl.Message(content=f"Processed ...")
    await msg.send()

    msg.content = f"You can now ask questions!"
    await msg.update()

    cl.user_session.set("chain", qa)


@cl.on_message
async def main(message: str):
    chain = cl.user_session.get("chain")  # type: RetrievalQAWithSourcesChain
    cb = cl.AsyncLangchainCallbackHandler(
        stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
    )

    answer = chain({"question": message})

    if cb.has_streamed_final_answer:
        await cb.final_stream.update()
    else:
        await cl.Message(content=answer).send()