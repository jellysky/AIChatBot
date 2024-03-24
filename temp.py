import langchain
import langchain_openai
import langchain_core
import langchain_community
import openai
import chainlit as cl
import tiktoken
import chromadb
import pypdf
import tiktoken
import unstructured
import llama_index

import os
import io
from io import BytesIO
import PyPDF2

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.vectorstores import Chroma
from langchain.vectorstores.vectara import VectaraRetriever
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory, ConversationBufferMemory
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
os.environ['OPENAI_API_KEY'] = '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'
path_for_directory_of_pdf = 'MangrovePDFs'

pdf_loader = DirectoryLoader(path_for_directory_of_pdf, glob="**/*.pdf")
loaders =[pdf_loader]
documents = []

for loader in loaders:
    documents.extend(loader.load())

from langchain.text_splitter import CharacterTextSplitter

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(documents)

if not os.path.exists(f'{path_for_directory_of_pdf}db'):
    os.mkdir(f'{path_for_directory_of_pdf}db')

embeddings =OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents, embeddings, persist_directory=f"{path_for_directory_of_pdf}db")

vectorstore.persist()
vectorstore = None
vectorstore = Chroma(persist_directory=f'{path_for_directory_of_pdf}db', embedding_function=embeddings)

from langchain_openai import ChatOpenAI
from langchain_core.output_parsers.pydantic import PydanticOutputParser

llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k")

memory = ConversationBufferMemory(llm=llm, memory_key="chat_history")
qa = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), memory=memory, get_chat_history=lambda h: h)

question = "What are mangroves?"
result = qa({"question": question})
print(result)

question = "What are mangroves?"
result = qa({"question": question})
print("question", question)
print("answer",result['answer'])

