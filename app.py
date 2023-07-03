import os
from apikey import apikey
import streamlit as st
import pinecone
##import nest_asyncio
##nest_asyncio.apply()
from langchain.document_loaders.sitemap import SitemapLoader
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA

os.environ['OPENAI_API_KEY'] = apikey
st.title('KYC gpt')

# initialize pinecone
pinecone.init(
    api_key="my-pinecone-token",  # find at app.pinecone.io
    environment="my-pinecone-env"  # next to aclepi key in console
)

loader = SitemapLoader(
    "https://www.anz.com.au/sitemap.xml",
    filter_urls=["https://www.anz.com.au/support/know-your-customer/"]
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1200,
    chunk_overlap  = 200,
    length_function = len,
)

docs_chunks = text_splitter.split_documents(docs)
embeddings = OpenAIEmbeddings()

index_name = "anz"

##create a new index
##docsearch = Pinecone.from_documents(docs_chunks, embeddings, index_name=index_name)

# if you already have an index, you can load it like this
docsearch = Pinecone.from_existing_index(index_name, embeddings)
query = "What all details i need to submit for KYC"
docs = docsearch.similarity_search(query)
##print(docs[0])


##prompt = st.text_input('ask me here')


llm = OpenAI(temperature=0.9)


##if prompt:
##    response = llm(prompt)
##    st.write(response)

qa_with_sources = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True)

query = st.text_input('ask me here')
if query:
    result = qa_with_sources({"query": query})
    st.write(result["result"])


# Testing OpenAI functions

##def get_kyc_info(registration_number: str):
##    kyc_info = {
##        "regnum": registration_number,
##        "country": "AU",
##    }
##    return json.dumps(kyc_info)

##functions = [
##    {
##        "name": "get_kyc_info",
##        "description": "Get regnum and country of an entity for KYC",
##        "parameters": {
##            "type": "object",
##            "properties": {
##                "registration_number": {
##                    "type": "string",
##                    "description": "The registration number of an entity, e.g. 1234567",
##                },
##            },
##            "required": ["registration_number"],
##        },
##    }
##]