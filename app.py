# Importing necessary libraries
import os
import streamlit as st 
from dotenv import load_dotenv
import pickle
from PyPDF2 import PdfReader
from streamlit_extras.add_vertical_space import add_vertical_space
# Custom libraries
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain

# Function to check if a valid PDF is uploaded
def checkPDF(pdf):
    if pdf is None:
        return False
    return True

# Function to extract text from a PDF file
def getPDFtext(pdf):
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    return text

# Function to split a long text into smaller chunks
def textTochunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,        # Size of each chunk
        chunk_overlap=200,      # Overlap between consecutive chunks
        length_function=len     # Function to calculate the length of the text
    )
    chunks = text_splitter.split_text(text=text)
    return chunks

# Function to check if embeddings for a PDF already exist
def checkEmbedding(store_name):
    if os.path.exists(f"./embeddings/{store_name}.pkl"):
        return True
    return False

# Function to compute or load embeddings for the text chunks
def createEmbeddings(chunks, store_name):
    if checkEmbedding(store_name):
        with open(f"./embeddings/{store_name}.pkl", "rb") as f:
            VectorStore = pickle.load(f)
        st.write('Embeddings Loaded from the Disk.')
    else:
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        with open(f"./embeddings/{store_name}.pkl", "wb") as f:
            pickle.dump(VectorStore, f)
        st.write('Embeddings Computation Completed.')
    return VectorStore

# Function to get a response to a user's question using LLM model
def getResponse(query, vectorstore):
    repo_id = "tiiuae/falcon-7b-instruct"
    docs = vectorstore.similarity_search(query=query, k=3)
    llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0.6, "max_new_tokens":500})
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    response = chain.run(input_documents=docs, question=query)
    return response

# Main function that runs the Streamlit app
def main():
    # Set up the sidebar with some information about the app
    with st.sidebar:
        st.title("LLM Chat App")
        st.markdown('''
        # About me
        This app is an LLM-powered chatbot built using:
        - Streamlit
        - Langchain
        - Huggingface Embeddings
        ''')
        add_vertical_space(2)
        st.write("Made by Siddhartha Shakya")

    # Main content of the app
    st.header("Chat With PDF")
    pdf = st.file_uploader("Upload your PDF", type='pdf')  # Allow user to upload a PDF file

    # If a valid PDF file is uploaded
    if checkPDF(pdf):
        store_name = pdf.name[:-4]  # Extract the store name from the PDF file name (excluding the .pdf extension)
        pdf_reader = PdfReader(pdf)
        text = getPDFtext(pdf_reader)  # Extract text from the PDF
        chunks = textTochunks(text)    # Split text into smaller chunks
        vectorstore = createEmbeddings(chunks, store_name)  # Compute or load embeddings for the chunks

        # If embeddings are successfully created or loaded
        if vectorstore:
            query = st.text_input(f"Ask me anything about your {store_name} file: ")  # Allow user to input a question

            # If the user input a question
            if query:
                response = getResponse(query, vectorstore)  # Get response using LLM model
                st.write(response)  # Display the response

# Entry point of the app
if __name__ == '__main__':
    load_dotenv()  # Load environment variables from .env file
    main()  # Run the main function and start the Streamlit app
