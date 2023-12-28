import streamlit as st


from PyPDF2 import PdfFileReader, PdfFileWriter,PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
#from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.llms import HuggingFaceHub


import pickle
import os
#load api key lib
#from dotenv import load_dotenv
import base64
import os

def main():
    st.header("📄Chat with your pdf file🤗")

    #upload a your pdf file
    pdf = st.file_uploader("Upload your PDF", type='pdf')
    try:
        st.write(pdf.name)
    except:
        pass
    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""
        for page in pdf_reader.pages:
            text+= page.extract_text()

        #langchain_textspliter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 200,
            length_function = len
        )

        chunks = text_splitter.split_text(text=text)

        
        #store pdf name
        store_name = pdf.name[:-4]
        
        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl","rb") as f:
                vectorstore = pickle.load(f)
            #st.write("Already, Embeddings loaded from the your folder (disks)")
        else:
            #embedding (Openai methods) 
            #embeddings = OpenAIEmbeddings()
            embeddings = HuggingFaceInstructEmbeddings(model_name = "hkunlp/instructor-xl")
            #Store the chunks part in db (vector)
            vectorstore = FAISS.from_texts(chunks,embedding=embeddings)

            with open(f"{store_name}.pkl","wb") as f:
                pickle.dump(vectorstore,f)
            
            #st.write("Embedding computation completed")

        #st.write(chunks)
        
        #Accept user questions/query

        query = st.text_input("Ask questions about related your upload pdf file")
        #st.write(query)

        if query:
            docs = vectorstore.similarity_search(query=query,k=3)
            #st.write(docs)
            
            #openai rank lnv process
            #llm = ChatOpenAI(openai_api_key="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")
            llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
            chain = load_qa_chain(llm=llm, chain_type= "stuff")
            
            with get_openai_callback() as cb:
                response = chain.run(input_documents = docs, question = query)
                print(cb)
            st.write(response)

    else:
        st.write('Please upload the PDF to get started')

if __name__=="__main__":
    main()
