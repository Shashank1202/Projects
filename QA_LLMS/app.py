from flask import Flask, render_template, request
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.llms import GooglePalm
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
#from langchain.vectorstores.faiss import BM25Retriever

import os
from dotenv import load_dotenv
load_dotenv()
app = Flask(__name__)

llm = GooglePalm(api_key=os.getenv("GOOGLE_API_KEY"), temperature=0)

instructor_embeddings = HuggingFaceInstructEmbeddings()
vectordb_file_path = "faiss_index"

def create_vector_db():
    loader = CSVLoader(file_path='codebasics_faqs.csv', source_column='prompt')
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)
    vectordb.save_local("faiss_index")

def get_qa_chain():
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings)

    prompt_template = """ Give the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "reponse" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know". Don't try to make up an answer.

    CONTEXT: {context}

    QUESTIONS: {question}"""
    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    #retriever= BM25Retriever(vectordb)

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        #retriever=retriever,
                                        chain_type="stuff",
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

    return chain

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['user_input']
        chain = get_qa_chain()
        result = chain(user_input)
        return render_template('index.html', result=result, user_input=user_input)
    return render_template('index.html', result=None, user_input=None)

if __name__ == '__main__':
    create_vector_db()
    app.run(debug=True)
