# # app.py

# from flask import Flask, request, jsonify
# import os
# import tempfile
# from langchain.chains import ConversationalRetrievalChain
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain.memory import ConversationBufferMemory
# from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv

# app = Flask(__name__)

# load_dotenv()
# groq_api_key = os.environ["GROQ_API_KEY"]

# # Global variables to store the conversational chain and chat history
# chain = None
# chat_history = []

# def create_conversational_chain(vector_store):
#     llm = ChatGroq(
#         temperature=0.5,
#         model_name="mixtral-8x7b-32768",
#         groq_api_key=groq_api_key
#     )

#     memory = ConversationBufferMemory(
#         memory_key="chat_history",
#         return_messages=True
#     )

#     chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         chain_type="stuff",
#         retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
#         memory=memory
#     )

#     return chain

# @app.route('/upload', methods=['POST'])
# def upload_files():
#     global chain
    
#     if 'files[]' not in request.files:
#         return jsonify({"error": "No file part"}), 400
    
#     files = request.files.getlist('files[]')
    
#     text = []
#     for file in files:
#         file_extension = os.path.splitext(file.filename)[1]
#         with tempfile.NamedTemporaryFile(delete=False) as temp_file:
#             file.save(temp_file.name)
#             temp_file_path = temp_file.name

#         loader = None
#         if file_extension == ".pdf":
#             loader = PyPDFLoader(temp_file_path)
#         elif file_extension == ".docx" or file_extension == ".doc":
#             loader = Docx2txtLoader(temp_file_path)
#         elif file_extension == ".txt":
#             loader = TextLoader(temp_file_path)

#         if loader:
#             text.extend(loader.load())
#             os.remove(temp_file_path)

#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size=768,
#         chunk_overlap=128,
#         length_function=len
#     )
#     text_chunks = text_splitter.split_documents(text)

#     embedding = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2",
#         model_kwargs={"device": "cpu"}
#     )

#     vector_store = Chroma.from_documents(
#         documents=text_chunks,
#         embedding=embedding,
#         persist_directory="chroma_store_groq"
#     )

#     chain = create_conversational_chain(vector_store=vector_store)

#     return jsonify({"message": "Files processed successfully"}), 200

# @app.route('/chat', methods=['POST'])
# def chat():
#     global chain, chat_history
    
#     if not chain:
#         return jsonify({"error": "Please upload documents first"}), 400
    
#     user_input = request.json.get('message')
#     if not user_input:
#         return jsonify({"error": "No message provided"}), 400

#     result = chain({
#         "question": user_input,
#         "chat_history": chat_history
#     })
    
#     chat_history.append((user_input, result["answer"]))
    
#     return jsonify({"response": result["answer"]}), 200

# if __name__ == '__main__':
#     app.run(debug=True)



# app.py

from flask import Flask, request, jsonify
import os
import tempfile
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  

load_dotenv()
groq_api_key = os.environ["GROQ_API_KEY"]

# Global variables to store the conversational chain and chat history
chain = None
chat_history = []

def create_conversational_chain(vector_store):
    llm = ChatGroq(
        temperature=0.5,
        model_name="mixtral-8x7b-32768",
        groq_api_key=groq_api_key
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
        memory=memory
    )

    return chain

@app.route('/upload', methods=['POST'])
def upload_files():
    global chain
    print("uploading")
    
    if 'files[]' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    files = request.files.getlist('files[]')
    print("uploading",files)
    text = []
    for file in files:
        file_extension = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            file.save(temp_file.name)
            temp_file_path = temp_file.name

        loader = None
        if file_extension == ".pdf":
            loader = PyPDFLoader(temp_file_path)
        elif file_extension == ".docx" or file_extension == ".doc":
            loader = Docx2txtLoader(temp_file_path)
        elif file_extension == ".txt":
            loader = TextLoader(temp_file_path)

        if loader:
            text.extend(loader.load())
            os.remove(temp_file_path)

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=768,
        chunk_overlap=128,
        length_function=len
    )
    text_chunks = text_splitter.split_documents(text)
    print("here, split",text_chunks)
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    print("here, emb")
    vector_store = Chroma.from_documents(
        documents=text_chunks,
        embedding=embedding,
        persist_directory="chroma_store_groq"
    )
    print("here, store")
    chain = create_conversational_chain(vector_store=vector_store)
    print("here, chain")
    return jsonify({"message": "Files processed successfully"}), 200

@app.route('/chat', methods=['POST'])
def chat():
    global chain, chat_history
    
    if not chain:
        return jsonify({"error": "Please upload documents first"}), 400
    
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    result = chain({
        "question": user_input,
        "chat_history": chat_history
    })
    
    chat_history.append((user_input, result["answer"]))
    
    return jsonify({"response": result["answer"]}), 200

if __name__ == '__main__':
    app.run(debug=True)