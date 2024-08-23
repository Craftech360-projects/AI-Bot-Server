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

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing (CORS) for cross-domain requests

# Load environment variables from .env file
load_dotenv()
groq_api_key = os.environ["GROQ_API_KEY"]

# Global variables to store the conversational chain and chat history
chain = None  # This will hold the retrieval-based conversational chain
chat_history = []  # This will store the conversation history

def create_conversational_chain(vector_store):
    """
    Creates a conversational retrieval chain using the provided vector store.
    This chain will be used to answer user queries based on the uploaded documents.
    """
    # Initialize the language model with the Groq API key
    llm = ChatGroq(
        temperature=0.5,  # Controls the randomness of the output
        model_name="mixtral-8x7b-32768",
        groq_api_key=groq_api_key
    )

    # Create a memory object to store conversation history
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    """"
    conversational retrieval chain
    
    llm: This is the language model that will generate the responses. In your case, llm is an instance of the ChatGroq model.
        chain_type: This specifies how the retrieved documents are combined with the user's question before being passed to the LLM.
    Possible Values:
    "stuff": The retrieved documents are concatenated together ("stuffed") and passed to the language model as a single input.
    "map_reduce": The retrieved documents are processed in smaller chunks. The LLM generates answers for each chunk, and then these answers are aggregated (reduced) to form the final response.
    "refine": The initial response is generated based on the first retrieved document, and then subsequent documents are used to refine that answer.

    Effect: The "stuff" chain type is generally faster but may be limited by the LLM's context window (the maximum amount of text the model can process at once). This method works well when the amount of relevant text is relatively small.



    c. retriever=vector_store.as_retriever(search_kwargs={"k": 2})
    retriever: This defines how the documents are retrieved based on the user's query.
    vector_store.as_retriever(): This method converts the vector_store (which holds the document embeddings) into a retriever object. The retriever is responsible for finding the most relevant document chunks based on the user's question.
    search_kwargs={"k": 2}: This parameter specifies additional options for the retrieval process:
    k: This indicates the number of top results to return from the vector store. In this case, it will retrieve the top 2 most relevant document chunks.
    Effect: Setting k=2 means that for every query, the retriever will find the top 2 most relevant pieces of text. These pieces will then be fed into the language model to help generate an informed response. The higher the value of k, the more text is retrieved, which can improve the response quality but may also make the process slower and more resource-intensive.
    
    
     memory=memory
    memory: This manages the conversation history, ensuring the chatbot can maintain context across multiple interactions.
    ConversationBufferMemory: The specific memory type you're using, ConversationBufferMemory, stores all the messages exchanged in the conversation (both from the user and the bot).
    Effect: By using this memory, the chatbot can generate responses that consider the entire conversation history, not just the most recent user input. This is crucial for maintaining coherent and contextually appropriate conversations, especially in multi-turn dialogues.
    
    
    """
    # Create the conversational retrieval chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
        memory=memory
    )

    return chain

"""loading existing vector store"""

# def load_or_create_vector_store():
#     """
#     Loads the vector store from disk if it exists, otherwise returns None.
#     """
#     if os.path.exists("chroma_store_groq"):
#         # If the vector store exists, load it (without specifying the embedding model)
#         vector_store = Chroma(
#             persist_directory="chroma_store_groq"
#         )
#         print("Loaded vector store from disk.")
#     else:
#         vector_store = None
#         print("No vector store found. Please upload documents.")
#     return vector_store


# # Load the vector store on server startup
# vector_store = load_or_create_vector_store()
# if vector_store:
#     chain = create_conversational_chain(vector_store=vector_store)


@app.route('/upload', methods=['POST'])
def upload_files():
    """
    Endpoint to upload documents, process them, and create a vector store.
    The documents are split into chunks, embeddings are generated, and stored in a vector store.
    """
    global chain  # Access the global chain variable
    print("Uploading files...")  # Debug statement to indicate the start of file upload
    
    if 'files[]' not in request.files:
        # Return an error if no files were uploaded
        return jsonify({"error": "No file part"}), 400
    
    files = request.files.getlist('files[]')
    print("Files received:", files)  # Debug statement to list the uploaded files

    text = []  # Initialize an empty list to store the text from all documents
    for file in files:
        file_extension = os.path.splitext(file.filename)[1]  # Get the file extension
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            file.save(temp_file.name)  # Save the uploaded file to a temporary location
            temp_file_path = temp_file.name

        # Determine the appropriate loader based on file extension
        loader = None
        if file_extension == ".pdf":
            loader = PyPDFLoader(temp_file_path)
        elif file_extension in [".docx", ".doc"]:
            loader = Docx2txtLoader(temp_file_path)
        elif file_extension == ".txt":
            loader = TextLoader(temp_file_path)

        # If a loader was found, process the file and append the text to the list
        if loader:
            text.extend(loader.load())
            os.remove(temp_file_path)  # Clean up by removing the temporary file

    # Split the text into smaller chunks for easier processing
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=768,
        chunk_overlap=128,
        length_function=len
    )
    text_chunks = text_splitter.split_documents(text)
    print("Text has been split into chunks:", text_chunks)  # Debug statement to show the chunks

    # Generate embeddings for the text chunks using a pre-trained model
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    print("Embeddings generated.")  # Debug statement to indicate embeddings creation

    # Store the text chunks and their embeddings in a vector database (Chroma)
    # vector_store = Chroma.from_documents(
    #     documents=text_chunks,
    #     embedding=embedding,
    #     persist_directory="chroma_store_groq"
    # )
    # print("Vector store created and saved.")  # Debug statement to indicate vector store creation
    
    # Load the existing vector store if it exists
    if os.path.exists("chroma_store_groq"):
        vector_store = Chroma(
            persist_directory="chroma_store_groq",
            embedding=embedding
        )
        print("Loaded existing vector store from disk.")
        # Add new documents to the existing vector store
        vector_store.add_documents(documents=text_chunks)
    else:
        # Create a new vector store if it doesn't exist
        vector_store = Chroma.from_documents(
            documents=text_chunks,
            embedding=embedding,
            persist_directory="chroma_store_groq"
        )
        print("Created new vector store.")

    # Create the conversational chain using the vector store
    chain = create_conversational_chain(vector_store=vector_store)
    print("Conversational chain created.")  # Debug statement to indicate the chain has been set up

    # Return a success message
    return jsonify({"message": "Files processed successfully"}), 200

@app.route('/chat', methods=['POST'])
def chat():
    """
    Endpoint for handling chat messages. The user sends a query, and the bot responds based on the uploaded documents.
    """
    global chain, chat_history  # Access the global chain and chat_history variables
    
    if not chain:
        # If no documents have been uploaded yet, return an error
        return jsonify({"error": "Please upload documents first"}), 400
    
    user_input = request.json.get('message')  # Get the user message from the request
    if not user_input:
        # If no message was provided, return an error
        return jsonify({"error": "No message provided"}), 400

    # Process the user's input through the conversational chain
    result = chain({
        "question": user_input,
        "chat_history": chat_history
    })
    
    # Append the user input and the bot's response to the chat history
    chat_history.append((user_input, result["answer"]))
    
    # Return the bot's response as JSON
    return jsonify({"response": result["answer"]}), 200

if __name__ == '__main__':
    # Start the Flask app with debug mode enabled
    app.run(debug=True)
