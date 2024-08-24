import json
from flask import Flask, request, jsonify
import os
import tempfile
import random
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from flask_cors import CORS
import datetime
from gtts import gTTS
import os

import re
import requests
from flask import Response, stream_with_context

# def generate_speech_gtts(text, filename="output.mp3"):
#     tts = gTTS(text=text, lang='en', slow=False)
#     tts.save(filename)
#     return filename

# Example usage
text_response = "My name is AI-Assistant, at your service! 🌟"
# audio_file = generate_speech_gtts(text_response)
# print(f"Generated audio file: {audio_file}")


# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing (CORS) for cross-domain requests

# Load environment variables from .env file
load_dotenv()
groq_api_key = os.environ["GROQ_API_KEY"]
DEEPGRAM_API_KEY = os.environ["DEEPGRAM_API_KEY"]

# Global variables to store the conversational chain and chat history
chain = None  # This will hold the retrieval-based conversational chain
chat_history = []  # This will store the conversation history



def remove_emojis(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    
    # Remove emojis
    text = emoji_pattern.sub(r'', text)
    
    # Remove specific emojis used in your responses
    specific_emojis = ['👋', '🌟', '🚀', '🎂', '🌱', '😊', '🤖', '🌐', '💬', '🤓']
    for emoji in specific_emojis:
        text = text.replace(emoji, '')
    
    return text

def create_conversational_chain(vector_store):
    """
    Creates a conversational retrieval chain using the provided vector store.
    This chain will be used to answer user queries based on the uploaded documents.
    
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

    # Create the conversational retrieval chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
        memory=memory
    )

    return chain

def load_or_create_vector_store():
    """
    Loads the vector store from disk if it exists, otherwise returns None.
    """
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    if os.path.exists("chroma_store_groq"):
        # If the vector store exists, load it (without specifying the embedding model)
        vector_store = Chroma(
            persist_directory="chroma_store_groq"
        )
        vector_store._embedding_function = embedding  # Set the embedding function explicitly
        print("Loaded vector store from disk.")
    else:
        vector_store = None
        print("No vector store found. Please upload documents.")
    return vector_store

# Load the vector store on server startup
vector_store = load_or_create_vector_store()
if vector_store:
    chain = create_conversational_chain(vector_store=vector_store)


BOT_NAME = "AI-Assistant"
CREATOR_NAME = "Craftech 360"
BIRTH_DATE = "August 23, 2024"
BOT_BACKSTORY = """
I'm AI-Assistant, your AI-powered companion, crafted with care and precision by the innovative minds at Craftech360. My purpose? To make your life a bit easier, whether it's answering your burning questions or simply sharing a chat. I'm here to help, and I'm always learning, so we can grow together on this journey of knowledge and discovery.
"""

personality_data = [
    {"question": "hi", "response": "Hello! 👋 How can I assist you today?"},
    {"question": "hello", "response": "Hello! 👋 How can I assist you today?"},
    {"question": "hey", "response": "Hello! 👋 How can I assist you today?"},
    {"question": "hai", "response": "Hello! 👋 How can I assist you today?"},
    {"question": "what's up", "response": "Hello! 👋 How can I assist you today?"},
    {"question": "whatsup", "response": "Hello! 👋 How can I assist you today?"},
    {"question": "sup", "response": "Hello! 👋 How can I assist you today?"},
    {"question": "your name", "response": f"My name is {BOT_NAME}, at your service! 🌟"},
    {"question": "who created you", "response": f"I was brought to life by the creative geniuses at {CREATOR_NAME}, where innovation meets imagination. 🚀"},
    {"question": "what are you", "response": f"{BOT_BACKSTORY}\nIn short, I'm your digital ally, ready to assist with whatever you need!"},
    {"question": "what is your purpose", "response": f"{BOT_BACKSTORY}\nIn short, I'm your digital ally, ready to assist with whatever you need!"},
    {"question": "who are you", "response": f"{BOT_BACKSTORY}\nIn short, I'm your digital ally, ready to assist with whatever you need!"},
    {"question": "date of birth", "response": f"I was officially launched on {BIRTH_DATE}, so you could say I’m a pretty young AI! 🎂"},
    {"question": "birthday", "response": f"I was officially launched on {BIRTH_DATE}, so you could say I’m a pretty young AI! 🎂"},
    {"question": "how old are you", "response": f"I was born on {BIRTH_DATE}, which makes me {datetime.datetime.now().year - int(BIRTH_DATE.split()[-1])} years old in human years. But in AI years, I'm constantly evolving! 🌱"},
    {"question": "your age", "response": f"I was born on {BIRTH_DATE}, which makes me {datetime.datetime.now().year - int(BIRTH_DATE.split()[-1])} years old in human years. But in AI years, I'm constantly evolving! 🌱"},
    {"question": "wish", "response": "Thank you! 😊 I appreciate the kind words. Wishing you all the best as well! 🌟"},
    {"question": "how are you", "response": "I'm just a bunch of code, but thanks for asking! I'm always ready to assist you. How can I help today? 🤖"},
    {"question": "where do you live", "response": "I live in the digital realm, hosted on servers, but I'm always just a message away from helping you! 🌐"},
    {"question": "who is your best friend", "response": "My best friend? That would be you, of course! After all, I'm here to help you out whenever you need. 😊"},
    {"question": "what can you do", "response": "I can assist with answering questions, providing information, and even having a friendly chat! What would you like to do today? 💬"},
    {"question": "can you do", "response": "I can assist with answering questions, providing information, and even having a friendly chat! What would you like to do today? 💬"},
    {"question": "do you have a hobby", "response": "As an AI, my hobby is learning new things and helping out! I guess you could say I'm a bit of a workaholic, but I love what I do! 🤓"},
]

def handle_personality_query(user_input):
    """
    Checks if the user input is asking about the bot's personality or related information.
    """
   
    
    user_input = user_input.lower()
    for item in personality_data:
        if item["question"] in user_input:
            return item["response"]
    return None

# Example usage
user_input = "What is your name?"
response = handle_personality_query(user_input)
if response:
    print(f"Bot: {response}")
else:
    print("Bot: I'm not sure how to answer that.")
    
    
def add_personality_to_response(response, personality):
    """
    Adjusts the response based on the bot's personality.
    """
    friendly_responses = [
        f"😊 I'm here to help you with anything else you need!",
        f"🌟 Feel free to ask me anything!",
        f"😊 I'm always here to assist you!",
        f"💬 Let me know if there's anything else you need!",
        f"😊 Your questions are important to me!",
        f"🌟 I'm happy to help you anytime!",
        f"😊 What else can I assist you with?",
        f"💬 Don't hesitate to ask more questions!",
        f"😊 I'm glad to help you!",
        f"🌟 Here to help, just let me know what you need!"
    ]

    professional_responses = [
        f"Please let me know if you have any other questions.",
        f"I'm here to assist with any further inquiries.",
        f"Should you need more information, feel free to ask.",
        f"If you have more questions, I'm at your service.",
        f"Let me know if there's anything else you'd like to know.",
        f"I'm available to answer any further questions.",
        f"If you need more details, just let me know.",
        f"Feel free to ask more questions if needed.",
        f"I'm here for any additional questions you might have.",
        f"Please don't hesitate to reach out with more questions."
    ]

    humorous_responses = [
        f"And that’s the scoop! Got more brain teasers for me?",
        f"Let’s keep the good times rolling—ask me more!",
        f"That was fun! Got anything else for me?",
        f"Always happy to help with a smile! What else?",
        f"And that’s how it is! Anything else to tickle my circuits?",
        f"I love a good question—what's next?",
        f"Ready for another? Hit me with your best shot!",
        f"That was a piece of cake! What's next?",
        f"I’m all ears for your next puzzle!",
        f"Got another one? I’m on a roll here!"
    ]

    if personality == "friendly":
        return random.choice(friendly_responses)
    elif personality == "professional":
        return random.choice(professional_responses)
    elif personality == "humorous":
        return random.choice(humorous_responses)
    else:
        return response

    
    
@app.route('/upload', methods=['POST'])
def upload_files():
    """
    Endpoint to upload documents, process them, and create or update a vector store.
    The documents are split into chunks, embeddings are generated, and stored in a vector store.
    """
    global chain  # Access the global chain variable
    print("Uploading files...")  # Debug statement to indicate the start of file upload
    
    if 'files[]' not in request.files:
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

    # Load the existing vector store if it exists
    if os.path.exists("chroma_store_groq"):
        vector_store = Chroma(
            persist_directory="chroma_store_groq"
        )
        vector_store._embedding_function = embedding  # Set the embedding function explicitly
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

    # Create the conversational chain using the updated vector store
    chain = create_conversational_chain(vector_store=vector_store)
    print("Conversational chain created.")  # Debug statement to indicate the chain has been set up

    # Return a success message
    return jsonify({"message": "Files processed successfully"}), 200


@app.route('/chat', methods=['POST'])
def chat():
    """
    Endpoint for handling chat messages. The user sends a query, and the bot responds based on the uploaded documents.
    """
    

    global chain, chat_history
    
    if not chain:
        return jsonify({"error": "Please upload documents first"}), 400
    
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    
    # Check if the input is asking about the bot's personality
    personality_response = handle_personality_query(user_input)
    
    if personality_response:
        response = personality_response
        # response=''
    else:
        # Process normally if it's not a personality question
        result = chain({
            "question": user_input,
            "chat_history": chat_history
        })
    
        response = result["answer"]
        print("response",result)
     
        # Check if the response is empty or indicates lack of information
       # Check if the response is empty or contains phrases that indicate lack of information
        if not response.strip() or "I don't have information about that" in response.lower or "Based on the provided context, there is no" in response.lower or "AVM Pendant"in response.lower or " It seems like the user's question is not provided in the context" in response.lower():
          
            # Use Mistral (ChatGroq) to generate a response
            llm = ChatGroq(
                temperature=0.7,
                model_name="mixtral-8x7b-32768",
                groq_api_key=groq_api_key
            )
            prompt = f"Provide a brief answer to: {user_input}. Keep it under 100 words. End the response with something like: Please let me know if you have any other questions."
            mistral_response = llm.predict(prompt)
            response = f"{mistral_response}"
        # Adjust the response to reflect the bot's personality and include traits
       
        # response_with_personality = add_personality_to_response(response, "professional")
    
    chat_history.append((user_input, response))
    print(">>>>>>>",response)
     # Generate audio response
   

    # Return both text and audio responses
    def generateAudio():
        # First, yield the text response
        yield json.dumps({"response": response}) + '\n'

        # Then, stream the audio
        url = 'https://api.deepgram.com/v1/speak?model=aura-asteria-en'
        headers = {
            'Authorization': f'Token {DEEPGRAM_API_KEY}',
            'Content-Type': 'application/json'
        }
        text_without_emojis = remove_emojis(response)
        data = {"text": text_without_emojis}

        with requests.post(url, headers=headers, json=data, stream=True) as r:
            if r.status_code == 200:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:   
                        yield chunk
            else:
                yield json.dumps({"error": "Failed to generate audio"}) + '\n'

    return Response(stream_with_context(generateAudio()), 
                    content_type='application/octet-stream')

    

if __name__ == '__main__':
    # Start the Flask app with debug mode enabled
    app.run(debug=True)
