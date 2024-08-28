import json
from flask import Flask, request, jsonify, Response
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
import re
import requests
from flask import stream_with_context

# Initialize the Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing (CORS) for cross-domain requests

# Load environment variables from .env file
load_dotenv()
groq_api_key = os.environ["GROQ_API_KEY"]
DEEPGRAM_API_KEY = os.environ["DEEPGRAM_API_KEY"]

# Global variables to store the conversational chain and chat history
chain = None
chat_history = []

# Bot personality data
BOT_NAME = "AI-Assistant"
CREATOR_NAME = "Craftech 360"
BIRTH_DATE = "August 23, 2024"
BOT_BACKSTORY = """
I'm AI-Assistant, your AI-powered companion, crafted with care and precision by the innovative minds at Craftech360. My purpose? To make your life a bit easier, whether it's answering your burning questions or simply sharing a chat. I'm here to help, and I'm always learning, so we can grow together on this journey of knowledge and discovery.
"""

personality_data = [
    {"question": "hai, how are you?", "response": "Hai 👋 I'm just a bunch of code, but thanks for asking! I'm always ready to assist you. How can I help today? 🤖"},
    {"question": "hi", "response": "Hello! 👋 How can I assist you today?"},
    {"question": "hello, how are you", "response": "Hello! I'm just a bunch of code, but thanks for asking! I'm always ready to assist you. How can I help today? 🤖"},
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
    {"question": "date of birth", "response": f"I was officially launched on {BIRTH_DATE}, so you could say I'm a pretty young AI! 🎂"},
    {"question": "birthday", "response": f"I was officially launched on {BIRTH_DATE}, so you could say I'm a pretty young AI! 🎂"},
    {"question": "how old are you", "response": f"I was born on {BIRTH_DATE}, which makes me {datetime.datetime.now().year - int(BIRTH_DATE.split()[-1])} years old in human years. But in AI years, I'm constantly evolving! 🌱"},
    {"question": "your age", "response": f"I was born on {BIRTH_DATE}, which makes me {datetime.datetime.now().year - int(BIRTH_DATE.split()[-1])} years old in human years. But in AI years, I'm constantly evolving! 🌱"},
    {"question": "wish", "response": "Thank you! 😊 I appreciate the kind words. Wishing you all the best as well! 🌟"},
    {"question": "how are you", "response": "I'm just a bunch of code, but thanks for asking! I'm always ready to assist you. How can I help today? 🤖"},
    {"question": "where do you live", "response": "I live in the digital realm, hosted on servers, but I'm always just a message away from helping you! 🌐"},
    {"question": "who is your best friend", "response": "My best friend? That would be you, of course! After all, I'm here to help you out whenever you need. 😊"},
    {"question": "what can you do", "response": "I can assist with answering questions, providing information, and even having a friendly chat! What would you like to do today? 💬"},
    {"question": "can you do", "response": "I can assist with answering questions, providing information, and even having a friendly chat! What would you like to do today? 💬"},
    {"question": "do you have a hobby", "response": "As an AI, my hobby is learning new things and helping out! I guess you could say I'm a bit of a workaholic, but I love what I do! 🤓"},
    {"question": "tell me about deloitte", "response": "Deloitte provides industry-leading audit, consulting, tax and advisory services to many of the world's most admired brands, including nearly 90% of the Fortune 500®. We strive to make an impact that matters by creating trust and confidence in a more equitable society. With approximately 457,000 people worldwide, our network spans more than 150 countries and territories."},
    {"question": "why does gen ai matter", "response": "Generative AI is at a pivotal moment, with promising experiments and use cases beginning to pay off. It's balancing high expectations with challenges such as data quality, investment costs, effective measurement, and an evolving regulatory landscape. Change management and deep organizational integration are critical to overcoming barriers, unlocking value, and building for the future of Gen AI."},
    {"question": "share some gen ai facts", "response": "Here are some Gen AI facts: 43% of employees across Asia Pacific are using Gen AI for work in 2024. Daily users save about 6.3 hours per week. Over 11 billion working hours across Asia Pacific will be impacted by AI each week. 67% of Gen AI users reported improved work or study satisfaction. 60% of students believe Gen AI has influenced career decisions. Younger workers are twice as likely to use Gen AI compared to older workers."},
    {"question": "what value does deloitte's gen ai coe enable for clients", "response": "Deloitte's Gen AI COE enables various values for clients, including: Increasing revenue through hyper-personalized marketing, speeding up new product and service development, reducing costs by automating job functions, uncovering new ideas and insights, and unleashing creativity."},
    {"question": "what are deloitte's gen ai coe capabilities", "response": "Deloitte has expertise across Gen AI strategy, implementation, and monitoring. We have 37,000+ global AI, Analytics & Reasoning professionals, including 1200+ Prompt Engineers. Deloitte is recognized as a leader in Worldwide AI Services by IDC and has alliances with leading Gen AI tech players like NVIDIA and Google."},
    {"question": "tell me about deloitte's gen ai coe", "response": "Deloitte's Gen AI COE aligns overall vision to realize immediate value, foster accelerated innovation, track the rapidly evolving landscape, and provide guardrails for innovation. It consists of five pillars: AI Research Lab, AI Institute, AI Enable, AI Leap, and AI Consortium."},
    {"question": "what is the ai research lab", "response": "The AI Research Lab is part of Deloitte's Gen AI COE. It provides Generative AI Insights, an Experience Garage for experimenting with various models, and an Incubator to accelerate and expand Generative AI capabilities."},
    {"question": "what is the ai institute", "response": "The AI Institute is Deloitte's engine for bridging the AI talent gap. It offers AI training programs, including Learning Pathways, AI Experiential Learning Program, and AI Certification Programs."},
    {"question": "what is ai enable", "response": "AI Enable brings forth innovative solutions including Concept to Activation methodology, Lifecycle Edge AI Platform, and AI Factory as-a-Service to help organizations adapt to and harness the power of Gen AI."},
    {"question": "what is ai leap", "response": "AI Leap is a methodology to develop comprehensive plans for bringing Gen AI services and products to market. It includes Advisory services, a Sales Excellence Program, and Pursuit Axis for access to sales and business development resources."},
    {"question": "what is the ai consortium", "response": "The AI Consortium is a single source for knowledge management and asset repository with over 200 AI assets across industries and business functions. It includes an Asset Catalogue, AI Ignition series, and AI Dossier to help organizations improve productivity and efficiency."}
]

def remove_emojis(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE)
    
    text = emoji_pattern.sub(r'', text)
    
    specific_emojis = ['👋', '🌟', '🚀', '🎂', '🌱', '😊', '🤖', '🌐', '💬', '🤓']
    for emoji in specific_emojis:
        text = text.replace(emoji, '')
    
    return text

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
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
        memory=memory
    )

    return chain



def load_or_create_vector_store():
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    
    if os.path.exists("chroma_store_groq"):
        vector_store = Chroma(
            persist_directory="chroma_store_groq"
        )
        vector_store._embedding_function = embedding
        print("Loaded vector store from disk.")
    else:
        vector_store = None
        print("No vector store found. Please upload documents.")
    return vector_store

vector_store = load_or_create_vector_store()
if vector_store:
    chain = create_conversational_chain(vector_store=vector_store)

def handle_personality_query(user_input):
    user_input = user_input.lower()
    best_match = None
    max_words = 0

    for item in personality_data:
        question_words = set(item["question"].lower().split())
        input_words = set(user_input.split())
        common_words = question_words.intersection(input_words)
        
        if len(common_words) > max_words:
            max_words = len(common_words)
            best_match = item["response"]
        
      
    
    # If we found a match with at least 2 common words, return it
    if max_words >= 2:
        return best_match

    return None

@app.route('/upload', methods=['POST'])
def upload_files():
    global chain
    print("Uploading files...")
    
    if 'files[]' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    files = request.files.getlist('files[]')
    print("Files received:", files)

    text = []
    for file in files:
        file_extension = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            file.save(temp_file.name)
            temp_file_path = temp_file.name

        loader = None
        if file_extension == ".pdf":
            loader = PyPDFLoader(temp_file_path)
        elif file_extension in [".docx", ".doc"]:
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
    print("Text has been split into chunks:", text_chunks)

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    print("Embeddings generated.")

    if os.path.exists("chroma_store_groq"):
        vector_store = Chroma(
            persist_directory="chroma_store_groq"
        )
        vector_store._embedding_function = embedding
        print("Loaded existing vector store from disk.")
        vector_store.add_documents(documents=text_chunks)
    else:
        vector_store = Chroma.from_documents(
            documents=text_chunks,
            embedding=embedding,
            persist_directory="chroma_store_groq"
        )
        print("Created new vector store.")

    chain = create_conversational_chain(vector_store=vector_store)
    print("Conversational chain created.")

    return jsonify({"message": "Files processed successfully"}), 200

def filter_results_based_on_similarity(results, threshold=0.6):
    filtered_results = []
    for result in results:
        if result['similarity_score'] >= threshold:
            filtered_results.append(result)
    return filtered_results

def is_unsatisfactory_response(response):
    """
    Determine if the response is unsatisfactory.
    Returns True if the response is unsatisfactory, otherwise False.
    """
    # Check if the response is empty or None
    if not response:
        return True
    
    # Check for common fallback or generic phrases
    unsatisfactory_phrases = [
        "I don't have information about that",
        "Based on the provided context, there is no information about",
        "I'm not sure",
        "I don't have a clear answer",
        "Sorry, I couldn't find any information"
    ]
    
    for phrase in unsatisfactory_phrases:
        if phrase in response:
            return True
    
    # Additional checks can be added here, such as checking response length or specific content
    
    return False

@app.route('/chat', methods=['POST'])
def chat():
    global chain, chat_history
 
    if not chain:
        return jsonify({"error": "Please upload documents first"}), 400
    
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    
    personality_response = handle_personality_query(user_input)
    if personality_response:
        response = personality_response
    else:
        # Get the result from the ConversationalRetrievalChain
        result = chain({
            "question": user_input,
        })
        response = result["answer"]  # Directly use the answer generated by the chain
        
        print("resul")
        print(response)
        # If the response is not satisfactory, fall back to the LLM directly
        if not response or is_unsatisfactory_response(response):
            print("unsatisfactory result");
            llm = ChatGroq(
                temperature=0.7,
                model_name="mixtral-8x7b-32768",
                groq_api_key=groq_api_key
            )
            prompt = f"Provide a brief answer to: {user_input}. Keep it under 100 words."
            response = llm.predict(prompt)

    chat_history.append((user_input, response))
    print(">>>>>>>", response)


    # Rest of the function remains the same...

    def generate():
        yield json.dumps({"response": response}) + '\n'

        url = 'https://api.deepgram.com/v1/speak?model=aura-asteria-en'
        headers = {
            'Authorization': f'Token {DEEPGRAM_API_KEY}',
            'Content-Type': 'application/json'
        }
        text_without_emojis = remove_emojis(response)
        data = {"text": text_without_emojis}

        with requests.post(url, headers=headers, json=data, stream=True) as r:
            if r.status_code == 200:
                yield b'--audio\n'
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        yield chunk
                yield b'\n--audio--\n'
            else:
                yield json.dumps({"error": "Failed to generate audio"}) + '\n'

    return Response(stream_with_context(generate()), 
                    content_type='multipart/mixed; boundary=audio')
if __name__ == '__main__':
    app.run(debug=True)