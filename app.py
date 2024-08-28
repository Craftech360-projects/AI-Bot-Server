
import json
from flask import Flask, request, jsonify
import os
import tempfile
import random
import difflib
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


# Example usage
text_response = "My name is AI-Assistant, at your service! 🌟"


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
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
        memory=memory
    )

    return chain

# def load_or_create_vector_store():
#     """
#     Loads the vector store from disk if it exists, otherwise returns None.
#     """
#     embedding = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2",
#         model_kwargs={"device": "cpu"}
#     )
    
#     if os.path.exists("chroma_store_groq"):
#         # If the vector store exists, load it (without specifying the embedding model)
#         vector_store = Chroma(
#             persist_directory="chroma_store_groq"
#         )
#         vector_store._embedding_function = embedding  # Set the embedding function explicitly
#         print("Loaded vector store from disk.")
#     else:
#         vector_store = None
#         print("No vector store found. Please upload documents.")
#     return vector_store

# # Load the vector store on server startup
# vector_store = load_or_create_vector_store()
# if vector_store:
#     chain = create_conversational_chain(vector_store=vector_store)


BOT_NAME = "Zephy"
CREATOR_NAME = "Craftech 360"
BIRTH_DATE = "August 23, 2024"
BOT_BACKSTORY = """
I'am Zephy, an AI-Assistant, your AI-powered companion. My purpose? To make your life a bit easier, whether it's answering your burning questions or simply sharing a chat. I'm here to help.
"""

personality_data = [
     {"question": "about deloitte", "response": "Deloitte provides industry-leading audit, consulting, tax and advisory services to many of the world’s most admired brands, including nearly 90% of the Fortune 500®. At Deloitte, we strive to live our purpose of making an impact that matters by creating trust and confidence in a more equitable society. We leverage our unique blend of business acumen, command of technology, and strategic technology alliances to advise our clients across industries as they build their future. Deloitte is proud to be part of the largest global professional services network serving our clients in the markets that are most important to them. Bringing more than 175 years of service, our network of member firms spans more than 150 countries and territories. Learn how Deloitte’s approximately 457,000 people worldwide connect for impact at www.deloitte.com"},
    
    {"question": "why it gen ai matters?", "response": "As promising experiments and use cases begin to pay off, it’s clear that we have arrived at a pivotal moment for Generative AI, balancing leaders’ high expectations with challenges such as data quality, investment costs, effective measurement and an evolving regulatory landscape. Now more than ever, change management and deep organizational integration are critical to overcoming barriers, unlocking value and building for the future of Gen AI."},
    
    {"question": "share gen ai facts with us", "response": "Here are some Generative AI facts and statistics: 43% of employees across Asia Pacific are using Gen AI for work in 2024; Daily Gen AI users save approximately 6.3 hours per week; Over 11 billion working hours across Asia Pacific will be impacted by AI weekly; 67% of Gen AI users reported improved work or study satisfaction; 60% of students believe Gen AI has influenced their career decisions; Younger workers are twice as likely to use Gen AI compared to older workers."},
    
    {"question": "value levers we enable for our clients using gen ai", "response": "Gen AI COE enable Values across Client Business areas for example: Increase revenue generation through hyper personalized marketing for target customers through Content Generation, Increase pace of new product & service development and speedier GTM, Reduce cost around 30% or higher through automating job functions & undertaking job substitutions, Uncover new ideas, insights, questions and generally unleash creativity and much more"},
    
   
    {"question": "gen ai coe", "response": "Deloitte has well established Gen AI COE to align overall vision in order to  realize immediate value, foster accelerated innovation, track the rapidly evolving landscape, provision guardrails for innovation to gain a competitive advantage, and ensure the ethical and responsible use of AI. Deloitte Gen AI COE serves as the nucleus fostering an environment of innovation and ensuring maximum value and impact through 5 Pillars: First One is AI Research Lab: Where innovation meets intelligence: Discover leading-edge insights, leverages deep industry knowledge. Second is AI Institute: A Leading-edge educational institute, rapidly develops new-age digital skills for professionals.Third is AI Enable: that Transform Delivery to creates a profound potential for innovation and optimization across all service lines. Fourth is AI Leap: which Accelerate Go-to-market alignment with strategic partners and rapid development via the AI Enable Incubator. And Lastly is AI Consortium to Shape new businesses and markets with Centralized AI Assets Repository that drive big, bold, globally-impactful play."},
    
     {"question": "deloitte gen ai coe capabilities", "response": "At Deloitte we have expertise and understanding which allows us to be a supportive partner to you across your strategy, implementation and monitor Gen AI journey stages with our High quality talent at scale. We have around 37,000+ Global AI, Analytics & Reasoning Talent strength including Semantic Reasoning & Inference, Natural Learning Techniques, Conversational AI professionals and 1200+ Prompt Engineers. Deloitte named a leader in the Worldwide AI Services 2023 Vendor Assessment by IDC – 3 times in a row and has Alliances with leading Gen AI tech players Like nvidia and Google, recognized as a Global Consulting & Service Partner of the year."},
    
    {"question": "ai research lab", "response": "From the Market to the Lab, the Deloitte AI Institute leverages deep industry knowledge to lead the AI conversation and uncover insights to make sense of this complex ecosystem. And for the same, Gen AI Research Lab provision 3 main components: First One is Generative AI Insights to explore cutting-edge insights, trends, industry-specific AI use cases, and major studies along with publications that will revolutionize the business. Second is Experience Garage which is Experimental Portal with access of variety of Modals that offers experiencing and experimenting to all with assets like Starter Kit, Templated Use Cases, Fluency Programs and more And Lastly An Incubator to Accelerate and expand Generative AI capabilities, seize the momentum, and provide early-stage proof of concepts"},
    
    {"question": "ai institute", "response": "Deloitte’s engine for bridging the AI talent gap by educating way to AI talent growth. The Institute offers AI training programs for all professionals. Programs are categorized into technical and nontechnical, based on each professional's learning objectives. In addition, we provision events, newsletters, hackathons and ideathons, for Generative Learning.  AI Institute has 3 main programs: First One is Learning Pathways, A comprehensive curriculum across business and industries along with innovative learning solutions that enable professionals to create an impact. Second is AI Experiential Learning Program which provision valuable hands-on experience with AI technologies by participating Learning programs, Hackathons, and much more. And Lastly is AI Certification Programs to Empower professionals with certification around AI Automation, Data Engineer, ML Engineer, or Full Stack Developer"},
    
    {"question": "ai enable", "response": "As the influence of Generation AI continues to reshape Industry, it’s crucial for organizations to adapt and embrace the rapid advancements in AI technology. The future of work depends on harnessing the power of gen AI in a way that fosters collaboration and innovation. And for the same, AI Enable bring forth 3 Innovative Solutions:.First One is Concept to Activation with Proven methodology to enable deep industry expertise from Ideate, Deliver, and to Scale, to make sense of this complex ecosystem and bridging the ethics gap surrounding AI. Second is Lifecycle Edge AI Platform with capabilities to enable from Knowledge Based Assistance to Action Oriented AI to provision productivity across Software Development Lifecycle. And Lastly is, AI Factory as-a-Service that is one-stop solution for Gen AI, with comprehensive suite of capabilities, delivered with experienced professionals and deep industry experience"},
    
    {"question": "ai leap", "response": "AI Leap is a methodology to develop a comprehensive plan to bring Gen AI services and product to market. It is designed to mitigate the risk and provision a Singular One Services to our Clients through 3 Levers: First One is Advisory, A dedicated team that activate value through awareness of potentials, enablement of services, and provision expertise to increase adoption. Second is Sales Excellence Program to develop a roadmap to greatly enhance win rates and collaboration, eliminate obstacles, and streamline our efforts to secure key strategic deals globally. And Lastly is Pursuit Axis for access to sales and business development training, tips, and tools, as well as a central repository for knowledge, resources, and strategies to assist with proposals, meeting, and other client-facing materials"},
    
    {"question": "ai consortium", "response": "AI Consortium with 200+ AI Assets across Industry and Business Functions like Financial Services, Technology, Media & Telecom and much more acts as a Single Source for Knowledge management and Assets Repository to help organization improve productivity, retain information, and maximize efficiency with 3 main components: First One is Asset Catalogue a collection of assets across business, industries, and software development lifecycle including Knowledge Based Assistances, Action Oriented AI and much more. Second is AI Ignition, A series of newsletters, videos, and podcasts which illuminate knowledge and ignite the discussion across latest trends in technologies and its potential realization. And Lastly is AI Dossier, A curated selection of Generative AI use cases, videos, and much more aimed at inspiring ideas, uncovering value-driven programs, and guiding organizations toward maximizing the potential"},
   
    {"question": "about zephy", "response": f"{BOT_BACKSTORY}. In short, I'm your digital ally, ready to assist with whatever you need!"},
    
    {"question": "hi", "response": "Hello!  How can I assist you today?"},
    {"question": "hello", "response": "Hello!  How can I assist you today?"},
    {"question": "hello, how are you", "response": "Hello! I'm just a bunch of code, but thanks for asking! I'm always ready to assist you. How can I help today? "},
    {"question": "hai, how are you?", "response": "Hai  I'm just a bunch of code, but thanks for asking! I'm always ready to assist you. How can I help today? "},
    {"question": "hey", "response": "Hello!  How can I assist you today?"},
    {"question": "hai", "response": "Hello!  How can I assist you today?"},
    {"question": "what's up", "response": "Hello!  How can I assist you today?"},
    {"question": "whatsup", "response": "Hello!  How can I assist you today?"},
    {"question": "sup", "response": "Hello!  How can I assist you today?"},
    {"question": "your name", "response": f"My name is {BOT_NAME}, at your service! "},
    {"question": "who created you", "response": f"I was brought to life by the creative geniuses at {CREATOR_NAME}, where innovation meets imagination. "},
    {"question": "what are you", "response": f"{BOT_BACKSTORY}. In short, I'm your digital ally, ready to assist with whatever you need!"},
    {"question": "what is your purpose", "response": f"{BOT_BACKSTORY}. In short, I'm your digital ally, ready to assist with whatever you need!"},
    {"question": "who are you", "response": f"{BOT_BACKSTORY}. In short, I'm your digital ally, ready to assist with whatever you need!"},
    
    {"question": "date of birth", "response": f"I was officially launched on {BIRTH_DATE}, so you could say I’m a pretty young AI! "},
    {"question": "birthday", "response": f"I was officially launched on {BIRTH_DATE}, so you could say I’m a pretty young AI! "},
    {"question": "how old are you", "response": f"I was born on {BIRTH_DATE}, which makes me {datetime.datetime.now().year - int(BIRTH_DATE.split()[-1])} years old in human years. But in AI years, I'm constantly evolving! "},
    {"question": "your age", "response": f"I was born on {BIRTH_DATE}, which makes me {datetime.datetime.now().year - int(BIRTH_DATE.split()[-1])} years old in human years. But in AI years, I'm constantly evolving! "},
    {"question": "wish", "response": "Thank you!  I appreciate the kind words. Wishing you all the best as well! "},
    {"question": "how are you", "response": "I'm just a bunch of code, but thanks for asking! I'm always ready to assist you. How can I help today? "},
    {"question": "where do you live", "response": "I live in the digital realm, hosted on servers, but I'm always just a message away from helping you! "},
    {"question": "who is your best friend", "response": "My best friend? That would be you, of course! After all, I'm here to help you out whenever you need. "},
    {"question": "what can you do", "response": "I can assist with answering questions, providing information, and even having a friendly chat! What would you like to do today? "},
    {"question": "can you do", "response": "I can assist with answering questions, providing information, and even having a friendly chat! What would you like to do today? "},
    {"question": "do you have a hobby", "response": "As an AI, my hobby is learning new things and helping out! I guess you could say I'm a bit of a workaholic, but I love what I do! "},
]

def handle_personality_query(user_input):
    """
    Checks if the user input is asking about the bot's personality or related information.
    """
   
    
    # user_input = user_input.lower()
    # for item in personality_data:
    #     if item["question"] in user_input:
    #         return item["response"]
    # return None

    # user_inputLower = user_input.lower()
    # print(">user_inputLower>>",user_inputLower);
    # for item in personality_data:
    #   if item["question"].lower() == user_inputLower:
    #     matched_question = item["question"]
    #     matched_response = item["response"]
    #     print(f"Matched question: {matched_question}")
    #     print(f"Matched response: {matched_response}")
    #     return item["response"]
    # return None
    user_input = user_input.lower()
    
    # First, try to find an exact match
    for item in personality_data:
        if item["question"].lower() == user_input:
            print(f"Exact match found: {item['question']}")
            return item["response"]
    
    # If no exact match, find the closest match
    questions = [item["question"].lower() for item in personality_data]
    closest_match = difflib.get_close_matches(user_input, questions, n=1, cutoff=0.7)  # Adjust cutoff as needed

    if closest_match:
        for item in personality_data:
            if item["question"].lower() == closest_match[0]:
                print(f"Closest match found: {item['question']}")
                return item["response"]
    
    # Fallback response if no close match is found
    print("No close match found.")
    return None

 # In case no match is found

# Example usage
# user_input = "About Deloitte"
# response = handle_personality_query(user_input)
# if response:
#     print(f"Bot: {response}")
# else:
#     print("Bot: I'm not sure how to answer that.")
    
    
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
    
    # if not chain:
    #     return jsonify({"error": "Please upload documents first"}), 400
    
    user_input = request.json.get('message')
    
    print("userinputt>>>>",user_input)
    if not user_input:
        return jsonify({"error": "No message provided"}), 400
    
    # Check if the input is asking about the bot's personality
    personality_response = handle_personality_query(user_input)
    
    if personality_response:
        response = personality_response
        # response=''
    else:
       
            
        professional_responses = [
        "Please let me know if you have any other questions.",
        "I'm here to assist with any further inquiries.",
        "Should you need more information, feel free to ask.",
        "If you have more questions, I'm at your service.",
        "Let me know if there's anything else you'd like to know.",
        "I'm available to answer any further questions.",
        "If you need more details, just let me know.",
        "Feel free to ask more questions if needed.",
        "I'm here for any additional questions you might have.",
        "Please don't hesitate to reach out with more questions."
             ]
        
        llm = ChatGroq(
                temperature=0.7,
                model_name="mixtral-8x7b-32768",
                groq_api_key=groq_api_key
            )
        print("user_input",user_input)
        prompt = f"Provide a brief answer to: {user_input}. end response something like this,{random.choice(professional_responses)}."
        mistral_response = llm.predict(prompt)
        response = f"{mistral_response}"
        # Adjust the response to reflect the bot's personality and include traits
       
        # response_with_personality = add_personality_to_response(response, "professional")
    
    chat_history.append((user_input, response))
    print(">>>>>>>",response)
     # Generate audio response
    return jsonify({"response": response})


if __name__ == '__main__':
    # Start the Flask app with debug mode enabled
     app.run(debug=True) 