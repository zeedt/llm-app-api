from flask import Flask
from werkzeug.utils import secure_filename
from flask import request
import os
import pinecone
from langchain.vectorstores import Pinecone
from langchain.chains import LLMChain, RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder, HumanMessagePromptTemplate, PipelinePromptTemplate
from langchain import PromptTemplate
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory, FileChatMessageHistory
from langchain.schema import SystemMessage


from langchain.vectorstores import Pinecone
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment=os.environ.get('PINECONE_ENV'))

llm  = ChatOpenAI(model='gpt-3.5-turbo', temperature=0, verbose=True)
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 10000, chunk_overlap = 100)

app_index = 'app-index'

# To be loaded anytime document is uploaded and the index is created in pinecone
vector_store = '' 

### Create encoding
### Ensure to update this method when you always want to retain all your indexes. Does not mater here cuz it's free
def create_index_in_pinecone(chunks):
    for idx in pinecone.list_indexes():
        pinecone.delete_index(idx)
    pinecone.create_index(app_index, dimension=1536, metric='cosine')
    
def load_vector(chunks):
    embeddings = OpenAIEmbeddings()
    return Pinecone.from_documents(embedding=embeddings, documents=chunks, index_name=app_index)

def load_document(uuid):
    from langchain.document_loaders import PyPDFLoader
    print(f'Loading {uuid}')
    loader = PyPDFLoader(f'{uuid}.pdf')
    data = loader.load()
    return data

def summarize_document(data):
    template = '''
    Generate a title for this summary and Summarise the document starting on a new paragraph 
    `{text}`
    '''
    prompt = PromptTemplate(
        input_variables=['text'],
        template=template
    )
    chain = LLMChain(prompt=prompt, llm=llm)
    return chain.run({'text': data})

def summarize_document_using_map_reduce_with_custom_prompt(chunks, description):
    map_prompt = '''
    Summarise the document below 
    `{text}`
    '''
    map_prompt_template = PromptTemplate(input_variables=['text'], template=map_prompt)
    combine_prompt = description +  ''' TEXT: `{text}` '''
    combine_prompt_template = PromptTemplate(input_variables=['text'], template=combine_prompt)
    chain = load_summarize_chain(llm=llm, 
                                 chain_type='map_reduce', 
                                 verbose=False, 
                                 map_prompt=map_prompt_template,
                                 combine_prompt=combine_prompt_template
                                )
    return chain.run(chunks)

def chatbot(app_uuid, user_uuid, message):
    global vector_store
    if (len(message) < 5):
        print('Message should be atleast 5 characters')
        return 'Message should be atleast 5 characters'
    chat_memory = FileChatMessageHistory(f'{user_uuid}-{app_uuid}-chat.json')  
    conversation_memory = ConversationBufferMemory(
        memory_key=f'{user_uuid}-{app_uuid}-chat', 
        chat_memory=chat_memory,
        return_messages = True
    )
    if (vector_store == ''):
        loaded_data = load_document(f'./files/{app_uuid}')
        chunks = text_splitter.split_documents(loaded_data)
        vector_store = load_vector(chunks)
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 3})
    chain = RetrievalQA.from_chain_type(retriever=retriever, llm=llm, memory=conversation_memory,)
    response = chain.run(message)
    print(type(response))
    print(response)
    return response

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "Hello, World!"

@app.route("/v1/summarize/<uuid>")
def summarize_doc_v1(uuid):
    loaded_data = load_document(f'./files/{uuid}')
    return summarize_document(loaded_data)


@app.route("/v2/summarize/<uuid>")
def summarize_doc_v2(uuid):
    loaded_data = load_document(f'./files/{uuid}')
    chunks = text_splitter.split_documents(loaded_data)
    description = '''
        Add title to the summary
    
    The summary should cover the following
    
    1. INTRODUCTION
    2. KEY POINTS
    3. PROBLEMS IF ANY
    4. ALTERNATIVES IF ANY
    5. CONCLUSION
    6. RECOMMENDATIONS IF ANY
    
    '''
    return summarize_document_using_map_reduce_with_custom_prompt(chunks, description)


@app.route("/v2/summarize", methods = ['POST'])
def summarize_with_custom_description_doc_v2():
    request_data = request.get_json()
    uuid = request_data['uuid']
    description = request_data['description']
    loaded_data = load_document(f'./files/{uuid}')
    chunks = text_splitter.split_documents(loaded_data)
    return summarize_document_using_map_reduce_with_custom_prompt(chunks, description)
    

@app.route("/load")
def load_data():
    loaded_data = load_document('./files/a5560595-70bf-446b-b641-fdc8c536da76')
    return str(len(loaded_data))


@app.route("/test/<value>")
def hello_world2(value):
    return "Hello, World! " + value

@app.route('/upload/<unique_id>', methods = ['POST'])   
def upload_file(unique_id):   
    if request.method == 'POST':   
        f = request.files['file'] 
        f.save(os.path.join('./files',secure_filename(unique_id+'.pdf')))
    loaded_data = load_document(f'./files/{unique_id}')
    chunks = text_splitter.split_documents(loaded_data)
    create_index_in_pinecone(chunks)
    vector_store = load_vector(chunks)
    return 'FIle saved'

@app.route("/v2/chatbot", methods = ['POST'])
def ask_question():
    request_data = request.get_json()
    uuid = request_data['uuid']
    message = request_data['message']
    loaded_data = load_document(f'./files/{uuid}')
    chunks = text_splitter.split_documents(loaded_data)
    return chatbot(uuid, 'test-user-id', message)