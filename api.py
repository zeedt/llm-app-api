from flask import Flask, request
import os
# from summarize import load_document, load_vector, text_splitter, vector_store, chatbot, summarize_document_using_map_reduce_with_custom_prompt, create_index_in_pinecone
from werkzeug.utils import secure_filename
from summarize import *

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


if __name__ == '__main__':
    app.run(debug=True, port=8111)