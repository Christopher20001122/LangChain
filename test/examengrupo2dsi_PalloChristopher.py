import os
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.document_loaders import UnstructuredHTMLLoader
import streamlit as st
from langchain.vectorstores import ElasticsearchStore, Vectara

os.environ['OPENAI_API_KEY'] = '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'
default_doc_name = 'doc.html'

def process_doc(
        path: str = '',
        is_local: bool = False,
        question: str = 'Cuáles son los autores?'
):
    _, loader = os.system(f'curl -o {default_doc_name} {path}') if not is_local \
        else None, UnstructuredHTMLLoader(f"./{default_doc_name}") if not is_local \
        else UnstructuredHTMLLoader(path)

    doc = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=80000, chunk_overlap=7500)
    texts = text_splitter.split_documents(doc)
    embedding = OpenAIEmbeddings()
    from langchain.vectorstores import FAISS
    faiss = FAISS.from_documents(texts, embedding)

    faiss = faiss.from_documents(
        texts,
        embedding=embedding,
    )

    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type='refine', retriever=faiss.as_retriever())
    st.write(qa.run(question))

def client():
    st.set_page_config(page_title='Gestionar LLM with LangChain', layout='wide')

    st.image('https://aka-cdn.uce.edu.ec/ares/tmp/logo3.png', width=150)
    st.title('Gestionar LLM con LangChain')

    st.markdown('## Subir Archivo HTML')
    uploader = st.file_uploader('Selecciona un archivo HTML', type='html')

    if uploader:
        with open(f'./{default_doc_name}', 'wb') as f:
            f.write(uploader.getbuffer())
        st.success('Archivo HTML guardado!')

    question = st.text_input('Pregunta',
                             placeholder='Haz una pregunta sobre el documento', disabled=not uploader)

    if st.button('Pregunta'):
        if uploader:
            with st.spinner('Procesando pregunta...'):
                result = process_doc(
                    path=default_doc_name,
                    is_local=True,
                    question=question
                )
                st.success('Pregunta procesada exitosamente!')
                st.write('Respuesta:', result)
        else:
            st.info('Por favor, sube el archivo HTML primero.')

if __name__ == '__main__':
    client()

    #streamlit run test/lang_script.py --server.port 2023

