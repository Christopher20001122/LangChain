import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.document_loaders import PyPDFLoader
import streamlit as st

os.environ['OPENAI_API_KEY'] = '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'
default_doc_name = 'doc.pdf'


def process_doc(
        path: str = 'https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf',
        is_local: bool = False,
        question: str = 'Cuáles son los autores del pdf?'
):
    _, loader = os.system(f'curl -o {default_doc_name} {path}'), PyPDFLoader(f"./{default_doc_name}") if not is_local \
        else PyPDFLoader(path)

    doc = loader.load_and_split()
    db = Chroma.from_documents(doc, embedding=OpenAIEmbeddings())
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type='stuff', retriever=db.as_retriever())
    return qa.run(question)


def main():
    st.set_page_config(page_title='Gestionar LLM con LangChain', layout='wide')

    st.image('https://aka-cdn.uce.edu.ec/ares/tmp/logo3.png', width=150)
    st.title('Manage LLM with LangChain')

    col1, col2 = st.columns(2)

    # Subir PDF
    with col1:
        st.markdown('## Subir PDF')
        uploader = st.file_uploader('Selecciona un archivo PDF', type='pdf', key='uploader')
        uploaded = False

        if uploader:
            with open(f'./{default_doc_name}', 'wb') as f:
                f.write(uploader.getbuffer())
            uploaded = True
            st.success('PDF cargado exitosamente!')

    # Pregunta
    with col2:
        st.markdown('## Hacer una Pregunta')
        if not uploaded:
            st.warning('Por favor, carga un PDF primero.')
        else:
            question = st.text_input('Escribe una pregunta sobre el PDF cargado',
                                     placeholder='Escribe aquí tu pregunta', key='question')
            if st.button('Enviar Pregunta', key='send_question'):
                with st.spinner('Procesando pregunta...'):
                    result = process_doc(
                        path=default_doc_name,
                        is_local=True,
                        question=question
                    )
                    st.success('Pregunta procesada!')
                    st.write('Respuesta:', result)


if __name__ == '__main__':
    main()

#streamlit run test/lang_script.py
