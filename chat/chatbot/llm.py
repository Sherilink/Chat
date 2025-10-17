from langchain.chat_models import ChatOpenAI
from django.conf import settings

def answer_question_from_docs(user, question):
    from langchain.vectorstores import FAISS
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.chains import RetrievalQA

    embed = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    store = FAISS.load_local(f"vectorstores/user_{user.id}", embed)
    retriever = store.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=settings.OPENAI_API_KEY)
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    return qa.run(question)

