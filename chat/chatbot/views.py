from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib.auth import login, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.http import JsonResponse, HttpResponse
from .models import ChatThread, Message, Document
import os
from dotenv import load_dotenv

from langchain_community.llms import LlamaCpp
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

load_dotenv()
LLAMA_MODEL_PATH = os.getenv("LLAMA_MODEL_PATH", "models/phi-2.Q4_K_M.gguf")
_llm_instance = None

def get_llama_model():
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LlamaCpp(model_path=LLAMA_MODEL_PATH)
    return _llm_instance

def prepare_doc_retriever(doc_path, filetype='txt'):
    if filetype == 'pdf':
        loader = UnstructuredPDFLoader(doc_path)
        pages = loader.load()
        doc_text = "\n".join([page.page_content for page in pages])
    else:
        with open(doc_path, "r", encoding="utf-8", errors="ignore") as f:
            doc_text = f.read()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(doc_text)
    embeddings = HuggingFaceEmbeddings()
    vector_db = FAISS.from_texts(chunks, embeddings)
    retriever = vector_db.as_retriever()
    return retriever

def get_llama_response(user_message, chat_history=None):
    llm = get_llama_model()
    system_prompt = "<<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n"
    prompt = "[INST] " + system_prompt
    if chat_history:
        for exchange in chat_history[-2:]:
            prompt += f"{exchange['user']}\n{exchange['bot']}\n"
    prompt += f"{user_message} [/INST]"
    output = llm(prompt, max_tokens=200)
    print("DEBUG Llama Output:", output)  
    if isinstance(output, dict):
        return output["choices"][0]["text"].strip()
    elif isinstance(output, str):
        return output.strip()
    else:
        return str(output)

def signup_view(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect("chat_home")
    else:
        form = UserCreationForm()
    return render(request, "signup.html", {"form": form})

def login_view(request):
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            login(request, form.get_user())
            return redirect("chat_home")
    else:
        form = AuthenticationForm()
    return render(request, "login.html", {"form": form})

def logout_view(request):
    logout(request)
    return redirect("login")

def home_view(request):
    chat_history = request.session.get('chat_history', [])
    bot_response = None

    if request.method == "POST":
        if "clear_chat" in request.POST:
            chat_history = []
            request.session['chat_history'] = []
            bot_response = "Chat history cleared."
        else:
            user_message = request.POST.get("user_input", "").strip()
            if user_message:
                bot_response = get_llama_response(user_message, chat_history)
                chat_history.append({'user': user_message, 'bot': bot_response})
                request.session['chat_history'] = chat_history
            else:
                bot_response = "Please type something!"

        return render(request, 'home.html', {
            'chat_history': chat_history,
            'bot_response': bot_response,
        })

    return render(request, 'home.html', {
        'chat_history': chat_history,
        'bot_response': bot_response,
    })

@login_required
def chat_view(request):
    threads = ChatThread.objects.filter(user=request.user)
    selected_thread_id = request.GET.get("thread")
    selected_thread = ChatThread.objects.filter(id=selected_thread_id, user=request.user).first() if selected_thread_id else None
    messages = selected_thread.messages.all() if selected_thread else []

    if request.method == "POST" and "new_thread" in request.POST:
        new_thread = ChatThread.objects.create(user=request.user, title=f"Chat {threads.count() + 1}")
        return redirect(f"/chat?thread={new_thread.id}")

    if request.method == "POST" and "delete_thread" in request.POST and selected_thread:
        selected_thread.delete()
        return redirect("chat")

    if request.method == "POST" and selected_thread and request.POST.get("message"):
        user_msg = request.POST.get("message")
        Message.objects.create(thread=selected_thread, sender="user", content=user_msg)
        chat_history = [
            {"user": m.content, "bot": "" if m.sender == "user" else m.content}
            for m in selected_thread.messages.all()
        ]
        bot_msg = get_llama_response(user_msg, chat_history)
        Message.objects.create(thread=selected_thread, sender="bot", content=bot_msg)
        return redirect(f"/chat?thread={selected_thread.id}")

    return render(request, "chat.html", {
        "threads": threads,
        "messages": messages,
        "selected_thread": selected_thread
    })

@login_required
def upload_document(request):
    message = ""
    if request.method == "POST" and request.FILES.get("document"):
        doc = Document(
            user=request.user,
            file=request.FILES["document"],
            title=request.FILES["document"].name
        )
        doc.save()
        message = "File uploaded successfully."
        return redirect("documents")

    docs = Document.objects.filter(user=request.user)
    return render(request, "documents.html", {
        "documents": docs,
        "message": message,
    })

@login_required
def download_document(request, doc_id):
    doc = get_object_or_404(Document, id=doc_id, user=request.user)
    response = HttpResponse(doc.file, content_type='application/octet-stream')
    response['Content-Disposition'] = f'attachment; filename="{doc.title}"'
    return response

@login_required
def delete_all_chats(request):
    if request.method == "POST":
        ChatThread.objects.filter(user=request.user).delete()
        Message.objects.filter(thread__user=request.user).delete()
        return JsonResponse({"status": "success", "message": "All chats deleted."})
    return JsonResponse({"error": "Invalid request"}, status=400)

@login_required
def ask_doc_question(request):
    if request.method == "POST":
        doc_id = request.POST.get("doc_id")
        question = request.POST.get("question")
        if not doc_id:
            return JsonResponse({"error": "doc_id not provided"})
        try:
            document = Document.objects.get(id=doc_id, user=request.user)
            doc_path = document.file.path
            ext = os.path.splitext(doc_path)[-1].lower()
            filetype = 'pdf' if ext == '.pdf' else 'txt'
            retriever = prepare_doc_retriever(doc_path, filetype=filetype)
            llm = get_llama_model()
            qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
            answer = qa.run(question)
        except Document.DoesNotExist:
            answer = "Document not found."
        except Exception as e:
            answer = f"Bot error: {str(e)}"
        return JsonResponse({"answer": answer})
    return JsonResponse({"error": "POST required"})
