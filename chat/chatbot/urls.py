from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_view, name='chat_home'),
    path("signup/", views.signup_view, name="signup"),
    path("login/", views.login_view, name="login"),
    path("logout/", views.logout_view, name="logout"),
    path("chat/", views.chat_view, name="chat"),
    path("delete_all_chats/", views.delete_all_chats, name="delete_all_chats"),
    path("upload/", views.upload_document, name="upload_document"),
    path("documents/", views.upload_document, name="documents"),
    path("documents/download/<int:doc_id>/", views.download_document, name="download_document"),
    path("ask-question/", views.ask_doc_question, name="ask_question"),
]
