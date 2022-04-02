from django.urls import path
from . import views

urlpatterns = [
    path('register/', views.PostView.as_view(), name= 'posts_list'),
]