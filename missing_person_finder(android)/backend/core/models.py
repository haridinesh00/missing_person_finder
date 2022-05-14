
from django.db import models


class Register(models.Model):
    firstname = models.CharField(max_length=20)
    lastname = models.CharField(max_length=10)
    image = models.FileField(upload_to='media')
# Create your models here.
