# Generated by Django 3.2.12 on 2022-03-26 13:52

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0006_alter_register_image'),
    ]

    operations = [
        migrations.AlterField(
            model_name='register',
            name='image',
            field=models.FileField(upload_to='media'),
        ),
    ]
