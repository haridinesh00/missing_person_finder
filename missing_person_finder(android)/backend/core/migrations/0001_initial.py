# Generated by Django 3.2.12 on 2022-03-22 13:02

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Register',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('firstname', models.CharField(max_length=20)),
                ('lastname', models.CharField(max_length=20)),
                ('place', models.CharField(max_length=20)),
                ('description', models.CharField(max_length=60)),
                ('images', models.ImageField(upload_to='media')),
            ],
        ),
    ]