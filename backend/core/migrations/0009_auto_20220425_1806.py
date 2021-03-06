# Generated by Django 3.2.12 on 2022-04-25 12:36

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0008_auto_20220328_1815'),
    ]

    operations = [
        migrations.AddField(
            model_name='register',
            name='country',
            field=models.CharField(default=django.utils.timezone.now, max_length=20),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='register',
            name='description',
            field=models.CharField(default=django.utils.timezone.now, max_length=100),
            preserve_default=False,
        ),
    ]
