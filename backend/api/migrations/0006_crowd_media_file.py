# Generated by Django 3.2.14 on 2022-07-23 06:27

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('api', '0005_incident_camera_id'),
    ]

    operations = [
        migrations.AddField(
            model_name='crowd',
            name='media_file',
            field=models.FileField(default='default.png', upload_to='uploads/'),
        ),
    ]