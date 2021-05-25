# Generated by Django 3.2 on 2021-05-15 20:46

from django.conf import settings
import django.contrib.auth.models
import django.contrib.auth.validators
from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone
import uuid


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('auth', '0012_alter_user_first_name_max_length'),
    ]

    operations = [
        migrations.CreateModel(
            name='UserModel',
            fields=[
                ('password', models.CharField(max_length=128, verbose_name='password')),
                ('last_login', models.DateTimeField(blank=True, null=True, verbose_name='last login')),
                ('is_superuser', models.BooleanField(default=False, help_text='Designates that this user has all permissions without explicitly assigning them.', verbose_name='superuser status')),
                ('username', models.CharField(error_messages={'unique': 'A user with that username already exists.'}, help_text='Required. 150 characters or fewer. Letters, digits and @/./+/-/_ only.', max_length=150, unique=True, validators=[django.contrib.auth.validators.UnicodeUsernameValidator()], verbose_name='username')),
                ('first_name', models.CharField(blank=True, max_length=150, verbose_name='first name')),
                ('last_name', models.CharField(blank=True, max_length=150, verbose_name='last name')),
                ('email', models.EmailField(blank=True, max_length=254, verbose_name='email address')),
                ('is_staff', models.BooleanField(default=False, help_text='Designates whether the user can log into this admin site.', verbose_name='staff status')),
                ('is_active', models.BooleanField(default=True, help_text='Designates whether this user should be treated as active. Unselect this instead of deleting accounts.', verbose_name='active')),
                ('date_joined', models.DateTimeField(default=django.utils.timezone.now, verbose_name='date joined')),
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('groups', models.ManyToManyField(blank=True, help_text='The groups this user belongs to. A user will get all permissions granted to each of their groups.', related_name='user_set', related_query_name='user', to='auth.Group', verbose_name='groups')),
                ('user_permissions', models.ManyToManyField(blank=True, help_text='Specific permissions for this user.', related_name='user_set', related_query_name='user', to='auth.Permission', verbose_name='user permissions')),
            ],
            options={
                'verbose_name': 'user',
                'verbose_name_plural': 'users',
                'abstract': False,
            },
            managers=[
                ('objects', django.contrib.auth.models.UserManager()),
            ],
        ),
        migrations.CreateModel(
            name='TimerModel',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('client_computation', models.FloatField(default=0.0)),
                ('client_network_send', models.FloatField(default=0.0)),
                ('client_network_receive', models.FloatField(default=0.0)),
                ('client_idle', models.FloatField(default=0.0)),
                ('compensator_computation', models.FloatField(default=0.0)),
                ('compensator_network_send', models.FloatField(default=0.0)),
                ('server_computation', models.FloatField(default=0.0)),
                ('runtime_total', models.FloatField(default=0.0)),
            ],
        ),
        migrations.CreateModel(
            name='TrafficModel',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('client_server', models.CharField(default='0.00 KB', max_length=32)),
                ('server_client', models.CharField(default='0.00 KB', max_length=32)),
                ('client_compensator', models.CharField(default='0.00 KB', max_length=32)),
                ('compensator_server', models.CharField(default='0.00 KB', max_length=32)),
                ('traffic_total', models.CharField(default='0.00 KB', max_length=32)),
            ],
        ),
        migrations.CreateModel(
            name='HyFedProjectModel',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('tool', models.CharField(default='', max_length=255)),
                ('algorithm', models.CharField(default='', max_length=255)),
                ('name', models.CharField(default='', max_length=255)),
                ('description', models.CharField(default='', max_length=255)),
                ('status', models.CharField(choices=[('Created', 'Created'), ('Parameters Ready', 'Parameters Ready'), ('Aggregating', 'Aggregating'), ('Done', 'Done'), ('Aborted', 'Aborted'), ('Failed', 'Failed')], default='Created', max_length=31)),
                ('step', models.CharField(default='Init', max_length=255)),
                ('comm_round', models.PositiveIntegerField(default=1)),
                ('result_dir', models.CharField(default='', max_length=1000)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('coordinator', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
                ('timer', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='hyfed_server.timermodel')),
                ('traffic', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='hyfed_server.trafficmodel')),
            ],
        ),
        migrations.CreateModel(
            name='TokenModel',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('participant', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='projects', to=settings.AUTH_USER_MODEL)),
                ('project', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='participants', to='hyfed_server.hyfedprojectmodel')),
            ],
            options={
                'unique_together': {('project', 'participant')},
            },
        ),
    ]
