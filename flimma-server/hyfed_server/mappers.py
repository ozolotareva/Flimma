"""
    Mapper to map an algorithm name to the corresponding server project, model, and serializer

    Copyright 2021 Reza NasiriGerdeh. All Rights Reserved.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

# HyFed project
from hyfed_server.project.hyfed_server_project import HyFedServerProject
from hyfed_server.model.hyfed_models import HyFedProjectModel
from hyfed_server.serializer.hyfed_serializers import HyFedProjectSerializer

# Flimma project
from flimma_server.project.flimma_server_project import FlimmaServerProject
from flimma_server.model.flimma_model import FlimmaProjectModel
from flimma_server.serializer.flimma_serializers import FlimmaProjectSerializer

# server_project, project_model, and project_serializer are mappers used in webapp_view
server_project = dict()
project_model = dict()
project_serializer = dict()

# HyFed project mapper values
hyfed_tool = 'HyFed'
server_project[hyfed_tool] = HyFedServerProject
project_model[hyfed_tool] = HyFedProjectModel
project_serializer[hyfed_tool] = HyFedProjectSerializer

# Flimma project mapper values
flimma_name = 'Flimma'
server_project[flimma_name] = FlimmaServerProject
project_model[flimma_name] = FlimmaProjectModel
project_serializer[flimma_name] = FlimmaProjectSerializer


