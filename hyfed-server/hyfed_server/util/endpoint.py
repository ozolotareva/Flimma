"""
    URL endpoints used by the clients to communicate with the server

    Copyright 2021 Reza NasiriGerdeh and Reihaneh TorkzadehMahani. All Rights Reserved.

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


class EndPoint:
    PROJECT_JOIN = 'client/hyfed/project-join/'
    PROJECT_INFO = 'client/hyfed/project-info/'
    PROJECT_STARTED = 'client/hyfed/project-started/'
    MODEL_AGGREGATION = 'client/hyfed/model-aggregation/'
    GLOBAL_MODEL = 'client/hyfed/global-model/'
    RESULT_DOWNLOAD = 'client/hyfed/result-download/'
