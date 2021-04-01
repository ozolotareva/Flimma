"""
    A class to indicate the current operation of the client

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


class ClientOperation:
    WAITING_FOR_START = "Waiting for Project Start"
    WAITING_FOR_AGGREGATION = "Waiting for Aggregation"
    SENDING_LOCAL_PARAMETERS = "Sending Parameters"
    COMPUTING_LOCAL_PARAMETERS = "Computing Parameters"
    DOWNLOADING_RESULTS = "Downloading Results"
    FINISHING_UP = "Finishing up"
    ABORTED = "Aborted"
    DONE = "Done"
