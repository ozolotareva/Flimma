"""
    Flimma client GUI

    Copyright 2021 Olga Zolotareva, Reza NasiriGerdeh, and Mohammad Bakhtiari. All Rights Reserved.

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

from hyfed_client.widget.join_widget import JoinWidget
from hyfed_client.widget.hyfed_project_status_widget import HyFedProjectStatusWidget
from hyfed_client.util.hyfed_parameters import HyFedProjectParameter, ConnectionParameter, AuthenticationParameter

from flimma_client.widget.flimma_project_info_widget import FlimmaProjectInfoWidget
from flimma_client.widget.flimma_dataset_widget import FlimmaDatasetWidget
from flimma_client.project.flimma_client_project import FlimmaClientProject
from flimma_client.util.flimma_parameters import FlimmaProjectParameter

import threading

import logging
logger = logging.getLogger(__name__)


class FlimmaClientGUI:
    """ Flimma Client GUI """

    def __init__(self):

        # create the join widget
        self.join_widget = JoinWidget(title="Flimma Client",
                                      local_server_name="Localhost",
                                      local_server_url="http://localhost:8000",
                                      local_compensator_name="Localhost",
                                      local_compensator_url="http://localhost:8001",
                                      external_server_name="Flimma-Server",
                                      external_server_url="https://exbio.wzw.tum.de/flimma/api",
                                      external_compensator_name="HyFed-SDU",
                                      external_compensator_url="https://compensator.compbio.sdu.dk",)

        # show the join widget
        self.join_widget.show()

        # if join was NOT successful, terminate the client GUI
        if not self.join_widget.is_joined():
            return

        # if join was successful, get connection and authentication parameters from the join widget
        connection_parameters = self.join_widget.get_connection_parameters()
        authentication_parameters = self.join_widget.get_authentication_parameters()

        #  create Flimma project info widget based on the authentication and connection parameters
        self.flimma_project_info_widget = FlimmaProjectInfoWidget(title="Flimma Project Info",
                                                                   connection_parameters=connection_parameters,
                                                                   authentication_parameters=authentication_parameters)

        # Obtain Flimma project info from the server
        # the project info will be set in project_parameters attribute of the info widget
        self.flimma_project_info_widget.obtain_project_info()

        # if Flimma project info cannot be obtained from the server, exit the GUI
        if not self.flimma_project_info_widget.project_parameters:
            return

        # add basic info of the project such as project id, project name, description, and etc to the info widget
        self.flimma_project_info_widget.add_project_basic_info()

        # add Flimma specific project info to the widget
        self.flimma_project_info_widget.add_flimma_project_info()

        # add accept and decline buttons to the widget
        self.flimma_project_info_widget.add_accept_decline_buttons()

        # show project info widget
        self.flimma_project_info_widget.show()

        # if participant declined to proceed, exit the GUI
        if not self.flimma_project_info_widget.is_accepted():
            return

        # if user agreed to proceed, create and show the Flimma dataset selection widget
        self.flimma_dataset_widget = FlimmaDatasetWidget(title="Flimma Dataset Selection")
        self.flimma_dataset_widget.add_quit_run_buttons()
        self.flimma_dataset_widget.show()

        # if the participant didn't click on 'Run' button, terminate the client GUI
        if not self.flimma_dataset_widget.is_run_clicked():
            return

        # if participant clicked on 'Run', get all the parameters needed
        # to create the client project from the widgets
        connection_parameters = self.join_widget.get_connection_parameters()
        authentication_parameters = self.join_widget.get_authentication_parameters()
        project_parameters = self.flimma_project_info_widget.get_project_parameters()

        server_url = connection_parameters[ConnectionParameter.SERVER_URL]
        compensator_url = connection_parameters[ConnectionParameter.COMPENSATOR_URL]
        username = authentication_parameters[AuthenticationParameter.USERNAME]
        token = authentication_parameters[AuthenticationParameter.TOKEN]
        project_id = authentication_parameters[AuthenticationParameter.PROJECT_ID]

        algorithm = project_parameters[HyFedProjectParameter.ALGORITHM]
        project_name = project_parameters[HyFedProjectParameter.NAME]
        project_description = project_parameters[HyFedProjectParameter.DESCRIPTION]
        coordinator = project_parameters[HyFedProjectParameter.COORDINATOR]

        # Flimma specific parameters
        normalization = project_parameters[FlimmaProjectParameter.NORMALIZATION]
        min_count = project_parameters[FlimmaProjectParameter.MIN_COUNT]
        min_total_count = project_parameters[FlimmaProjectParameter.MIN_TOTAL_COUNT]
        group1 = project_parameters[FlimmaProjectParameter.GROUP1]
        group2 = project_parameters[FlimmaProjectParameter.GROUP2]
        confounders = project_parameters[FlimmaProjectParameter.CONFOUNDERS]

        # path to input dataset files
        flimma_counts_file_path = self.flimma_dataset_widget.get_counts_file_path()
        flimma_design_file_path = self.flimma_dataset_widget.get_design_file_path()

        # create Flimma client project
        flimma_client_project = FlimmaClientProject(username=username,
                                                     token=token,
                                                     server_url=server_url,
                                                     compensator_url=compensator_url,
                                                    project_id=project_id,
                                                     algorithm=algorithm,
                                                     name=project_name,
                                                     description=project_description,
                                                     coordinator=coordinator,
                                                     result_dir='./', #result_dir='./flimma_client/result',
                                                     log_dir='./', #log_dir='./flimma_client/log',
                                                     normalization = normalization,
                                                     min_count = min_count,
                                                     min_total_count = min_total_count,
                                                     group1 = group1,
                                                     group2 = group2,
                                                     confounders = confounders,
                                                     flimma_counts_file_path = flimma_counts_file_path,
                                                     flimma_design_file_path = flimma_design_file_path)

        # run Flimma client project as a thread
        flimma_project_thread = threading.Thread(target=flimma_client_project.run)
        flimma_project_thread.setDaemon(True)
        flimma_project_thread.start()

        # create and show Flimma project status widget
        flimma_project_status_widget = HyFedProjectStatusWidget(title="Flimma Project Status",
                                                            project=flimma_client_project)
        flimma_project_status_widget.add_static_labels()
        flimma_project_status_widget.add_progress_labels()
        flimma_project_status_widget.add_status_labels()

        flimma_project_status_widget.add_log_and_quit_buttons()
        flimma_project_status_widget.show()


client_gui = FlimmaClientGUI()
