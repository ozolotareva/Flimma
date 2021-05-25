"""
    A widget to add Flimma specific project info

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

from hyfed_client.widget.hyfed_project_info_widget import HyFedProjectInfoWidget
from hyfed_client.util.gui import add_label_and_textbox
from flimma_client.util.flimma_parameters import FlimmaProjectParameter


class FlimmaProjectInfoWidget(HyFedProjectInfoWidget):
    def __init__(self, title, connection_parameters, authentication_parameters):

        super().__init__(title=title, connection_parameters=connection_parameters,
                         authentication_parameters=authentication_parameters)

    # Flimma project specific info
    def add_flimma_project_info(self):
        add_label_and_textbox(self, label_text="Normalization",
                              value=self.project_parameters[FlimmaProjectParameter.NORMALIZATION], status='disabled')
        add_label_and_textbox(self, label_text="Min count",
                              value=self.project_parameters[FlimmaProjectParameter.MIN_COUNT], status='disabled')
        add_label_and_textbox(self, label_text="Min total count",
                              value=self.project_parameters[FlimmaProjectParameter.MIN_TOTAL_COUNT], status='disabled')
        add_label_and_textbox(self, label_text="Group1",
                              value=self.project_parameters[FlimmaProjectParameter.GROUP1], status='disabled')
        add_label_and_textbox(self, label_text="Group2",
                              value=self.project_parameters[FlimmaProjectParameter.GROUP2], status='disabled')
        add_label_and_textbox(self, label_text="Confounders",
                              value=self.project_parameters[FlimmaProjectParameter.CONFOUNDERS], status='disabled')
