"""
    Model class for Flimma project specific fields

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

from django.db import models
from hyfed_server.model.hyfed_models import HyFedProjectModel


class FlimmaProjectModel(HyFedProjectModel):
    """
        The model inherits from HyFedProjectModel
        so it implicitly has id, name, status, etc, fields defined in the parent model
    """

    normalization = models.CharField(max_length=512, default="UQ")
    min_count = models.PositiveIntegerField(default=10)
    min_total_count = models.PositiveIntegerField(default=15)
    group1 = models.CharField(max_length=512, default="")
    group2 = models.CharField(max_length=512, default="")
    confounders = models.CharField(max_length=512, default="")
