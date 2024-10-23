# import pytest
# from my_project.db import get_db_connection
# from my_project.config import MySQLSettings


# @pytest.fixture(scope="session")
# def db_connection():
#     settings = MySQLSettings()
#     connection = get_db_connection(settings)
#     yield connection
#     connection.close()

import os
import pytest


# @pytest.fixture(scope="session", autouse=True)
# def set_env_vars():
#     os.environ["MODE"] = "test"
