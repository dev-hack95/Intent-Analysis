import sys
from logger import logging

def error_message_details(error, error_detail:sys):
    _, _, exec_db = error_detail.exc_info()
    file_name = exec_db.tb_frame.f_code.co_filename # Which filename probably that error come from
    error_message = "Error occured in python script name [{0}] line number [{1}] error message [{2}]".format(file_name, exec_db.tb_lineno, str(error))
    return error_message

class CustomException(Exception):

    def __init__(self, error_message, error_detail:sys) -> None:
        super().__init__(error_message) # Inharating error from parent class
        self.error_message = error_message_details(error_message, error_detail=error_detail)

    def __str__(self) -> str:
        return self.error_message
