
class Metadata():
    def __init__(self, ):
        pass

class Model:
    metadata = Metadata()

    def __init__(self, 
                 data_path:str = None,
                 log_file_dir:str = None,
                 log_file_name:str = None,
                 ):
        
        self.data_path = data_path
        self.log_file_dir = log_file_dir
        self.log_file_name = log_file_name
