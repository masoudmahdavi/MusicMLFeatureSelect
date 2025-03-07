
class Metadata():
    def __init__(self, ):
        pass

class Model:
    metadata = Metadata()

    def __init__(self, 
                 data_path:str = None
                 ):
        
        self.data_path = data_path
