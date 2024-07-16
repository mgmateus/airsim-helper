

class dict_object:
    def __init__(self, _dict):
        if isinstance(_dict, dict):
            self.__dict__.update(_dict) 
        else:
            n_dict = {}
            for key in _dict:
                n_dict[key] = None
            self.__dict__.update(n_dict) 

    def __str__(self):
        return str(self.__dict__)
    
    def __setitem__(self, key : str, value):
        self.__dict__[key] = value

    def keys(self, as_list : bool = False):
        return list(self.__dict__.keys()) if as_list else self.__dict__.keys()
    
    def items(self):
        return self.__dict__.items()
    
    def update(self, other : dict):
        if isinstance(other, dict):
            for key, cnt in other.items():
                self.__dict__[key] = cnt

        return self.__dict__