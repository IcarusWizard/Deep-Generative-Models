class DatasetRegistry(object):
    """
        This data structure maintain a global registry for different datasets.
        Dataset should be store here as a pair of a name and a callable function.  
    """
    def __init__(self):
        self.dataset_dict = {}

    def add(self, name, fn):
        if name in self.dataset_dict.keys():
            print('Warning: overwriting dataset function for {}'.format(name))
        self.dataset_dict[name] = fn

    def make(self, name, **kwargs):
        fn = self.dataset_dict[name]
        return fn(**kwargs)

# create a global registry
data_registry = DatasetRegistry()

def add_dataset(name, fn_string):
    data_registry.add(name, fn_string)

def make_dataset(name, **kwargs):
    return data_registry.make(name, **kwargs)