import os
from importlib import  import_module
from importlib.util import find_spec

def get_model(model_name):
    '''
    Given the model name , back the model class and model parameters
    @param model_name:
    @return:
    '''
    model_file_name = model_name.upper()

    # model_path = os.getcwd() + '/model/' + model_file_name + '.' + model_file_name
    model_path = '.'.join(['model', model_file_name])
    model_parameter_path = '.'.join(['parameters', model_file_name])

    if  find_spec(model_path,__name__) and find_spec(model_parameter_path,__name__):
        return getattr(import_module(model_path,__name__),model_file_name),\
               getattr(import_module(model_parameter_path,__name__),model_file_name)
    else:
        raise Exception(f'{model_name} is not exist in {model_path}, please check')

def get_lossF(lossname):
    loss_name = lossname.name.upper()
    path = '.'.join(['lossF','lossF'])

    if  find_spec(path,__name__):
        return getattr(import_module(path,__name__),loss_name)
    else:
        raise Exception(f'{path} is not exist in {loss_name}, please check')