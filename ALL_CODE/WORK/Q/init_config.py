import os
import json
import inspect

from models import core
from models.core import *
from inspect import getmembers, isfunction

def get_model(model, dict_params = {}):
    args_model = inspect.getfullargspec(model).args
    defaults_model = inspect.getfullargspec(model).defaults
    if not dict_params:
        for n, i in enumerate(args_model):
            num_arg = len(args_model)
            num_def = len(defaults_model)
            if (n + num_def) > (num_arg-1):
                a = (n + num_def)-(num_arg-1) -1
                dict_params.update({i:defaults_model[a]})
            else:
                dict_params.update({i:None})

    print("Args of model:")
    print(dict_params)
    for i in dict_params:
        inputt = input("Enter %s:"%(i))
        if '' != inputt:
            if '[' and ']' in inputt:
                a = json.loads(inputt)
            else:
                if inputt.isnumeric():
                    a = int(inputt)
                else:
                    try:
                        a = float(inputt)
                    except ValueError:
                        a = inputt

            dict_params.update({i:a})
    return dict_params

def main(re_params=True):
    print("List avaiable model:")
    list_model = getmembers(core, isfunction)
    for i in list_model:
        print(i[0])
    
    use_model = input("Enter name model:")
    config_path = os.path.join('.', 'configs', '%s.json'%(use_model))
    model = eval(use_model)

    if not os.path.exists(config_path):
        print("Config file isn't exists")
        dict_params = get_model(model)
        with open(config_path, 'w') as outfile:
                json.dump(dict_params, outfile, indent = 4)
    else:
        if re_params:
            dict_params = json.load(open(config_path))
            dict_params = get_model(model, dict_params)
            with open(config_path, 'w') as outfile:
                json.dump(dict_params, outfile, indent = 4)
        else:
            dict_params = json.load(open(config_path))

    print("Init model:")
    print(dict_params)
    init_model = model(**dict_params)
    init_model.summary()

if __name__=="__main__":
    main(re_params=False)