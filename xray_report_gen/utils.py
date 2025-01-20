import os
import json

PATH_DOCKER = "/xray_report_gen"
PATH_HOST = "/home/freinque/pycharm_projects/xray_report_gen"

def set_api_keys(path_docker=PATH_DOCKER, path_host=PATH_HOST):
    """
    set api keys environment variables
    
    :param path_docker: 
    :param path_host: 
    :return: 
    """
    try:
        with open(os.path.join(path_docker, "hf_token.txt"), 'r') as file:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = file.read().strip()
        print('loaded api key for huggingface')
    except:
        with open(os.path.join(path_host, "hf_token.txt"), 'r') as file:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = file.read().strip()
        print('loaded api key for huggingface')
        pass
    else:
        print('unable to load api key for huggingface')

    try:
        with open(os.path.join(path_docker, "oa_token.txt"), 'r') as file:
            os.environ["OPENAI_API_KEY"] = file.read().strip()
        print('loaded api key for openai')
    except:
        with open(os.path.join(path_host, "oa_token.txt"), 'r') as file:
            os.environ["OPENAI_API_KEY"] = file.read().strip()
        print('loaded api key for openai')
        pass
    else:
        print('unable to load keys from home testProject')


def parse_annotations(df, col, regions):
    for region in regions:
        df[col+'_'+region] = df[col].apply(lambda x: json.loads(x.replace('\'', '\"'))[region])
    return df
