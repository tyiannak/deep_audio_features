import requests
import os


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)


def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

    return None


def download_missing(modification):
    """
    Checks if CNN models exist. If not, the script
    downloads them from Google Drive.

    Parameters
    ----------
    modification: dictionary containing at least the following variables:
        - model_paths (string): paths of the discussed models
        - download_models (boolean):
            if true the missing models will be downloaded
        - google_drive_ids (list of strings):
            list containing the ids of the google drive files

    """
    which_models = []
    for idx, path in enumerate(modification['model_paths']):
        if os.path.exists(path):
            continue
        else:
            which_models.append((idx, path))
    if not which_models:
        print('All CNN models already exist (No need to download).')
    else:
        print('Downloading models...')
        if modification['download_models']:
            for idx, dest in which_models:
                file_id = modification['google_drive_ids'][idx]
                download_file_from_google_drive(file_id, dest)

