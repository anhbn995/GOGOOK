import uuid
from config.default import ROOT_DATA_FOLDER


def get_workspace_dir(user_id, wks_id):
    return '{}/data/{}/{}'.format(ROOT_DATA_FOLDER, user_id, uuid.UUID(str(wks_id)).hex)
