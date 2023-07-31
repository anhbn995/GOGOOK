from app.database import db
from sqlalchemy.dialects.postgresql import UUID
import uuid
from sqlalchemy.sql import func
from config.default import ROOT_DATA_FOLDER


class Result(db.Model):
    """ Result model """

    __tablename__ = "results"
    id = db.Column(UUID(as_uuid=True), primary_key=True)
    task_id = db.Column(db.Integer, nullable=True)
    path = db.Column(db.String(255), nullable=False)
    images = db.Column(db.JSON, nullable=False)
    model_id = db.Column(UUID(as_uuid=True), nullable=True)
    file_name = db.Column(db.String(255), nullable=False)
    format = db.Column(db.String(255), nullable=False, default='geojson')
    meta = db.Column(db.JSON, nullable=False)
    workspace_id = db.Column(UUID(as_uuid=True), db.ForeignKey('workspaces.id'), nullable=False)
    ref_id = db.Column(UUID(as_uuid=True), nullable=True)
    created_at = db.Column(db.DateTime, server_default=func.now())
    updated_at = db.Column(db.DateTime, server_default=func.now(), onupdate=func.now())

    def file_path(self, user_id, type='geojson'):
        result_dir = '{}/data/{}/{}/result'.format(ROOT_DATA_FOLDER, user_id, uuid.UUID(str(self.workspace_id)).hex)
        file = '{}/{}.{}'.format(result_dir, uuid.UUID(str(self.id)).hex, type)
        return file

    def transform(self, user_id):
        transformed = {
            "id": "{}".format(self.id),
            "hex_id": uuid.UUID(str(self.id)).hex,
            "task_id": "{}".format(self.task_id),
            "workspace_id": "{}".format(self.workspace_id),
            "path": self.path,
            "images": self.images,
            "file_name": self.file_name,
            "format": self.format,
            "meta": self.meta,
            "created_at": "{}".format(self.created_at),
            "updated_at": "{}".format(self.updated_at),
        }
        if self.format == 'tif':
            transformed["tile_url"] = '/result_tile/{}/{}/{}'.format(user_id, uuid.UUID(str(self.workspace_id)).hex,
                                           uuid.UUID(str(self.id)).hex) + '/{z}/{x}/{y}'
        if self.ref_id:
            transformed["ref_id"] = self.ref_id
        return transformed