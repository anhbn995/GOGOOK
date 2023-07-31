from app.database import db
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import uuid

from config.default import ROOT_DATA_FOLDER


class Imagery(db.Model):
    """ Imagery model """
    fillable = ['workspace_id', 'name', 'meta', 'acquired', 'src']

    __tablename__ = "images"
    id = db.Column(UUID(as_uuid=True), primary_key=True)
    created_by = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    owner_id = db.Column(db.Integer, nullable=False)
    workspace_id = db.Column(UUID(as_uuid=True), db.ForeignKey('workspaces.id'), nullable=False)
    name = db.Column(db.String(255), nullable=False)
    src = db.Column(db.String(255), nullable=True)
    thumbnail = db.Column(db.String(255), nullable=True)
    acquired = db.Column(db.DateTime, nullable=True)
    path = db.Column(db.String(500), nullable=False)
    meta = db.Column(db.JSON, nullable=False)
    tile_url = db.Column(db.String, nullable=False)
    created_at = db.Column(db.DateTime, server_default=func.now())
    updated_at = db.Column(db.DateTime, server_default=func.now(), onupdate=func.now())
    task_id = db.Column(db.Integer, nullable=True)
    folder_id = db.Column(db.Integer, nullable=True)

    @property
    def file_path(self):
        return image_2_file_path(self.created_by, self.workspace_id, self.id)

    def transform(self):
        img_size = 0
        try:
            import os
            img_size = os.path.getsize(self.file_path) / 1024 ** 2
        except Exception as e:
            print(e)
        owner = None
        if self.owner:
            owner = self.owner.transform()
        meta = self.meta
        result = {
            "id": "{}".format(self.id),
            "workspace_id": "{}".format(self.workspace_id),
            "name": "{}".format(self.name),
            "acquired": None if not self.acquired else self.acquired.strftime('%Y/%m/%d'),
            "created_at": self.created_at.strftime('%Y/%m/%d'),
            "updated_at": self.updated_at.strftime('%Y/%m/%d'),
            "size_mb": img_size,
            "images_count": 2,
            "tile_url": self.tile_url,
            "thumbnail": "/workspaces/{}/imageries/{}/thumbnail".format(self.workspace_id, self.id),
            "task_id" : self.task_id,
            "created_by": owner,
            "image_type": self.src
        }
        if self.image_type and self.meta:
            meta_bands = self.meta['BANDS']
            for idx, bands in enumerate(self.image_type.bands):
                try:
                    meta_bands[idx]['DESCRIPTION'] = bands['description']
                except Exception as e:
                    pass
            meta['BANDS'] = meta_bands
        result['meta'] = meta
        return result

def image_2_file_path(user_id, workspace_id, image_id):
    return '{}/data/{}/{}/{}.tif'.format(ROOT_DATA_FOLDER,
                                            user_id,
                                            uuid.UUID(str(workspace_id)).hex,
                                            uuid.UUID(str(image_id)).hex)


class UserImage(db.Model):

    __tablename__ = "user_image"

    id = db.Column(db.Integer, primary_key=True)
    image_id = db.Column(UUID(as_uuid=True), nullable=False)
    user_id = db.Column(db.Integer, nullable=False)
    owner = db.Column(db.Boolean, nullable=True)

    def transformed_image(self):
        transformed_image = self.image.transform()
        transformed_image['owner'] = self.owner
        transformed_image['deletable'] = False
        return transformed_image
