from app.controllers.internals import fs


def internal_fs_routes_register(app):
    @app.route('/internals/fs/directory/make', methods=['POST'])
    def fs_mkdirs():
        return fs.mkdirs()

    @app.route('/internals/fs/directory/size', methods=['POST'])
    def fs_size_dir():
        return fs.size_dir()

    @app.route('/pyapi/internals/fs/directory/delete', methods=['DELETE'])
    def fs_delete_dir():
        return fs.delete_dir()

    @app.route('/pyapi/internals/fs/directories/delete', methods=['DELETE'])
    def fs_delete_dirs():
        return fs.delete_dirs()

    @app.route('/internals/fs/file/copy', methods=['POST'])
    def fs_copyfile():
        return fs.copy_file()

    @app.route('/internals/fs/files/copy', methods=['POST'])
    def fs_copyfiles():
        return fs.copy_files()

    @app.route('/internals/fs/files/store', methods=['POST'])
    def fs_store_file():
        return fs.store_file()

    @app.route('/internals/fs/files/store_shp', methods=['POST'])
    def fs_store_file_shp():
        return fs.store_file_shp()


    @app.route('/pyapi/internals/fs/file/delete', methods=['DELETE'])
    def fs_deletefile():
        return fs.delete_file()

    