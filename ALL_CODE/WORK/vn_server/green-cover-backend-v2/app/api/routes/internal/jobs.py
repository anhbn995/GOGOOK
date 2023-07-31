from app.controllers.internals import jc

def internal_job_routes_register(app):
    @app.route('/internals/jobs/kill', methods=['POST'])
    def internal_job_kill():
        return jc.kill()

    @app.route('/internals/jobs/delete', methods=['POST'])
    def internal_job_delete():
        return jc.delete()

    @app.route('/internals/jobs/requeue', methods=['POST'])
    def internal_job_requeue():
        return jc.requeue()

