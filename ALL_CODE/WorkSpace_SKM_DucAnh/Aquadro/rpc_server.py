from lib.rpc import RPCServer
from lib.rpc.routes import routing
import lib.rpc.routes.raster
import lib.rpc.routes.task
import lib.rpc.routes.job
import lib.rpc.routes.vector

RPCServer(routing).consume()
