from threading import Thread


class ImageDiscoveryThread(Thread):
    def __init__(self, **kwargs):
        super().__init__()
        self._list = kwargs.get('list_sub_futures')  # list of subgeometry
        self._list_lock = kwargs.get('list_lock')  # Lock for list_sub_futures
        self._out_lock = kwargs.get('out_lock')  # Lock for output
        self._output = kwargs.get('output')  # list of output
        print("vao day")

    def run(self):
        while True:
            # request to access shared resource
            # if there are many threads acquiring Lock, only one thread get the Lock
            # and other threads will get blocked
            self._list_lock.acquire()
            try:
                item = next(self._list)  # pop a number in list_sub_futures
                print(item)
            except StopIteration:
                return
            finally:
                # release the Lock, so other thread can gain the Lock to access list_sub_futures
                self._list_lock.release()

            result = self.send_discovery_api(item)
            if not result:
                pass

            try:
                self._out_lock.acquire()
                if isinstance(self._output, dict):
                    self._output.update(result)
                else:
                    self._output.append(result)
            except Exception as e:
                print('error', e)
                return
            finally:
                self._out_lock.release()

    def send_discovery_api(self, item):
        pass
