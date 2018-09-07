import time
class _SessionRunner(object):
    def __init__(self, sess, extra_fetches=()):
        self.sess = sess
        self.extra_fetches = extra_fetches

    def begin(self):
        self._start_time = time.time()
        self._step = -1

    def before_run(self):
        self._step += 1

    def after_run(self, extra_fetched_results):
        raise NotImplementedError

    def run(self, fetches,
            feed_dict=None,
            options=None,
            run_metadata=None):

        if not isinstance(fetches, list) and not isinstance(fetches, tuple):
            fetches = [fetches]

        if not hasattr(self, "_step"):
            self.begin()

        self.before_run()
        results = self.sess.run(fetches + self.extra_fetches,
                                feed_dict=feed_dict,
                                options=options,
                                run_metadata=run_metadata)

        self.after_run(results[len(fetches):])
        return results[:len(fetches)]

    def __call__(self, *args, **kwargs):
        self.run(*args, **kwargs)
