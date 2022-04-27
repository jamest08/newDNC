# based on serial_iterator

from __future__ import division

import numpy

from chainer.dataset import iterator
from chainer.iterators.order_samplers import ShuffleOrderSampler


class OnTheFlyIterator(iterator.Iterator):

    """Dataset iterator that serially reads the examples.

    This is a simple implementation of :class:`~chainer.dataset.Iterator`
    that just visits each example in either the order of indexes or a shuffled
    order.

    To avoid unintentional performance degradation, the ``shuffle`` option is
    set to ``True`` by default. For validation, it is better to set it to
    ``False`` when the underlying dataset supports fast slicing. If the
    order of examples has an important meaning and the updater depends on the
    original order, this option should be set to ``False``.

    This iterator saves ``-1`` instead of ``None`` in snapshots since some
    serializers do not support ``None``.

    Args:
        dataset: Dataset to iterate.
        batch_size (int): Number of examples within each batch.
        repeat (bool): If ``True``, it infinitely loops over the dataset.
            Otherwise, it stops iteration at the end of the first epoch.
        shuffle (bool): If ``True``, the order of examples is shuffled at the
            beginning of each epoch. Otherwise, examples are extracted in the
            order of indexes. If ``None`` and no ``order_sampler`` is given,
            the behavior is the same as the case with ``shuffle=True``.
        order_sampler (callable): A callable that generates the order
            of the indices to sample in the next epoch when a epoch finishes.
            This function should take two arguements: the current order
            and the current position of the iterator.
            This should return the next order. The size of the order
            should remain constant.
            This option cannot be used when ``shuffle`` is not ``None``.

    """

    def __init__(self, generator, batch_size,
                 repeat=True, shuffle=None, order_sampler=None):
        self.generator = generator
        self.batch_size = batch_size
        self._repeat = repeat
        self._shuffle = shuffle
        self.epoch_size = 2940  # ie 2940 mini-batches per epoch. should make this an arg in train_transformer.yaml

        if self._shuffle is not None:
            if order_sampler is not None:
                raise ValueError('`shuffle` is not `None` and a custom '
                                 '`order_sampler` is set. Please set '
                                 '`shuffle` to `None` to use the custom '
                                 'order sampler.')
            else:
                if self._shuffle:
                    order_sampler = ShuffleOrderSampler()
        else:
            if order_sampler is None:
                order_sampler = ShuffleOrderSampler()
        self.order_sampler = order_sampler

        self.reset()

    def __next__(self):
        # if not self._repeat and self.epoch > 0:
        #     raise StopIteration

        self._previous_epoch_detail = self.epoch_detail

        batch = next(self.generator)

        self.mini_batch += 1
        if self.mini_batch == self.epoch_size:
            self.epoch += 1
            self.is_new_epoch = True
            self.mini_batch = 0
        else:
            self.is_new_epoch = False

        return batch

    next = __next__

    @property
    def epoch_detail(self):
        return float(self.epoch + self.mini_batch/self.epoch_size)

    @property
    def previous_epoch_detail(self):
        if self._previous_epoch_detail < 0:
            return None
        return self._previous_epoch_detail

    # IMPORTANT: SERIALIZE FUNCTION REMOVED

    def reset(self):
        self.epoch = 0
        self.is_new_epoch = False

        self.mini_batch = 0 # number of batches in the epoch so far 

        # use -1 instead of None internally.
        self._previous_epoch_detail = -1.

    # @property
    # def _epoch_size(self):
    #     return self.num_batches

    @property
    def repeat(self):
        return self._repeat
