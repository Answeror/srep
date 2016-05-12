from __future__ import division
import logging
from logbook import Logger
import mxnet as mx
import numpy as np
from functools import partial
from contextlib import contextmanager
from nose.tools import assert_equal, assert_is_instance
import re
import time
import os
from .symbol import get_symbol
from . import ROOT, constant, utils


logger = Logger('sigr')


class SymbolMixin(object):

    def __init__(self, **kargs):
        symbol = kargs.pop('symbol', None)
        symbol_kargs = kargs.pop('symbol_kargs', {}).copy()
        if symbol is None:
            symbol = get_symbol(**symbol_kargs)
        super(SymbolMixin, self).__init__(symbol=symbol, **kargs)


class EvalMixin(object):

    def __init__(self, **kargs):
        self.num_eval_epoch = kargs.pop('num_eval_epoch', 1)
        super(EvalMixin, self).__init__(**kargs)

    def fit(self, train_data, eval_data=None, eval_metric='acc',
            epoch_end_callback=None, batch_end_callback=None, kvstore='local',
            optimizer='sgd', optimizer_params=(('learning_rate', 0.01),),
            eval_batch_end_callback=None, initializer=mx.init.Uniform(0.01),
            arg_params=None, aux_params=None, allow_missing=False,
            force_rebind=False, force_init=False, begin_epoch=0, num_epoch=None,
            validation_metric=None, monitor=None):
        assert num_epoch is not None, 'please specify number of epochs'

        self.bind(data_shapes=train_data.provide_data, label_shapes=train_data.provide_label,
                  for_training=True, force_rebind=force_rebind)
        if monitor is not None:
            self.install_monitor(monitor)
        self.init_params(initializer=initializer, arg_params=arg_params, aux_params=aux_params,
                         allow_missing=allow_missing, force_init=force_init)
        self.init_optimizer(kvstore=kvstore, optimizer=optimizer,
                            optimizer_params=optimizer_params)

        if validation_metric is None:
            validation_metric = eval_metric
        if not isinstance(eval_metric, mx.metric.EvalMetric):
            eval_metric = mx.metric.create(eval_metric)

        ################################################################################
        # training loop
        ################################################################################
        for epoch in range(begin_epoch, num_epoch):
            tic = time.time()
            eval_metric.reset()
            for nbatch, data_batch in enumerate(train_data):
                if monitor is not None:
                    monitor.tic()
                self.forward_backward(data_batch)
                self.update()
                self.update_metric(eval_metric, data_batch.label)

                if monitor is not None:
                    monitor.toc_print()

                if batch_end_callback is not None:
                    batch_end_params = mx.model.BatchEndParam(epoch=epoch, nbatch=nbatch,
                                                              eval_metric=eval_metric,
                                                              locals=locals())
                    for callback in mx.module.base_module._as_list(batch_end_callback):
                        callback(batch_end_params)

            # one epoch of training is finished
            for name, val in eval_metric.get_name_value():
                self.logger.info('Epoch[%d] Train-%s=%f', epoch, name, val)
            toc = time.time()
            self.logger.info('Epoch[%d] Time cost=%.3f', epoch, (toc-tic))

            if epoch_end_callback is not None:
                arg_params, aux_params = self.get_params()
                for callback in mx.module.base_module._as_list(epoch_end_callback):
                    callback(epoch, self.symbol, arg_params, aux_params)

            # ----------------------------------------
            # evaluation on validation set
            if eval_data and (epoch + 1) % self.num_eval_epoch == 0:
                res = self.score(eval_data, validation_metric,
                                 batch_end_callback=eval_batch_end_callback, epoch=epoch)
                for name, val in res:
                    self.logger.info('Epoch[%d] Validation-%s=%f', epoch, name, val)

            # end of 1 epoch, reset the data-iter for another epoch
            train_data.reset()


class MxNetModule(mx.mod.Module):

    def __init__(self, **kargs):
        super(MxNetModule, self).__init__(
            **{k: kargs[k] for k in kargs
               if k in ['symbol', 'data_names', 'label_names', 'logger',
                        'context', 'work_load_list']})


class Base(EvalMixin, SymbolMixin, MxNetModule):

    @property
    def num_semg_row(self):
        return self.symbol.num_semg_row

    @property
    def num_semg_col(self):
        return self.symbol.num_semg_col

    @property
    def data_shape_1(self):
        return self.symbol.data_shape_1

    @property
    def num_channel(self):
        return self.symbol.num_channel

    @property
    def num_feature(self):
        return self.symbol.num_feature

    @property
    def presnet_proj_type(self):
        return self.symbol.presnet_proj_type

    def __init__(self, **kargs):
        parent = kargs.pop('parent', {})
        kargs['data_names'] = kargs.get('data_names', getattr(parent, '_data_names', ('data',)))
        kargs['label_names'] = kargs.get('label_names', getattr(parent, '_label_names', ('softmax_label',)))
        super(Base, self).__init__(**kargs)

    def init_params(self, initializer=mx.init.Uniform(0.01), arg_params=None, aux_params=None,
                    allow_missing=False, force_init=False):
        if self.params_initialized and not force_init:
            return
        assert self.binded, 'call bind before initializing the parameters'

        if self._arg_params is None:
            param_arrays = [mx.nd.zeros(x[0].shape) for x in self._exec_group.param_arrays]
            self._arg_params = {name: arr for name, arr in zip(self._param_names, param_arrays)}

        if self._aux_params is None:
            aux_arrays = [mx.nd.zeros(x[0].shape) for x in self._exec_group.aux_arrays]
            self._aux_params = {name: arr for name, arr in zip(self._aux_names, aux_arrays)}

        def _impl(name, arr, cache):
            """Internal helper for parameter initialization"""
            if cache is not None:
                if name in cache:
                    cache_arr = cache[name]

                    # just in case the cached array is just the target itself
                    if cache_arr is not arr:
                        assert cache_arr.shape == arr.shape, '{} {} {}'.format(name, cache_arr.shape, arr.shape)
                        cache_arr.copyto(arr)
                else:
                    assert allow_missing, name
                    initializer(name, arr)
            else:
                initializer(name, arr)

        for name, arr in self._arg_params.items():
            _impl(name, arr, arg_params)

        for name, arr in self._aux_params.items():
            _impl(name, arr, aux_params)

        self.params_initialized = True
        self._params_dirty = False

        # copy the initialized parameters to devices
        self._exec_group.set_params(self._arg_params, self._aux_params)


class PresnetExecutorGroup(mx.module.executor_group.DataParallelExecutorGroup):

    def __init__(self, *args, **kargs):
        self.num_presnet = kargs.pop('num_presnet')
        super(PresnetExecutorGroup, self).__init__(*args, **kargs)

    def _bind_ith_exec(self, i, data_shapes, label_shapes, shared_group):
        """Internal utility function to bind the i-th executor.
        """
        nd = mx.nd

        data_shapes = self._sliced_shape(data_shapes, i)
        if label_shapes is not None:
            label_shapes = self._sliced_shape(label_shapes, i)
        shared_exec = None if shared_group is None else shared_group.execs[i]
        context = self.contexts[i]
        shared_data_arrays = self.shared_data_arrays[i]

        input_shapes = dict(data_shapes)
        if label_shapes is not None:
            input_shapes.update(dict(label_shapes))

        shapes = {}
        shapes.update(input_shapes)
        arg_shapes, _, aux_shapes = self.symbol.infer_shape(**shapes)
        assert arg_shapes is not None, "shape inference failed"

        if self.input_types is None:
            input_types = {k: mx.base.mx_real_t for k in input_shapes.keys()}
        else:
            input_types = self.input_types
        types = {}
        for name in self.arg_names:
            if name.endswith('_zero') or name.endswith('_one'):
                types[name] = mx.base.mx_real_t
        types.update(input_types)
        arg_types, _, aux_types = self.symbol.infer_type(**types)
        assert arg_types is not None, "type inference failed"

        data_names = [x[0] for x in data_shapes]

        arg_arrays = []
        grad_arrays = {} if self.for_training else None
        grad_req = {}
        for name in self.arg_names:
            if self.for_training:
                if name in self.param_names:
                    grad_req[name] = 'write'
                elif name in data_names:
                    grad_req[name] = 'write' if self.inputs_need_grad else 'null'
                else:
                    grad_req[name] = 'null'
            else:
                grad_req[name] = 'null'

        def _get_or_reshape(name, shared_data_arrays, arg_shape, arg_type, context, logger):
            """Internal helper to get a memory block or re-use by re-shaping"""
            if name in shared_data_arrays:
                arg_arr = shared_data_arrays[name]

                if np.prod(arg_arr.shape) >= np.prod(arg_shape):
                    # nice, we can directly re-use this data blob
                    assert arg_arr.dtype == arg_type
                    arg_arr = arg_arr.reshape(arg_shape)
                else:
                    logger.warning(('bucketing: data "%s" has a shape %s' % (name, arg_shape)) +
                                   (', which is larger than already allocated ') +
                                   ('shape %s' % (arg_arr.shape,)) +
                                   ('. Need to re-allocate. Consider putting ') +
                                   ('default_bucket_key to') +
                                   (' be the bucket taking the largest input for better ') +
                                   ('memory sharing.'))
                    arg_arr = nd.zeros(arg_shape, context, dtype=arg_type)

                    # replace existing shared array because the new one is bigger
                    shared_data_arrays[name] = arg_arr
            else:
                arg_arr = nd.zeros(arg_shape, context, dtype=arg_type)
                shared_data_arrays[name] = arg_arr

            return arg_arr

        # create or borrow arguments and gradients
        for j in range(len(self.arg_names)):
            name = self.arg_names[j]
            if name in self.param_names:  # model parameter
                if shared_exec is None:
                    arg_arr = nd.zeros(arg_shapes[j], context, dtype=arg_types[j])
                    if grad_req[name] != 'null':
                        grad_arr = nd.zeros(arg_shapes[j], context, dtype=arg_types[j])
                        grad_arrays[name] = grad_arr
                else:
                    arg_arr = shared_exec.arg_dict[name]
                    assert arg_arr.shape == arg_shapes[j]
                    assert arg_arr.dtype == arg_types[j]
                    if grad_req[name] != 'null':
                        grad_arrays[name] = shared_exec.grad_dict[name]
            else:  # data or label
                arg_arr = _get_or_reshape(name, shared_data_arrays, arg_shapes[j], arg_types[j],
                                          context, self.logger)

                # data might also need grad if inputs_need_grad is True
                if grad_req[name] != 'null':
                    grad_arrays[name] = _get_or_reshape('grad of ' + name, shared_data_arrays,
                                                        arg_shapes[j], arg_types[j], context,
                                                        self.logger)

            arg_arrays.append(arg_arr)

        # create or borrow aux variables
        if shared_exec is None:
            aux_arrays = [nd.zeros(s, context, dtype=t) for s, t in zip(aux_shapes, aux_types)]
        else:
            for j, arr in enumerate(shared_exec.aux_arrays):
                assert aux_shapes[j] == arr.shape
                assert aux_types[j] == arr.dtype
            aux_arrays = shared_exec.aux_arrays[:]

        executor = self.symbol.bind(ctx=context, args=arg_arrays,
                                    args_grad=grad_arrays, aux_states=aux_arrays,
                                    grad_req=grad_req, shared_exec=shared_exec)
        return executor


class FaugMixin(object):

    def __init__(self, **kargs):
        parent = kargs.pop('parent', {})
        kargs['data_names'] = kargs.get('data_names', getattr(parent, '_data_names', ('data',)))
        self.faug = kargs.pop('faug', 0)
        self.faug_classwise = kargs.pop('faug_classwise', False)
        if self.faug:
            kargs['data_names'] = list(kargs['data_names']) + ['faug']
            self.data_index = len(kargs['data_names']) - 1
            symbol_kargs = kargs.pop('symbol_kargs', getattr(parent, 'symbol_args', {})).copy()
            symbol_kargs.update(faug=self.faug)
            kargs['symbol_kargs'] = symbol_kargs
        super(FaugMixin, self).__init__(**kargs)

    def _wrap_data(self, data, faug):
        if data is None:
            return data
        from .data import Preload, FaugData
        return Preload([
            data,
            FaugData(faug, data.batch_size, self.num_feature)
        ])

    def fit(self, **kargs):
        if self.faug:
            kargs['train_data'] = self._wrap_data(kargs.pop('train_data'), self.faug)
            kargs['eval_data'] = self._wrap_data(kargs.pop('eval_data'), 0)
        return super(FaugMixin, self).fit(**kargs)

    def forward_backward(self, data_batch):
        if self.faug and self.faug_classwise:
            data = data_batch.data[self.data_index]
            data = data.asnumpy()
            zero = mx.nd.zeros(data.shape)
            data_batch.data[self.data_index] = zero

            mx.module.executor_group._load_label(
                data_batch,
                self._exec_group.label_arrays
            )
            net = self.symbol.feature_before_faug
            exec_ = self._exec_group.execs[0]
            exec_ = net.bind(
                ctx=self._context[0],
                args=exec_.arg_dict,
                aux_states=exec_.aux_dict
            )
            feature = exec_.outputs[0].asnumpy()

            label = data_batch.label[0].asnumpy()
            if data_batch.pad:
                feature = feature[:-data_batch.pad]
                label = label[:-data_batch.pad]
            for i in range(self.num_gesture):
                mask = label == i
                scale = feature[mask].std(axis=0)
                scale[np.isnan(scale)] = 0
                scale[scale > 3] = 3  # Limit to 3 std
                data[mask] *= scale
            data_batch.data[self.data_index] = mx.nd.array(data)
        super(FaugMixin, self).forward_backward(data_batch)


class AdaBNMixin(object):

    def __init__(self, **kargs):
        parent = kargs.get('parent', {})
        self.downsample = kargs.pop('downsample', getattr(parent, 'downsample', False))
        self.adabn = kargs.pop('adabn', getattr(parent, 'adabn', False))
        self.num_adabn_epoch = kargs.pop('num_adabn_epoch',
                                         getattr(parent, 'num_adabn_epoch', constant.NUM_ADABN_EPOCH))
        super(AdaBNMixin, self).__init__(**kargs)

    @property
    def lstm_window(self):
        return self.symbol.lstm_window

    @contextmanager
    def _restore_eval_data(self, eval_data):
        #  random_scale = eval_data.random_scale
        #  eval_data.random_scale = False
        shuffle = eval_data.shuffle
        eval_data.shuffle = True
        downsample = eval_data.downsample
        eval_data.downsample = self.downsample
        last_batch_handle = eval_data.last_batch_handle
        eval_data.last_batch_handle = 'roll_over'
        try:
            yield
        finally:
            #  eval_data.random_scale = random_scale
            eval_data.shuffle = shuffle
            eval_data.downsample = downsample
            eval_data.last_batch_handle = last_batch_handle

    def _update_adabn(self, eval_data):
        '''Update moving mean and moving var with eval data'''
        from time import time
        start = time()
        with self._restore_eval_data(eval_data):
            for _ in range(self.num_adabn_epoch):
                eval_data.reset()
                for nbatch, eval_batch in enumerate(eval_data):
                    self.forward(eval_batch, is_train=True)
                    for out in self.get_outputs():
                        #  Cause memory leak (though not increase after this _update_adabn) without this wait
                        #  TODO: fixme
                        out.wait_to_read()
                    #  for name, block in zip(self._exec_group.aux_names, self._exec_group.aux_arrays):
                        #  if 'moving' in name:
                            #  for a in block:
                                #  a.wait_to_read()
        logger.debug(
            'AdaBN with {} epochs takes {} seconds',
            self.num_adabn_epoch,
            time() - start
        )

    def _try_update_adabn(self, eval_data, reset):
        assert self.binded and self.params_initialized
        if self.adabn:
            self._update_adabn(eval_data)
        if not reset and self.adabn:
            eval_data.reset()

    def score(
        self,
        eval_data,
        eval_metric,
        num_batch=None,
        batch_end_callback=None,
        reset=True,
        epoch=0
    ):
        self._try_update_adabn(eval_data, reset)
        return super(AdaBNMixin, self).score(
            eval_data,
            eval_metric,
            num_batch,
            batch_end_callback,
            reset,
            epoch
        )

    def predict(
        self,
        eval_data,
        num_batch=None,
        merge_batches=True,
        reset=True,
        always_output_list=False
    ):
        self._try_update_adabn(eval_data, reset)
        return super(AdaBNMixin, self).predict(
            eval_data,
            num_batch,
            merge_batches,
            reset,
            always_output_list
        )


class LSTMBindMixin(object):

    def __init__(self, **kargs):
        parent = kargs.get('parent', {})
        self.lstm = kargs.pop('lstm', getattr(parent, 'lstm', False))
        self.num_lstm_hidden = kargs.pop('num_lstm_hidden', getattr(parent, 'num_lstm_hidden', constant.NUM_LSTM_HIDDEN))
        self.num_lstm_layer = kargs.pop('num_lstm_layer', getattr(parent, 'num_lstm_layer', constant.NUM_LSTM_LAYER))
        symbol_kargs = kargs.pop('symbol_kargs', getattr(parent, 'symbol_args', {})).copy()
        symbol_kargs.update(
            lstm=self.lstm,
            num_lstm_hidden=self.num_lstm_hidden,
            num_lstm_layer=self.num_lstm_layer
        )
        kargs['symbol_kargs'] = symbol_kargs
        super(LSTMBindMixin, self).__init__(**kargs)

    def bind(self, data_shapes, label_shapes=None, for_training=True,
             inputs_need_grad=False, force_rebind=False, shared_module=None):
        """Bind the symbols to construct executors. This is necessary before one
        can perform computation with the module.

        Parameters
        ----------
        data_shapes : list of (str, tuple)
            Typically is `data_iter.provide_data`.
        label_shapes : list of (str, tuple)
            Typically is `data_iter.provide_label`.
        for_training : bool
            Default is `True`. Whether the executors should be bind for training.
        inputs_need_grad : bool
            Default is `False`. Whether the gradients to the input data need to be computed.
            Typically this is not needed. But this might be needed when implementing composition
            of modules.
        force_rebind : bool
            Default is `False`. This function does nothing if the executors are already
            binded. But with this `True`, the executors will be forced to rebind.
        shared_module : Module
            Default is `None`. This is used in bucketing. When not `None`, the shared module
            essentially corresponds to a different bucket -- a module with different symbol
            but with the same sets of parameters (e.g. unrolled RNNs with different lengths).
        """
        # force rebinding is typically used when one want to switch from
        # training to prediction phase.
        if force_rebind:
            self._reset_bind()

        if self.binded:
            self.logger.warning('Already binded, ignoring bind()')
            return

        self.for_training = for_training
        self.inputs_need_grad = inputs_need_grad
        self.binded = True

        if not for_training:
            assert not inputs_need_grad
        else:
            pass
            # this is not True, as some module might not contains a loss function
            # that consumes the labels
            # assert label_shapes is not None

        self._data_shapes = data_shapes
        self._label_shapes = label_shapes

        if shared_module is not None:
            assert isinstance(shared_module, Module) \
                and shared_module.binded and shared_module.params_initialized
            shared_group = shared_module._exec_group
        else:
            shared_group = None

        if self.lstm:
            self._exec_group = ExecutorGroup(self._symbol, self._context,
                                             self._work_load_list, data_shapes,
                                             label_shapes, self._param_names,
                                             for_training, inputs_need_grad,
                                             shared_group, logger=self.logger,
                                             num_lstm_layer=self.num_lstm_layer,
                                             num_lstm_hidden=self.num_lstm_hidden)
        else:
            self._exec_group = PresnetExecutorGroup(self._symbol, self._context,
                                                    self._work_load_list, data_shapes,
                                                    label_shapes, self._param_names,
                                                    for_training, inputs_need_grad,
                                                    shared_group, logger=self.logger,
                                                    num_presnet=self.symbol.num_presnet)
            #  self._exec_group = mx.module.executor_group.DataParallelExecutorGroup(
                #  self._symbol, self._context,
                #  self._work_load_list, data_shapes,
                #  label_shapes, self._param_names,
                #  for_training, inputs_need_grad,
                #  shared_group, logger=self.logger)

        if shared_module is not None:
            self.params_initialized = True
            self._arg_params = shared_module._arg_params
            self._aux_params = shared_module._aux_params

        if self.params_initialized:
            # if the parameters are already initialized, we are re-binding
            # so automatically copy the already initialized params
            self._exec_group.set_params(self._arg_params, self._aux_params)

        if shared_module is not None and shared_module.optimizer_initialized:
            self.borrow_optimizer(shared_module)


class LSTMMixin(LSTMBindMixin):

    def score(
        self,
        eval_data,
        eval_metric,
        num_batch=None,
        batch_end_callback=None,
        reset=True,
        epoch=0
    ):
        assert self.binded and self.params_initialized

        if not self.lstm:
            return super(LSTMMixin, self).score(
                eval_data,
                eval_metric,
                num_batch,
                batch_end_callback,
                reset,
                epoch
            )

        bottleneck = self._get_bottleneck(
            eval_data,
            num_batch=num_batch,
            merge_batches=True,
            reset=reset,
            always_output_list=False
        )
        mod = self._get_lstm_mod(bottleneck)
        return mod.score(
            bottleneck,
            eval_metric,
            num_batch,
            batch_end_callback,
            reset,
            epoch
        )

    def predict(
        self,
        eval_data,
        num_batch=None,
        merge_batches=True,
        reset=True,
        always_output_list=False
    ):
        assert self.binded and self.params_initialized

        if not self.lstm:
            return super(LSTMMixin, self).predict(
                eval_data,
                num_batch,
                merge_batches,
                reset,
                always_output_list
            )

        bottleneck = self._get_bottleneck(
            eval_data,
            num_batch=num_batch,
            merge_batches=merge_batches,
            reset=reset,
            always_output_list=always_output_list
        )
        mod = self._get_lstm_mod(bottleneck)
        return mod.predict(bottleneck)

    def _get_bottleneck(self, data, **kargs):
        mod = AdaBNModule(
            context=self._context,
            symbol=self.symbol.bottleneck,
            parent=self
        )
        mod.bind(data_shapes=data.provide_data, for_training=False)
        arg_params, aux_params = self.get_params()
        mod.init_params(arg_params=arg_params, aux_params=aux_params)
        bottleneck = mod.predict(data, **kargs).asnumpy()
        data.reset()
        return MovingWindow(
            bottleneck,
            batch_size=data.batch_size,
            window=self.symbol.lstm_window,
            segment=data.segment,
            label=data.gesture,
            stride=1 if not data.dense_window else data.window
        )

    def _get_lstm_mod(self, data):
        if self.symbol.lstm_bn:
            class Mod(LSTMBindMixin, AdaBNModule):
                pass
            if data.stride > 1:
                Mod = partial(Mod, num_adabn_epoch=1)
        else:
            class Mod(LSTMBindMixin, Base):
                pass
        mod = Mod(
            context=self._context,
            symbol=self.symbol.lstm,
            data_names=[x[0] for x in data.provide_data],
            parent=self
        )
        mod.bind(data_shapes=data.provide_data, for_training=False)
        arg_params, aux_params = self.get_params()
        mod.init_params(arg_params=arg_params, aux_params=aux_params)
        return mod


class ExecutorGroup(mx.module.executor_group.DataParallelExecutorGroup):

    def __init__(self, *args, **kargs):
        self.num_lstm_layer = kargs.pop('num_lstm_layer')
        self.num_lstm_hidden = kargs.pop('num_lstm_hidden')
        super(ExecutorGroup, self).__init__(*args, **kargs)

    def _bind_ith_exec(self, i, data_shapes, label_shapes, shared_group):
        """Internal utility function to bind the i-th executor.
        """
        nd = mx.nd

        data_shapes = self._sliced_shape(data_shapes, i)
        if label_shapes is not None:
            label_shapes = self._sliced_shape(label_shapes, i)
        shared_exec = None if shared_group is None else shared_group.execs[i]
        context = self.contexts[i]
        shared_data_arrays = self.shared_data_arrays[i]

        input_shapes = dict(data_shapes)
        if label_shapes is not None:
            input_shapes.update(dict(label_shapes))

        shapes = {}
        batch_size = data_shapes[0][1][0]
        shapes.update({'lstm_l%d_init_c' % i: (batch_size, self.num_lstm_hidden)
                       for i in range(self.num_lstm_layer)})
        shapes.update({'lstm_l%d_init_h' % i: (batch_size, self.num_lstm_hidden)
                       for i in range(self.num_lstm_layer)})
        shapes.update(input_shapes)
        arg_shapes, _, aux_shapes = self.symbol.infer_shape(**shapes)
        assert arg_shapes is not None, "shape inference failed"

        if self.input_types is None:
            input_types = {k: mx.base.mx_real_t for k in input_shapes.keys()}
        else:
            input_types = self.input_types
        types = {}
        types.update({'lstm_l%d_init_c' % i: mx.base.mx_real_t
                      for i in range(self.num_lstm_layer)})
        types.update({'lstm_l%d_init_h' % i: mx.base.mx_real_t
                      for i in range(self.num_lstm_layer)})
        types.update(input_types)
        arg_types, _, aux_types = self.symbol.infer_type(**types)
        assert arg_types is not None, "type inference failed"

        data_names = [x[0] for x in data_shapes]

        arg_arrays = []
        grad_arrays = {} if self.for_training else None
        grad_req = {}
        for name in self.arg_names:
            if self.for_training:
                if name in self.param_names:
                    grad_req[name] = 'write'
                elif name in data_names:
                    grad_req[name] = 'write' if self.inputs_need_grad else 'null'
                else:
                    grad_req[name] = 'null'
            else:
                grad_req[name] = 'null'

        def _get_or_reshape(name, shared_data_arrays, arg_shape, arg_type, context, logger):
            """Internal helper to get a memory block or re-use by re-shaping"""
            if name in shared_data_arrays:
                arg_arr = shared_data_arrays[name]

                if np.prod(arg_arr.shape) >= np.prod(arg_shape):
                    # nice, we can directly re-use this data blob
                    assert arg_arr.dtype == arg_type
                    arg_arr = arg_arr.reshape(arg_shape)
                else:
                    logger.warning(('bucketing: data "%s" has a shape %s' % (name, arg_shape)) +
                                   (', which is larger than already allocated ') +
                                   ('shape %s' % (arg_arr.shape,)) +
                                   ('. Need to re-allocate. Consider putting ') +
                                   ('default_bucket_key to') +
                                   (' be the bucket taking the largest input for better ') +
                                   ('memory sharing.'))
                    arg_arr = nd.zeros(arg_shape, context, dtype=arg_type)

                    # replace existing shared array because the new one is bigger
                    shared_data_arrays[name] = arg_arr
            else:
                arg_arr = nd.zeros(arg_shape, context, dtype=arg_type)
                shared_data_arrays[name] = arg_arr

            return arg_arr

        # create or borrow arguments and gradients
        for j in range(len(self.arg_names)):
            name = self.arg_names[j]
            if name in self.param_names:  # model parameter
                if shared_exec is None:
                    arg_arr = nd.zeros(arg_shapes[j], context, dtype=arg_types[j])
                    if grad_req[name] != 'null':
                        grad_arr = nd.zeros(arg_shapes[j], context, dtype=arg_types[j])
                        grad_arrays[name] = grad_arr
                else:
                    arg_arr = shared_exec.arg_dict[name]
                    assert arg_arr.shape == arg_shapes[j]
                    assert arg_arr.dtype == arg_types[j]
                    if grad_req[name] != 'null':
                        grad_arrays[name] = shared_exec.grad_dict[name]
            else:  # data or label
                arg_arr = _get_or_reshape(name, shared_data_arrays, arg_shapes[j], arg_types[j],
                                          context, self.logger)

                # data might also need grad if inputs_need_grad is True
                if grad_req[name] != 'null':
                    grad_arrays[name] = _get_or_reshape('grad of ' + name, shared_data_arrays,
                                                        arg_shapes[j], arg_types[j], context,
                                                        self.logger)

            arg_arrays.append(arg_arr)

        # create or borrow aux variables
        if shared_exec is None:
            aux_arrays = [nd.zeros(s, context, dtype=t) for s, t in zip(aux_shapes, aux_types)]
        else:
            for j, arr in enumerate(shared_exec.aux_arrays):
                assert aux_shapes[j] == arr.shape
                assert aux_types[j] == arr.dtype
            aux_arrays = shared_exec.aux_arrays[:]

        executor = self.symbol.bind(ctx=context, args=arg_arrays,
                                    args_grad=grad_arrays, aux_states=aux_arrays,
                                    grad_req=grad_req, shared_exec=shared_exec)
        return executor


class MovingWindow(mx.io.DataIter):

    def __init__(
        self,
        data,
        batch_size,
        window,
        segment,
        label=None,
        stride=1
    ):
        super(MovingWindow, self).__init__()

        assert_is_instance(data, np.ndarray)
        assert_equal(len(data.shape), 2)
        self.data = data
        self.label = label
        self.num_bottleneck = data.shape[1]
        self.batch_size = batch_size
        self.provide_data = [('data', (self.batch_size, window, self.num_bottleneck))]
        self.window = window
        self.segment = segment
        self.stride = stride

        # Dummy
        self.shuffle = False

        index = []
        cursor = -1
        while True:
            cursor += 1
            if cursor + self.window * self.stride <= self.data.shape[0]:
                if self.segment[cursor] != self.segment[cursor + (self.window - 1) * self.stride]:
                    cursor = cursor + (self.window - 1) * stride
                assert np.all(self.segment[cursor:cursor + (self.window - 1) * self.stride:self.stride]
                              == self.segment[cursor + self.stride:cursor + self.window * self.stride:self.stride])
            if cursor + (self.window - 1) * self.stride >= self.data.shape[0]:
                break
            index.append(cursor)
        self.index = np.array(index)

        self.reset()

    def reset(self):
        self.index.sort()
        if self.shuffle:
            np.random.shuffle(self.index)
        self._cursor = -1

    def iter_next(self):
        current_data = []
        current_label = []
        if not self._iter_next():
            return False
        pad = 0
        for i in range(self.batch_size):
            if i and not self._iter_next():
                current_data.append(current_data[-1])
                current_label.append(current_label[-1])
                pad += 1
            else:
                data = self.data[self.cursor:self.cursor + self.window * self.stride:self.stride][np.newaxis, ...]
                current_data.append(data)
                if self.label is None:
                    current_label.append(0)
                else:
                    assert np.all(self.label[self.cursor:self.cursor + (self.window - 1) * self.stride:self.stride]
                                  == self.label[self.cursor + self.stride:self.cursor + self.window * self.stride:self.stride])
                    current_label.append(self.label[self.cursor])
        self.current_data = mx.nd.array(np.concatenate(current_data, axis=0))
        self.current_label = mx.nd.array(current_label)
        self.pad = pad
        return True

    @property
    def cursor(self):
        return self.index[self._cursor]

    def _iter_next(self):
        self._cursor += 1
        return self._cursor < len(self.index)

    def getdata(self):
        return [self.current_data]

    def getlabel(self):
        return [self.current_label]

    def getpad(self):
        return self.pad


class AdaBNModule(AdaBNMixin, Base):
    pass


class Module(LSTMMixin, AdaBNMixin, FaugMixin, Base):

    @classmethod
    def parse(cls, text, **kargs):
        if text == 'convnet':
            return cls(**kargs)
        from .base_module import BaseModule
        from .sklearn_module import SklearnModule
        assert SklearnModule
        return BaseModule.parse(text, **kargs)

    def __init__(self, **kargs):
        self.kargs = kargs.copy()
        self.pixel_same_init = kargs.pop('pixel_same_init', False)
        self.revgrad = kargs.pop('revgrad', False)
        self.num_revgrad_batch = kargs.pop('num_revgrad_batch', None)
        self.tzeng = kargs.pop('tzeng', False)
        self.num_tzeng_batch = kargs.pop('num_tzeng_batch', constant.NUM_TZENG_BATCH)
        self.soft_label = kargs.pop('soft_label', False)
        self.random_scale = kargs.pop('random_scale', False)
        self.num_gesture = kargs.pop('num_gesture')
        self.num_subject = kargs.pop('num_subject', 0)
        target_gesture_loss_weight = kargs.pop('target_gesture_loss_weight', None)
        self.lambda_scale = kargs.pop('lambda_scale', constant.LAMBDA_SCALE)
        self.dual_stream = kargs.pop('dual_stream', False)
        self.for_training = kargs.pop('for_training', False)
        self.snapshot_period = kargs.pop('snapshot_period', 1)
        subject_loss_weight = kargs.pop('subject_loss_weight', None)
        symbol_kargs = kargs.pop('symbol_kargs', {}).copy()
        symbol_kargs.update(
            num_gesture=self.num_gesture,
            num_subject=self.num_subject,
            subject_loss_weight=subject_loss_weight,
            target_gesture_loss_weight=target_gesture_loss_weight,
            coral=kargs.pop('coral', False),
            revgrad=self.revgrad,
            tzeng=self.tzeng,
            soft_label=self.soft_label,
            for_training=self.for_training
        )
        kargs['symbol_kargs'] = symbol_kargs
        label_names = ['gesture_softmax_label']
        if self.soft_label:
            label_names.append('soft_label')
        if self.revgrad or self.tzeng:
            label_names.append('subject_softmax_label')
            # for i in range(self.num_gesture):
                # label_names.append('gesture%d_subject_softmax_label' % i)
        if target_gesture_loss_weight is not None:
            label_names.append('target_gesture_softmax_label')

        if not self.dual_stream:
            data_names = ['data']
        else:
            data_names = ['stream0_data', 'stream1_data']

        super(Module, self).__init__(
            data_names=data_names,
            label_names=label_names,
            **kargs
        )

    def _get_init(self):
        return Init(factor_type='in', magnitude=2, mod=self)

    def fit(self, **kargs):
        num_epoch = kargs.pop('num_epoch')
        num_train = kargs.pop('num_train')
        batch_size = kargs.pop('batch_size')
        epoch_size = num_train / batch_size
        lr_step = kargs.pop('lr_step')
        lr = kargs.pop('lr', 0.1)
        wd = kargs.pop('wd', 0.0001)
        gamma = kargs.pop('gamma')
        snapshot = kargs.pop('snapshot')
        params = kargs.pop('params', None)
        ignore_params = kargs.pop('ignore_params', [])
        fix_params = kargs.pop('fix_params', [])
        decay_all = kargs.pop('decay_all', False)
        lr_factor = kargs.pop('lr_factor', ()) or 0.1

        checkpoint = []
        #  if self.revgrad:
            #  checkpoint.append(partial(
                #  update_revgrad_epoch,
                #  self,
                #  gamma,
                #  self.lambda_scale,
                #  num_epoch
            #  ))
        if snapshot:
            def do_checkpoint(prefix, period=1):
                def _callback(iter_no, sym, arg, aux):
                    #  Always save the first epoch
                    if iter_no == 0 or (iter_no + 1) % period == 0:
                        mx.model.save_checkpoint(prefix, iter_no + 1, sym, arg, aux)
                period = int(max(1, period))
                return _callback
            checkpoint.append(do_checkpoint(snapshot, period=self.snapshot_period))

        if params:
            logger.info('Load params from {}', params)
            init = Load(
                params,
                default_init=self._get_init(),
                ignore=ignore_params,
                mod=self
            )
        else:
            init = self._get_init()

        #  self.monitor = mx.mon.Monitor(
            #  50,
            #  lambda d: mx.nd.norm(d) / np.sqrt(d.size),
            #  #  pattern='subject_.*fc\d*_(?:output|weight)',
            #  pattern='.*(gesture_last_fc|subject_last_fc|bottleneck)_(backward_data|backward_weight|_scale).*',
            #  sort=True
        #  )
        #  self.monitor_installed = False

        def nbatch(param):
            nbatch.value = param.nbatch

        batch_end_callback = [mx.callback.Speedometer(batch_size, 50), nbatch]
        if self.revgrad:
            batch_end_callback.append(partial(
                update_revgrad_batch,
                self,
                gamma,
                self.lambda_scale,
                num_epoch,
                self.num_revgrad_batch
            ))
        if self.tzeng:
            batch_end_callback.append(partial(
                update_tzeng, self, self.lambda_scale, self.num_tzeng_batch))

        #  def debug(params):
            #  data = params.locals['data_batch'].data[0].asnumpy()
            #  label = params.locals['data_batch'].label[0].asnumpy()
            #  import joblib as jb
            #  jb.dump(dict(data=data, label=label), 'dump')
            #  import sys
            #  sys.exit(0)

        #  batch_end_callback.append(debug)

        return super(Module, self).fit(
            eval_metric=(
                Accuracy(0, 'g') if not (self.revgrad or self.tzeng) else
                [Accuracy(0, 'g'), Accuracy(1, 's')]
            ),
            optimizer='Sigropt',
            optimizer_params=dict(
                learning_rate=lr,
                momentum=0.9,
                wd=wd,
                lr_scheduler=Scheduler(
                    lr_step=lr_step,
                    factor=lr_factor,
                    epoch_size=epoch_size
                ),
                tzeng=self.tzeng,
                revgrad=self.revgrad,
                get_nbatch=lambda: getattr(nbatch, 'value', -1),
                clip_gradient=1,
                fix_params=fix_params,
                decay_all=decay_all
            ),
            initializer=init,
            num_epoch=num_epoch,
            batch_end_callback=batch_end_callback,
            epoch_end_callback=checkpoint,
            **kargs
        )

    def forward_backward(self, data_batch):
        if hasattr(self, 'monitor'):
            if not self.monitor_installed:
                for exe in self._exec_group.execs:
                    self.monitor.install(exe)
                self.monitor_installed = True
            self.monitor.tic()
        super(Module, self).forward_backward(data_batch)

    def update(self):
        super(Module, self).update()
        if hasattr(self, 'monitor'):
            self.monitor.toc_print()

    def init_coral(self, params_fname, source_data, target_data):
        kargs = self.kargs.copy()
        kargs['symbol'] = self.symbol.get_internals()['bottleneck_relu_output']
        inst = type(self)(**kargs)
        inst.bind(data_shapes=target_data.provide_data, for_training=True)
        inst.load_params(params_fname)

        if not isinstance(source_data, (list, tuple)):
            source_data = [source_data]
        ds = []
        for data in source_data:
            ds.append(inst.predict(data).asnumpy())
        ds = np.vstack(ds)

        dt = inst.predict(target_data).asnumpy()

        from .coral import get_coral_params
        coral_weight, coral_bias = get_coral_params(ds, dt)

        def init_coral(name, arr):
            assert name.startswith('coral')
            if name.endswith('weight'):
                arr[:] = coral_weight
            elif name.endswith('bias') and coral_bias is not None:
                arr[:] = coral_bias
            else:
                assert False

        self.bind(data_shapes=target_data.provide_data, for_training=True)
        self.init_params(Load(params_fname, default_init=init_coral, mod=self))

    def init_optimizer(self, kvstore='local', optimizer='sgd',
                       optimizer_params=(('learning_rate', 0.01),), force_init=False):
        """Install and initialize optimizers.

        Parameters
        ----------
        kvstore : str or KVStore
            Default `'local'`.
        optimizer : str or Optimizer
            Default `'sgd'`
        optimizer_params : dict
            Default `(('learning_rate', 0.01),)`. The default value is not a dictionary,
            just to avoid pylint warning of dangerous default values.
        force_init : bool
            Default `False`, indicating whether we should force re-initializing the
            optimizer in the case an optimizer is already installed.
        """
        assert self.binded and self.params_initialized
        opt = mx.optimizer

        if self.optimizer_initialized and not force_init:
            self.logger.warning('optimizer already initialized, ignoring...')
            return

        (kvstore, update_on_kvstore) = mx.model._create_kvstore(kvstore, len(self._context), self._arg_params)

        if isinstance(optimizer, str):
            batch_size = self._exec_group.batch_size
            if kvstore and kvstore.type == 'dist_sync':
                batch_size *= kvstore.num_workers
            idx2name = {}
            # Fix bug in mxnet for multi-gpu training
            if update_on_kvstore:
                idx2name.update(enumerate(self._exec_group.param_names))
            else:
                for k in range(len(self._context)):
                    idx2name.update({i*len(self._context)+k: n
                                    for i, n in enumerate(self._exec_group.param_names)})
            optimizer_params = dict(optimizer_params)
            if 'rescale_grad' not in optimizer_params:
                optimizer_params['rescale_grad'] = 1.0/batch_size
            optimizer = opt.create(optimizer,
                                   sym=self.symbol, param_idx2name=idx2name,
                                   **optimizer_params)
        else:
            assert isinstance(optimizer, opt.Optimizer)

        self._optimizer = optimizer
        self._kvstore = kvstore
        self._update_on_kvstore = update_on_kvstore
        self._updater = None

        if not update_on_kvstore:
            self._updater = opt.get_updater(optimizer)
        if kvstore:
            # copy initialized local parameters to kvstore
            mx.model._initialize_kvstore(kvstore=kvstore,
                                         param_arrays=self._exec_group.param_arrays,
                                         arg_params=self._arg_params,
                                         param_names=self._param_names,
                                         update_on_kvstore=update_on_kvstore)
        if update_on_kvstore:
            kvstore.set_optimizer(self._optimizer)

        self.optimizer_initialized = True


class Accuracy(mx.metric.EvalMetric):

    def __init__(self, index, name):
        super(Accuracy, self).__init__('accuracy[%s]' % name)
        if not isinstance(index, list):
            index = [index]
        self.index = index

    def update(self, labels, preds):
        # mx.metric.check_label_shapes(labels, preds)

        for index in self.index:
            label = labels[index].asnumpy().astype('int32')
            assert label.ndim in (1, 2)
            if label.ndim == 1:
                pred_label = mx.nd.argmax_channel(preds[index]).asnumpy().astype('int32')
            else:
                pred_label = (preds[index].asnumpy() > 0.5).astype('int32')

            # mx.metric.check_label_shapes(label, pred_label)

            if label.ndim == 1:
                mask = label >= 0
                label = label[mask]
                pred_label = pred_label[mask]

            # mx.metric.check_label_shapes(label, pred_label)

            self.sum_metric += (pred_label == label).sum()
            self.num_inst += pred_label.size


class Scheduler(mx.lr_scheduler.LRScheduler):

    def __init__(self, lr_step, factor, epoch_size):
        if not isinstance(lr_step, (tuple, list)):
            lr_step = list(range(lr_step, 1000, lr_step))
        step = [epoch_size * s for s in lr_step]
        super(Scheduler, self).__init__()
        assert isinstance(step, list) and len(step) >= 1
        for i, _step in enumerate(step):
            if i != 0 and step[i] <= step[i-1]:
                raise ValueError("Schedule step must be an increasing integer list")
            if _step < 1:
                raise ValueError("Schedule step must be greater or equal than 1 round")
        self.step = step
        self.cur_step_ind = 0
        self.factor = factor
        self.count = 0

    def __call__(self, num_update):
        """
        Call to schedule current learning rate

        Parameters
        ----------
        num_update: int
            the maximal number of updates applied to a weight.
        """

        if self.cur_step_ind <= len(self.step)-1:
            if num_update > self.step[self.cur_step_ind]:
                self.count = self.step[self.cur_step_ind]
                self.cur_step_ind += 1
                self.base_lr *= (
                    self.factor[self.cur_step_ind - 1]
                    if isinstance(self.factor, (tuple, list)) else self.factor
                )
                logging.info("Update[%d]: Change learning rate to %0.5e",
                             num_update, self.base_lr)
        return self.base_lr


class Load(mx.init.Load):

    def __init__(self, params, *args, **kargs):
        self.ignore = kargs.pop('ignore', [])
        mod = kargs.pop('mod')
        if not os.path.exists(params) and os.path.exists(os.path.join(ROOT, params)):
            params = os.path.join(ROOT, params)
        super(Load, self).__init__(params, *args, **kargs)
        for name in list(self.param):
            for ignore in self.ignore:
                if re.match(ignore, name):
                    logger.info('Ignore param {}', name)
                    del self.param[name]
                    break
            if mod.adabn and (name.endswith('moving_mean')
                              or name.endswith('moving_var')):
                del self.param[name]
            if mod.lstm and 'lstm' in name and (name.endswith('_init_c')
                                                or name.endswith('_init_h')):
                del self.param[name]

    def __call__(self, name, arr):
        if name in self.param and ('gamma' in name or 'beta' in name):
            self.param[name] = self.param[name].reshape(arr.shape)
        return super(Load, self).__call__(name, arr)


class Init(mx.init.Xavier):

    def __init__(self, *args, **kargs):
        self.mod = kargs.pop('mod')
        super(Init, self).__init__(*args, **kargs)

    def __call__(self, name, arr):
        """Override () function to do Initialization

        Parameters
        ----------
        name : str
            name of corrosponding ndarray

        arr : NDArray
            ndarray to be Initialized
        """
        if not isinstance(name, mx.base.string_types):
            raise TypeError('name must be string')
        if not isinstance(arr, mx.ndarray.NDArray):
            raise TypeError('arr must be NDArray')
        if name.endswith('lambda'):
            self._init_zero(name, arr)
        elif name.endswith('_zero'):
            self._init_zero(name, arr)
        elif name.endswith('_one'):
            self._init_one(name, arr)
        elif name.endswith('gradscale_scale'):
            self._init_gradscale(name, arr)
        elif name.startswith('sum'):
            self._init_gamma(name, arr)
        elif self.mod.presnet_proj_type == 'A' and 'proj' in name and name.endswith('weight'):
            self._init_proj(name, arr)
        elif name.endswith('h2h_weight'):
            self._init_h2h(name, arr)
        elif 'im2col' in name and name.endswith('weight'):
            self._init_im2col(name, arr)
        elif self.mod.pixel_same_init and 'pixel' in name and name.endswith('weight'):
            self._init_pixel(name, arr)
        elif 'lstm' in name and name.endswith('_init_c'):
            self._init_zero(name, arr)
        elif 'lstm' in name and name.endswith('_init_h'):
            self._init_zero(name, arr)
        elif name.endswith('upsampling'):
            self._init_bilinear(name, arr)
        elif name.endswith('bias'):
            self._init_bias(name, arr)
        elif 'lstm' in name and name.endswith('gamma'):
            self._init_lstm_gamma(name, arr)
        elif name.endswith('gamma'):
            self._init_gamma(name, arr)
        elif name.endswith('beta'):
            self._init_beta(name, arr)
        elif name.endswith('weight'):
            self._init_weight(name, arr)
        elif name.endswith("moving_mean"):
            self._init_zero(name, arr)
        elif name.endswith("moving_var"):
            self._init_zero(name, arr)
        elif name.endswith("moving_inv_var"):
            self._init_zero(name, arr)
        elif name.endswith("moving_avg"):
            self._init_zero(name, arr)
        else:
            self._init_default(name, arr)

    def _init_one(self, _, arr):
        arr[:] = 1

    def _init_pixel(self, name, arr):
        if not hasattr(self, '_pixel_cache'):
            self._pixel_cache = {}
        tag = '_'.join(name.split('_')[:-2])
        if tag not in self._pixel_cache:
            self._init_weight(name, arr)
            self._pixel_cache[tag] = arr
        else:
            self._pixel_cache[tag].copyto(arr)

    def _init_proj(self, _, arr):
        '''Initialization of shortcut of kenel (2, 2)'''
        w = np.zeros(arr.shape, np.float32)
        for i in range(w.shape[1]):
            w[i, i, ...] = 0.25
        arr[:] = w

    def _init_gradscale(self, name, arr):
        arr[:] = {
            'gesture_gradscale_scale': 0,
            'bottleneck_gradscale_scale': 0,
            'subject_gradscale_scale': 1,
            'subject_confusion_gradscale_scale': 0,
        }[name]

    def _init_lstm_gamma(self, _, arr):
        arr[:] = 0.1

    def _init_h2h(self, _, arr):
        assert_equal(len(arr.shape), 2)
        assert_equal(arr.shape[0], arr.shape[1] * 4)
        n = arr.shape[1]
        eye = np.eye(n)
        for i in range(4):
            arr[i * n:(i + 1) * n] = eye

    def _init_im2col(self, _, arr):
        assert_equal(len(arr.shape), 4)
        assert_equal(arr.shape[0], arr.shape[1] * arr.shape[2] * arr.shape[3])
        arr[:] = np.eye(arr.shape[0]).reshape(arr.shape)


def update_revgrad_epoch(mod, gamma, scale, num_epoch, epoch, symbol, arg_params, aux_params):
    p = (epoch + 1) / num_epoch
    lam = 2 / (1 + np.exp(-gamma * p)) - 1
    lam *= scale
    aux_params['grl_lambda'][:] = lam
    aux_params = {'grl_lambda': aux_params['grl_lambda']}
    # Write back to devices
    mod._exec_group.set_params({}, aux_params)
    logger.info('Epoch[{}] Update lam {} with scale {}', epoch, lam, scale)


def update_revgrad_batch(mod, gamma, scale, num_epoch, interval, params):
    if interval > 1:
        confuse = params.nbatch % interval == interval - 1
    else:
        confuse = False
    mod._optimizer.confuse = confuse

    p = (params.epoch + 1) / num_epoch
    lam = 2 / (1 + np.exp(-gamma * p)) - 1
    lam *= scale

    for name, block in zip(mod._exec_group.aux_names, mod._exec_group.aux_arrays):
        if name == 'grl_lambda':
            for a in block:
                a[:] = lam if confuse else 0


# def update_tzeng(mod, epoch, symbol, arg_params, aux_params):
    # confuse = 1 - epoch % 2
    # aux_params['gesture_gradscale_scale'][:] = confuse
    # aux_params['bottleneck_gradscale_scale'][:] = 0.1 * confuse
    # aux_params['subject_gradscale_scale'][:] = 1 - confuse
    # aux_params['subject_confusion_gradscale_scale'][:] = confuse
    # mod._exec_group.set_params(arg_params, aux_params)
    # mod._optimizer.confuse = confuse
    # logger.info('Epoch[{}] Update Tzeng with confuse {}', epoch, confuse)


def update_tzeng(mod, scale, interval, params):
    '''Switch Tzeng confusion every 10 batches'''
    step = 50
    if params.nbatch % step == step - 1:
        confuse = (params.nbatch // step) % interval == interval - 1
        mod._optimizer.confuse = confuse
        for name, block in zip(mod._exec_group.aux_names, mod._exec_group.aux_arrays):
            if name == 'gesture_gradscale_scale':
                for a in block:
                    a[:] = confuse
            elif name == 'bottleneck_gradscale_scale':
                for a in block:
                    a[:] = scale * confuse
            elif name == 'subject_gradscale_scale':
                for a in block:
                    a[:] = 1 - confuse
            elif name == 'subject_confusion_gradscale_scale':
                for a in block:
                    a[:] = confuse


@mx.optimizer.register
class Sigropt(mx.optimizer.NAG):

    def __init__(self, tzeng, revgrad, get_nbatch, fix_params, decay_all, **kargs):
        self.decay_all = decay_all
        super(Sigropt, self).__init__(**kargs)
        self.tzeng = tzeng
        self.revgrad = revgrad
        self.get_nbatch = get_nbatch
        self.confuse = 0
        self.fix_params = fix_params

    @property
    def nbatch(self):
        return self.get_nbatch()

    def update(self, index, weight, grad, state):
        """Update the parameters.
        Parameters
        ----------
        index : int
            An unique integer key used to index the parameters
        weight : NDArray
            weight ndarray
        grad : NDArray
            grad ndarray
        state : NDArray or other objects returned by init_state
            The auxiliary state used in optimization.
        """
        assert(isinstance(weight, mx.nd.NDArray))
        assert(isinstance(grad, mx.nd.NDArray))
        lr = self._get_lr(index)
        wd = self._get_wd(index)
        self._update_count(index)

        grad = grad * self.rescale_grad
        if self.clip_gradient is not None:
            grad = mx.nd.clip(grad, -self.clip_gradient, self.clip_gradient)

        if lr:
            if state:
                mom = state
                mom *= self.momentum
                if wd:
                    # L1 reg
                    # grad += wd * mx.nd.sign(weight)
                    grad += wd * weight
                mom += grad
                grad += self.momentum * mom
                weight += -lr * grad
            else:
                assert self.momentum == 0.0
                if wd:
                    # weight += -lr * (grad + self.wd * mx.nd.sign(weight))
                    weight += -lr * (grad + self.wd * weight)
                else:
                    weight += -lr * grad

    def set_wd_mult(self, args_wd_mult):
        """Set individual weight decay multipler for parameters.
        By default wd multipler is 0 for all params whose name doesn't
        end with _weight, if param_idx2name is provided.

        Parameters
        ----------
        args_wd_mult : dict of string/int to float
            set the wd multipler for name/index to float.
            setting multipler by index is supported for backward compatibility,
            but we recommend using name and symbol.
        """
        if not self.decay_all:
            return super(Sigropt, self).set_wd_mult(args_wd_mult)

        self.wd_mult = {}
        for n in self.idx2name.values():
            if not (n.endswith('_weight')
                    or n.endswith('_gamma')
                    or n.endswith('_bias')
                    or n.endswith('_beta')) or 'zscore' in n:
                self.wd_mult[n] = 0.0
        if self.sym is not None:
            attr = self.sym.list_attr(recursive=True)
            for k, v in attr.items():
                if k.endswith('_wd_mult'):
                    self.wd_mult[k[:-len('_wd_mult')]] = float(v)
        self.wd_mult.update(args_wd_mult)

    def _get_lr(self, index):
        """get learning rate for index.

        Parameters
        ----------
        index : int
            The index for weight

        Returns
        -------
        lr : float
            learning rate for this index
        """
        lr = 1

        if index not in self.lr_mult and index in self.idx2name:
            index = self.idx2name[index]
        assert isinstance(index, str)

        if self.fix_params:
            for pattern in self.fix_params:
                if re.match(pattern, index):
                    return 0

        if self.tzeng:
            if index.startswith('subject'):
                lr = 1 - self.confuse
            else:
                lr = self.confuse
        #  if self.revgrad:
            #  if index.startswith('subject'):
                #  lr = 1 - self.confuse
        return super(Sigropt, self)._get_lr(index) if lr else 0


class RuntimeMixin(object):

    def __init__(self, **kargs):
        args = []
        backup = kargs.copy()
        self.params = kargs.pop('params')
        super(RuntimeMixin, self).__init__(**kargs)
        self.args = args
        self.kargs = backup

    def predict(self, eval_data, *args, **kargs):
        if not getattr(self, 'incache', False):
            return _predict(utils.LazyProxy(self.Clone), eval_data)

        if not self.binded:
            self.bind(data_shapes=[('data', (eval_data.batch_size, self.data_shape_1,
                                             self.num_semg_row, self.num_semg_col))],
                      for_training=False)
        if not self.params_initialized:
            self.init_params(Load(self.params, default_init=self._get_init(), mod=self))
        return super(RuntimeMixin, self).predict(eval_data, *args, **kargs)

    def predict_proba(self, eval_data, *args, **kargs):
        if not getattr(self, 'incache', False):
            return _predict_proba(utils.LazyProxy(self.Clone), eval_data)

        if not self.binded:
            self.bind(data_shapes=[('data', (eval_data.batch_size, self.data_shape_1,
                                             self.num_semg_row, self.num_semg_col))],
                      for_training=False)
        if not self.params_initialized:
            self.init_params(Load(self.params, default_init=self._get_init(), mod=self))
        return super(RuntimeMixin, self).predict(eval_data, *args, **kargs)

    @property
    def Clone(self):
        return partial(type(self), *self.args, **self.kargs)


@utils.cached
def _predict(mod, val):
    mod.incache = True

    val.reset()
    true = val.gesture.copy()
    segment = val.segment.copy()
    val.reset()
    assert np.all(true == val.gesture.copy())
    assert np.all(segment == val.segment.copy())

    out = mod.predict(val).asnumpy()
    assert_equal(out.ndim, 2)
    assert_equal(out.shape[1], mod.num_gesture)
    pred = out.argmax(axis=1)
    if mod.lstm:
        cut = lambda a: np.hstack(a[begin:end + mod.num_channel - mod.data_shape_1]
                                  for begin, end in utils.continuous_segments(segment))
        true = cut(true)
        segment = cut(segment)
    assert_equal(true.shape, pred.shape)
    return pred, true, segment


def _predict_proba(mod, val):
    mod.incache = True

    val.reset()
    true = val.gesture.copy()
    segment = val.segment.copy()
    val.reset()
    assert np.all(true == val.gesture.copy())
    assert np.all(segment == val.segment.copy())

    out = mod.predict(val).asnumpy()
    #  assert_equal(out.ndim, 2)
    #  assert_equal(out.shape[1], mod.num_gesture)
    #  pred = out.argmax(axis=1)
    if mod.lstm:
        cut = lambda a: np.hstack(a[begin:end + mod.num_channel - mod.data_shape_1]
                                  for begin, end in utils.continuous_segments(segment))
        true = cut(true)
        segment = cut(segment)
    #  assert_equal(true.shape, pred.shape)
    return out, true, segment


class RuntimeModule(RuntimeMixin, Module):
    pass
