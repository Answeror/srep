from __future__ import division
import mxnet as mx
from nose.tools import assert_equal
from . import constant
# import numpy as np


class GRL(mx.operator.CustomOp):

    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], 0 + in_data[0])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(
            in_grad[0],
            req[0],
            -aux[0] * out_grad[0]
        )


@mx.operator.register('GRL')
class GRLProp(mx.operator.CustomOpProp):

    def __init__(self):
        super(GRLProp, self).__init__(need_top_grad=True)

    def list_auxiliary_states(self):
        return ['lambda']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        output_shape = in_shape[0]
        return [data_shape], [output_shape], [(1,)]

    def create_operator(self, ctx, shapes, dtypes):
        return GRL()


class GradScale(mx.operator.CustomOp):

    def forward(self, is_train, req, in_data, out_data, aux):
        self.assign(out_data[0], req[0], 0 + in_data[0])

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(
            in_grad[0],
            req[0],
            aux[0] * out_grad[0]
        )


@mx.operator.register('GradScale')
class GradScaleProp(mx.operator.CustomOpProp):

    def __init__(self):
        super(GradScaleProp, self).__init__(need_top_grad=True)

    def list_auxiliary_states(self):
        return ['scale']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        output_shape = in_shape[0]
        return [data_shape], [output_shape], [(1,)]

    def create_operator(self, ctx, shapes, dtypes):
        return GradScale()


class Symbol(object):

    def get_bn_orig(self, name, data):
        return mx.symbol.BatchNorm(
            name=name,
            data=data,
            fix_gamma=True,
            momentum=0.9,
            # attr={'wd_mult': '0'}
            # eps=1e-5
        )

    def infer_shape(self, data):
        net = data
        if self.num_stream == 1:
            data_shape = (self.num_subject if self.minibatch else 1,
                          self.num_channel, self.num_semg_row, self.num_semg_col)
            shape = tuple(int(s) for s in net.infer_shape(data=data_shape)[1][0])
        else:
            shape = tuple(int(s) for s in net.infer_shape(
                **{'stream%d_data' % i: (self.num_subject if self.minibatch else 1,
                                         self.num_channel[i], self.num_semg_row, self.num_semg_col)
                    for i in range(self.num_stream) if 'stream%d_data_tag' % i in net.list_attr(recursive=True)})[1][0])
        return shape

    def get_bn(self, name, data):
        if not self.bng:
            if self.minibatch:
                net = data
                if self.num_stream == 1:
                    data_shape = (self.num_subject, self.num_channel, self.num_semg_row, self.num_semg_col)
                    shape = tuple(int(s) for s in net.infer_shape(data=data_shape)[1][0])
                else:
                    shape = tuple(int(s) for s in net.infer_shape(
                        **{'stream%d_data' % i: (self.num_subject, self.num_channel[i], self.num_semg_row, self.num_semg_col)
                           for i in range(self.num_stream) if 'stream%d_data_tag' % i in net.list_attr(recursive=True)})[1][0])
                net = mx.symbol.Reshape(net, shape=(-1, self.num_subject * shape[1]) + shape[2:])
                net = mx.symbol.BatchNorm(
                    name=name + '_norm',
                    data=net,
                    fix_gamma=True,
                    momentum=0.9,
                    attr={'wd_mult': '0', 'lr_mult': '0'}
                )
                net = mx.symbol.Reshape(data=net, shape=(-1,) + shape[1:])
                if len(shape) == 4:
                    gamma = mx.symbol.Variable(name + '_gamma', shape=(1, shape[1], 1, 1))
                    beta = mx.symbol.Variable(name + '_beta', shape=(1, shape[1], 1, 1))
                else:
                    gamma = mx.symbol.Variable(name + '_gamma', shape=(1, shape[1]))
                    beta = mx.symbol.Variable(name + '_beta', shape=(1, shape[1]))
                net = mx.symbol.broadcast_mul(net, gamma)
                net = mx.symbol.broadcast_plus(net, beta, name=name + '_last')
                #  new_shape = tuple(int(s) for s in net.infer_shape(data=data_shape)[1][0])
                #  assert new_shape == shape
                return net
            else:
                net = mx.symbol.BatchNorm(
                    name=name,
                    data=data,
                    fix_gamma=False,
                    momentum=0.9
                )
        else:
            net = self.get_bng(name, data)
        return net

    def get_bng(self, name, data):
        net = data
        if self.num_stream == 1:
            data_shape = (1, self.num_channel, self.num_semg_row, self.num_semg_col)
            shape = tuple(int(s) for s in net.infer_shape(data=data_shape)[1][0])
        else:
            shape = tuple(int(s) for s in net.infer_shape(
                **{'stream%d_data' % i: (1, self.num_channel[i], self.num_semg_row, self.num_semg_col)
                   for i in range(self.num_stream) if 'stream%d_data_tag' % i in net.list_attr(recursive=True)})[1][0])
        if shape[1] > 1:
            net = mx.symbol.Reshape(net, shape=(0, 1, -1))
        net = mx.symbol.BatchNorm(
            name=name,
            data=net,
            fix_gamma=False,
            momentum=0.9
        )
        if shape[1] > 1:
            net = mx.symbol.Reshape(name=name + '_restore', data=net, shape=(-1,) + shape[1:])
            #  new_shape = tuple(int(s) for s in net.infer_shape(data=data_shape)[1][0])
            #  assert new_shape == shape
        return net

    def get_bn_relu(self, name, data):
        net = data
        net = self.get_bn(name + '_bn', net)
        net = mx.symbol.Activation(name=name + '_relu', data=net, act_type='relu')
        return net

    def get_pixel_reduce(self, name, net, num_filter, no_bias, rows, cols):
        net = mx.symbol.Reshape(net, shape=(0, 0, -1))

        nets = mx.symbol.SliceChannel(net, num_outputs=rows * cols, axis=2)
        nets = [self.get_fc(name + '_fc%d' % i, nets[i], num_filter, no_bias) for i in range(rows * cols)]
        nets = [mx.symbol.Reshape(p, shape=(0, 0, 1)) for p in nets]
        net = mx.symbol.Concat(*nets, dim=2)

        # net = mx.symbol.SwapAxis(data=net, dim1=1, dim2=2)
        # net = mx.symbol.Reshape(net, shape=(0, -1, 1, 1))
        # num_group = rows * cols
        # net = mx.symbol.Convolution(
            # name=name + '_conv',
            # data=net,
            # num_group=num_group,
            # num_filter=num_filter * num_group,
            # kernel=(1, 1),
            # stride=(1, 1),
            # pad=(0, 0),
            # no_bias=no_bias
        # )
        # net = mx.symbol.Reshape(net, shape=(0, rows * cols, -1))
        # net = mx.symbol.SwapAxis(data=net, dim1=1, dim2=2)

        net = mx.symbol.Reshape(net, shape=(0, 0, rows, cols))
        return net

    def im2col(self, data, name, kernel, pad=(0, 0), stride=(1, 1)):
        shape = self.infer_shape(data)
        return mx.symbol.Convolution(
            name=name,
            data=data,
            num_filter=shape[1] * kernel[0] * kernel[1],
            kernel=kernel,
            stride=stride,
            pad=pad,
            no_bias=True,
            attr={'lr_mult': '0'}
        )

    def get_smooth_pixel_reduce(self, name, net, num_filter, no_bias, rows, cols, kernel=1, stride=1, pad=0):
        if kernel != 1:
            net = self.im2col(name=name + '_im2col', data=net,
                              kernel=(kernel, kernel),
                              pad=(pad, pad),
                              stride=(stride, stride))
            return self.get_smooth_pixel_reduce(name, net, num_filter, no_bias, rows, cols)

        net = mx.symbol.Reshape(net, shape=(0, 0, -1))

        nets = mx.symbol.SliceChannel(net, num_outputs=rows * cols, axis=2)
        W = [[mx.symbol.Variable(name=name + '_fc%d_weight' % (row * cols + col))
              for col in range(cols)] for row in range(rows)]
        nets = [mx.symbol.FullyConnected(name=name + '_fc%d' % i,
                                         data=nets[i],
                                         num_hidden=num_filter,
                                         no_bias=no_bias,
                                         weight=W[i // cols][i % cols])
                for i in range(rows * cols)]
        nets = [mx.symbol.Reshape(p, shape=(0, 0, 1)) for p in nets]
        net = mx.symbol.Concat(*nets, dim=2)
        net = mx.symbol.Reshape(net, shape=(0, 0, rows, cols))

        if self.fast_pixel_reduce:
            lhs, rhs = [], []
            for rs in range(rows):
                for cs in range(cols):
                    for ro, co in [(1, 0), (0, 1)]:
                        rt = rs + ro
                        ct = cs + co
                        if rt < rows and ct < cols:
                            lhs.append(W[rs][cs])
                            rhs.append(W[rt][ct])
            lhs = mx.symbol.Concat(*lhs, dim=0)
            rhs = mx.symbol.Concat(*rhs, dim=0)
            if self.pixel_reduce_norm:
                lhs = mx.symbol.L2Normalization(lhs)
                rhs = mx.symbol.L2Normalization(rhs)
            diff = lhs - rhs
            if self.pixel_reduce_reg_out:
                diff = mx.symbol.sum(diff, axis=1)
            R = mx.symbol.sum(mx.symbol.square(diff))
        else:
            R = []
            for rs in range(rows):
                for cs in range(cols):
                    for ro, co in [(1, 0), (0, 1)]:
                        rt = rs + ro
                        ct = cs + co
                        if rt < rows and ct < cols:
                            R.append(mx.symbol.sum(mx.symbol.square(W[rs][cs] - W[rt][ct])))
            R = mx.symbol.ElementWiseSum(*R)
        loss = mx.symbol.MakeLoss(data=R, grad_scale=self.pixel_reduce_loss_weight)

        return net, loss

    def get_feature(
        self,
        data,
        num_filter,
        num_pixel,
        num_block,
        num_hidden,
        num_bottleneck,
        dropout,
        prefix
    ):
        # from . import symbol_vgg16
        # net = symbol_vgg16.get_symbol(
            # data,
            # stride_downsample=True,
            # batch_norm=True
        # )
        # internals = net.get_internals()
        # net = internals['conv1_2_output']
        # net = get_bn_relu('conv', net)
        # net = get_pixel_reduce('pixel', net, 64, True)
        # net = mx.symbol.Dropout(name='pixel_drop', data=net, p=0.5)
        # net = get_bn_relu('pixel', net)
        # net = get_pixel_reduce('patch', net, 256, True)
        # net = get_bn_relu('patch', net)
        # net = mx.symbol.Dropout(name='patch_drop', data=net, p=0.5)

        # net = data
        # for i in range(2):
            # net = mx.symbol.Convolution(
                # name='conv%d' % (i + 1),
                # data=net,
                # num_filter=64,
                # kernel=(3, 3),
                # stride=(1, 1),
                # pad=(1, 1),
                # no_bias=True
            # )
            # net = get_bn_relu('conv%d' % (i + 1), net)
        # for i in range(5):
            # net = get_pixel_reduce('pixel%d' % (i + 1), net, 64, True)
            # net = get_bn_relu('pixel%d' % (i + 1), net)
        # net = mx.symbol.Dropout(name='drop', data=net, p=0.5)

        # from . import symbol_presnet
        # net = symbol_presnet.get_symbol(
            # data=data,
            # num_level=1,
            # num_block=3,
            # num_filter=64
        # )
        # net = get_pixel_reduce('presnet', net, 64, lr_mult)

        get_act = self.get_bn_relu
        # num_filter = 64
        net = data

        out = {}

        if not self.num_presnet:
            if not self.pool:
                for i in range(self.num_conv):
                    name = prefix + 'conv%d' % (i + 1)
                    net = Convolution(
                        name=name,
                        data=net,
                        num_filter=num_filter,
                        kernel=(3, 3),
                        stride=(1, 1),
                        pad=(1, 1),
                        no_bias=self.no_bias
                    )
                    net = get_act(name, net)
                    out[name] = net
            else:
                for i in range(4):
                    name = prefix + 'conv%d' % (i + 1)
                    net = Convolution(
                        name=name,
                        data=net,
                        num_filter=num_filter,
                        kernel=(3, 3),
                        stride=(1, 1),
                        pad=(1, 1),
                        no_bias=self.no_bias
                    )
                    net = get_act(name, net)
                    out[name] = net
                    net = mx.symbol.Pooling(
                        name=prefix + 'pool%d' % (i + 1),
                        data=net,
                        kernel=(3, 3),
                        stride=(1, 1),
                        pad=(1, 1),
                        pool_type='max'
                    )

            if self.drop_conv:
                net = mx.symbol.Dropout(name=prefix + 'conv_drop', data=net, p=dropout)

        conv = net

        # from . import symbol_vgg16
        # net = symbol_vgg16.get_symbol(net)
        # internals = net.get_internals()
        # net = internals['conv1_2_output']
        # net = get_bn_relu('vgg', net)

        def get_conv(name, net, k3, num_filter, stride):
            return Convolution(
                name=name,
                data=net,
                num_filter=num_filter // 4,
                kernel=(3, 3) if k3 else (1, 1),
                stride=(stride, stride),
                pad=(1, 1) if k3 else (0, 0),
                no_bias=self.no_bias
            )

        def get_branches(net, block, first_act, num_filter, rows, cols, stride):
            act = get_act(prefix + 'block%d_branch1_conv1' % (block + 1), net) if first_act else net
            b1 = get_conv(prefix + 'block%d_branch1_conv1' % (block + 1), act, False, num_filter, 1)

            b2 = get_act(prefix + 'block%d_branch2_conv2' % (block + 1), b1)
            b2 = get_conv(prefix + 'block%d_branch2_conv2' % (block + 1), b2, True, num_filter, stride)

            b3 = get_act(prefix + 'block%d_branch3_conv3' % (block + 1), b2)
            b3 = get_conv(prefix + 'block%d_branch3_conv3' % (block + 1), b3, True, num_filter, 1)

            #  b4 = get_act(prefix + 'block%d_branch4_pixel' % (block + 1), b1)
            #  b4 = self.get_pixel_reduce(
                #  prefix + 'block%d_branch4_pixel' % (block + 1),
                #  b4,
                #  num_filter // 4,
                #  no_bias=self.no_bias,
                #  rows=rows,
                #  cols=cols
            #  )
            return b1, b2, b3

        rows = self.num_semg_row
        cols = self.num_semg_col
        num_local = num_filter

        if self.num_presnet:
            net = Convolution(
                name=prefix + 'stem',
                data=net,
                num_filter=num_filter,
                kernel=(3, 3),
                stride=(1, 1),
                pad=(1, 1),
                no_bias=False
            )
            if self.drop_conv:
                net = mx.symbol.Dropout(name=prefix + 'stem_drop', data=net, p=dropout)

            if isinstance(self.num_presnet, (tuple, list)):
                num_presnet = self.num_presnet
            else:
                num_presnet = [self.num_presnet]
            block = 0
            shortcuts = []
            for level in range(len(num_presnet)):
                for i in range(num_presnet[level]):
                    if i == 0:
                        net = get_act(prefix + 'block%d_pre' % (block + 1), net)

                    shortcut = net

                    if level > 0 and i == 0:
                        stride = 2
                    else:
                        stride = 1

                    num_local = num_filter * 2 ** level
                    if self.presnet_promote:
                        num_local *= 4

                    if (level > 0 or self.presnet_promote) and i == 0:
                        if self.presnet_proj_type == 'A':
                            assert False
                            #  shortcut = Convolution(
                                #  name=prefix + 'block%d_proj' % (block + 1),
                                #  data=shortcut,
                                #  num_filter=num_local,
                                #  kernel=(2, 2),
                                #  stride=(2, 2),
                                #  pad=(0, 0),
                                #  no_bias=True,
                                #  attr={'lr_mult': '0'}
                            #  )
                        elif self.presnet_proj_type == 'B':
                            shortcut = Convolution(
                                name=prefix + 'block%d_proj' % (block + 1),
                                data=shortcut,
                                num_filter=num_local,
                                kernel=(1, 1),
                                stride=(stride, stride),
                                pad=(0, 0),
                                no_bias=False,
                            )
                        else:
                            assert False
                        if self.drop_presnet_proj:
                            shortcut = mx.symbol.Dropout(name=prefix + 'block%d_proj_drop' % (block + 1),
                                                         data=shortcut, p=dropout)
                    rows = self.num_semg_row // 2 ** level
                    cols = self.num_semg_col // 2 ** level
                    branches = get_branches(net, block, i > 0, num_local, rows, cols, stride)
                    if self.presnet_branch:
                        branches = [branches[i] for i in self.presnet_branch]
                    net = mx.symbol.Concat(*branches) if len(branches) > 1 else branches[0]
                    net = get_act(prefix + 'block%d_expand' % (block + 1), net)
                    net = Convolution(
                        name=prefix + 'block%d_expand' % (block + 1),
                        data=net,
                        num_filter=num_local,
                        kernel=(1, 1),
                        stride=(1, 1),
                        pad=(0, 0),
                        no_bias=False
                    )
                    if self.drop_presnet_branch:
                        net = mx.symbol.Dropout(name=prefix + 'block%d_drop' % (block + 1),
                                                data=net, p=dropout)

                    shortcuts.append(shortcut)
                    if not self.presnet_dense or block == 0:
                        net = shortcut + 0.1 * net
                    else:
                        W = [mx.symbol.Variable(name=prefix + 'block%d_dense%d_zero' % (block + 1, i + 1),
                                                shape=(1, 1, 1, 1),
                                                attr={'wd_mult': '0'}) for i in range(block)]
                        W.append(mx.symbol.Variable(name=prefix + 'block%d_dense%d_one' % (block + 1, block + 1),
                                                    shape=(1, 1, 1, 1),
                                                    attr={'wd_mult': '0'}))
                        assert len(W) == len(shortcuts)
                        dense = mx.symbol.ElementWiseSum(*[mx.symbol.broadcast_mul(mx.symbol.broadcast_to(w, shape=(1, num_local, 1, 1)), s) for w, s in zip(W, shortcuts)])
                        net = dense + 0.1 * net
                    block += 1

            net = get_act(prefix + 'presnet', net)
            if self.drop_presnet:
                net = mx.symbol.Dropout(name=prefix + 'presnet_drop', data=net, p=dropout)

        loss = []
        if num_pixel:
            for i in range(num_pixel):
                name = prefix + ('pixel%d' % (i + 1) if num_pixel > 1 else 'pixel')
                rows //= self.pixel_reduce_stride[i]
                cols //= self.pixel_reduce_stride[i]
                ret = self.get_smooth_pixel_reduce(name, net,
                                                   self.num_pixel_reduce_filter[i] or num_local,
                                                   no_bias=not self.pixel_reduce_bias,
                                                   rows=rows, cols=cols,
                                                   kernel=self.pixel_reduce_kernel[i],
                                                   stride=self.pixel_reduce_stride[i],
                                                   pad=self.pixel_reduce_pad[i])
                net = ret[0]
                if self.pixel_reduce_loss_weight > 0:
                    loss.append(ret[1])
                net = get_act(name, net)
                if i in self.drop_pixel:
                    net = Dropout(name=name + '_drop', data=net, p=dropout)
                out[name] = net
            if tuple(self.drop_pixel) == (-1,):
                net = Dropout(name=prefix + 'pixel_drop', data=net, p=dropout)
            if self.conv_shortcut:
                net = mx.symbol.Concat(mx.symbol.Flatten(conv), mx.symbol.Flatten(net), dim=1)
        out['loss'] = loss

        # if not self.num_presnet and not num_pixel:
            # net = mx.symbol.Dropout(name=prefix + 'drop', data=net, p=dropout)

        # net = mx.symbol.Dropout(name=prefix + 'drop', data=net, p=0.5)
        # net = mx.symbol.Pooling(data=net, kernel=(3, 3), pool_type='avg')
        for i in range(num_block):
            name = prefix + 'fc%d' % (i + 1)
            net = self.get_fc(name, net, num_hidden, no_bias=self.no_bias)
            net = get_act(name, net)
            net = Dropout(
                name=name + '_drop',
                data=net,
                p=dropout
            )
            out[name] = net

        # net = get_fc(prefix + 'fc', net, num_hidden, no_bias=True)
        # net = get_act(prefix + 'fc', net)
        # net = mx.symbol.Dropout(name=prefix + 'fc_drop', data=net, p=0.5)
        # net = get_presnet(
            # prefix + 'presnet',
            # net,
            # num_block,
            # lambda name, net: mx.symbol.Dropout(
                # name=name + '_drop',
                # data=get_fc(name, net, num_hidden, no_bias=True),
                # p=0.5
            # )
        # )
        # net = mx.symbol.Dropout(name=prefix + 'presnet_drop', data=net, p=0.5)

        net = self.get_fc(prefix + 'bottleneck', net, num_bottleneck, no_bias=self.no_bias)
        net = get_act(prefix + 'bottleneck', net)
        out[prefix + 'bottleneck'] = net

        return out

    def get_fc(self, name, data, num_hidden, no_bias=False):
        return mx.symbol.FullyConnected(
            name=name,
            data=data,
            num_hidden=num_hidden,
            no_bias=no_bias
        )

    def get_fc_bn_relu(self, name, data, num_hidden):
        net = self.get_fc(name=name, data=data, num_hidden=num_hidden, no_bias=self.no_bias)
        net = self.get_bn_relu(name, net)
        return net

    def get_fc_bn_relu_drop(self, name, data, num_hidden):
        net = self.get_fc(name=name, data=data, num_hidden=num_hidden, no_bias=self.no_bias)
        net = self.get_bn_relu(name, net)
        net = Dropout(name=name + '_drop', data=net, p=0.5)
        return net

    def get_presnet(self, name, net, num_block, get_trans, scale=1):
        if not isinstance(get_trans, list):
            get_trans = [get_trans, get_trans]

        for i in range(num_block):
            shortcut = net
            trans1_name = name + '_block%d_trans1' % (i + 1)
            if i > 0:
                net = self.get_bn_relu(trans1_name, net)
            net = get_trans[0](trans1_name, net)
            trans2_name = name + '_block%d_trans2' % (i + 1)
            net = self.get_bn_relu(trans2_name, net)
            net = get_trans[1](trans2_name, net)
            if scale == 1:
                net = shortcut + net
            else:
                net = shortcut + scale * net
        net = self.get_bn_relu(name, net)
        return net

    def get_branch(
        self,
        name,
        data,
        num_class,
        num_block,
        num_hidden,
        return_fc=False,
        fc_attr={},
        **kargs
    ):
        net = data
        if num_block and num_hidden:
            # net = get_presnet(
                # name + '_presnet',
                # net,
                # num_block,
                # [lambda name, net: get_fc(name, net, num_hidden // 4, no_bias=True),
                #  lambda name, net: get_fc(name, net, num_hidden, no_bias=True)],
                # scale=0.01
            # )
            for i in range(num_block):
                # net = self.get_fc(name + '_fc%d' % (i + 1), net, num_hidden, no_bias=self.no_bias)
                net = mx.symbol.FullyConnected(
                    name=name + '_fc%d' % (i + 1),
                    data=net,
                    num_hidden=num_hidden,
                    no_bias=self.no_bias,
                    attr=dict(**fc_attr)
                )
                net = self.get_bn_relu(name + '_fc%d' % (i + 1), net)
                if self.drop_branch:
                    net = Dropout(
                        name=name + '_fc%d_drop' % (i + 1),
                        data=net,
                        p=0.5
                    )
        # net = self.get_fc(name=name + '_last_fc', data=net, num_hidden=num_class)
        net = mx.symbol.FullyConnected(
            name=name + '_last_fc',
            data=net,
            num_hidden=num_class,
            no_bias=False,
            attr=dict(**fc_attr)
        )
        fc = net
        if self.tzeng:
            net = mx.symbol.Custom(data=net, name=name + '_gradscale', op_type='GradScale')
        if self.for_training:
            net = mx.symbol.SoftmaxOutput(name=name + '_softmax', data=net, **kargs)
        else:
            net = mx.symbol.SoftmaxActivation(name=name + '_softmax', data=net)
        return (net, fc) if return_fc else net

    def __init__(
        self,
        num_gesture,
        num_subject,
        num_filter=constant.NUM_FILTER,
        num_pixel=constant.NUM_PIXEL,
        num_hidden=constant.NUM_HIDDEN,
        num_bottleneck=constant.NUM_BOTTLENECK,
        num_feature_block=constant.NUM_FEATURE_BLOCK,
        num_gesture_block=constant.NUM_GESTURE_BLOCK,
        num_subject_block=constant.NUM_SUBJECT_BLOCK,
        dropout=constant.DROPOUT,
        coral=False,
        num_channel=1,
        revgrad=False,
        tzeng=False,
        num_presnet=0,
        presnet_branch=None,
        drop_presnet=False,
        bng=False,
        soft_label=False,
        minibatch=False,
        confuse_conv=False,
        confuse_all=False,
        subject_wd=None,
        drop_branch=False,
        pool=False,
        zscore=True,
        zscore_bng=False,
        output=None,
        num_stream=1,
        lstm=False,
        num_lstm_hidden=constant.NUM_LSTM_HIDDEN,
        num_lstm_layer=constant.NUM_LSTM_LAYER,
        lstm_last=0,
        lstm_dropout=constant.LSTM_DROPOUT,
        lstm_shortcut=False,
        lstm_bn=True,
        lstm_window=None,
        lstm_grad_scale=True,
        for_training=False,
        presnet_promote=False,
        faug=0,
        drop_conv=False,
        drop_presnet_branch=False,
        drop_presnet_proj=False,
        presnet_proj_type='A',
        bn_wd_mult=0,
        pixel_reduce_loss_weight=0,
        pixel_reduce_bias=False,
        pixel_reduce_kernel=1,
        pixel_reduce_stride=1,
        pixel_reduce_pad=0,
        pixel_reduce_norm=False,
        pixel_reduce_reg_out=False,
        num_pixel_reduce_filter=None,
        fast_pixel_reduce=True,
        num_conv=2,
        drop_pixel=(-1,),
        presnet_dense=False,
        conv_shortcut=False,
        num_semg_row=constant.NUM_SEMG_ROW,
        num_semg_col=constant.NUM_SEMG_COL,
        return_bottleneck=False,
        **kargs
    ):
        self.num_semg_row = num_semg_row
        self.num_semg_col = num_semg_col
        self.conv_shortcut = conv_shortcut
        self.presnet_dense = presnet_dense
        self.drop_pixel = drop_pixel
        self.num_conv = num_conv
        self.for_training = for_training
        self.num_channel = num_channel
        self.num_subject = num_subject
        self.num_presnet = num_presnet
        self.presnet_branch = presnet_branch
        self.drop_presnet = drop_presnet
        self.no_bias = True
        self.bng = bng
        self.tzeng = tzeng
        self.minibatch = minibatch
        self.drop_branch = drop_branch
        self.pool = pool
        self.num_stream = num_stream
        self.lstm = lstm
        self.lstm_bn = lstm_bn
        self.lstm_window = lstm_window
        self.drop_conv = drop_conv
        self.drop_presnet_branch = drop_presnet_branch
        self.drop_presnet_proj = drop_presnet_proj
        self.presnet_proj_type = presnet_proj_type
        self.bn_wd_mult = bn_wd_mult
        self.presnet_promote = presnet_promote
        self.pixel_reduce_loss_weight = pixel_reduce_loss_weight
        self.pixel_reduce_bias = pixel_reduce_bias
        if not isinstance(pixel_reduce_kernel, (list, tuple)):
            pixel_reduce_kernel = [pixel_reduce_kernel for _ in range(num_pixel)]
        self.pixel_reduce_kernel = pixel_reduce_kernel
        if not isinstance(pixel_reduce_stride, (list, tuple)):
            pixel_reduce_stride = [pixel_reduce_stride for _ in range(num_pixel)]
        self.pixel_reduce_stride = pixel_reduce_stride
        if not isinstance(pixel_reduce_pad, (list, tuple)):
            pixel_reduce_pad = [pixel_reduce_pad for _ in range(num_pixel)]
        self.pixel_reduce_pad = pixel_reduce_pad
        self.pixel_reduce_norm = pixel_reduce_norm
        self.pixel_reduce_reg_out = pixel_reduce_reg_out
        if not isinstance(num_pixel_reduce_filter, (list, tuple)):
            num_pixel_reduce_filter = [num_pixel_reduce_filter for _ in range(num_pixel)]
        self.num_pixel_reduce_filter = num_pixel_reduce_filter
        self.fast_pixel_reduce = fast_pixel_reduce

        def get_stream(prefix):
            if prefix:
                net = mx.symbol.Variable(name=prefix + 'data', attr={'tag': '1'})
            else:
                net = mx.symbol.Variable(name=prefix + 'data')

            if self.lstm:
                assert not zscore_bng
                if self.num_channel > 1:
                    net = mx.symbol.Reshape(net, shape=(0, -1, self.num_channel, self.num_semg_row, self.num_semg_col))
                net = mx.symbol.SwapAxis(net, dim1=0, dim2=1)
                net = mx.symbol.Reshape(net, shape=(-1, self.num_channel, self.num_semg_row, self.num_semg_col))

            if zscore:
                net = (self.get_bng if zscore_bng else self.get_bn)(prefix + 'zscore', net)
                shortcut = net

            features = self.get_feature(
                data=net,
                num_filter=num_filter,
                num_pixel=num_pixel,
                num_block=num_feature_block,
                num_hidden=num_hidden,
                num_bottleneck=num_bottleneck,
                dropout=dropout,
                prefix=prefix
            )

            if zscore:
                features[prefix + 'shortcut'] = shortcut

            return features[prefix + 'bottleneck'], features

        if num_stream == 1:
            feature, features = get_stream('')
            loss = features['loss']
        else:
            feature = mx.symbol.Concat(*[get_stream('stream%d_' % i)[0] for i in range(num_stream)])
            features = None
            loss = []

        bottleneck = feature

        if self.lstm:
            feature = mx.symbol.Reshape(feature, shape=(self.lstm_window, -1, num_bottleneck))
            feature = mx.symbol.SwapAxis(feature, dim1=0, dim2=1)

            if lstm_shortcut:
                shortcut = features['shortcut']
                shortcut = mx.symbol.Reshape(shortcut, shape=(self.lstm_window, -1, constant.NUM_SEMG_POINT))
                shortcut = mx.symbol.SwapAxis(shortcut, dim1=0, dim2=1)
                feature = mx.symbol.Concat(shortcut, feature, dim=2)

            from .lstm import lstm_unroll
            feature = lstm_unroll(
                prefix='',
                data=feature,
                num_lstm_layer=num_lstm_layer,
                seq_len=self.lstm_window,
                num_hidden=num_lstm_hidden,
                dropout=lstm_dropout,
                minibatch=self.minibatch,
                num_subject=self.num_subject,
                bn=self.lstm_bn
            )
            if lstm_last == 1:
                feature = feature[-1]
            elif lstm_last < 0:
                feature = mx.sym.Pooling(
                    mx.sym.Concat(*[mx.sym.Reshape(h, shape=(0, 0, 1, 1)) for h in feature[lstm_last:]], dim=3),
                    kernel=(1, 1),
                    global_pool=True,
                    pool_type='avg'
                )
            elif lstm_last:
                feature = mx.symbol.Concat(*feature[-lstm_last:], dim=0)
            else:
                feature = mx.symbol.Concat(*feature, dim=0)

        if coral:
            feature = mx.symbol.FullyConnected(
                name='coral',
                data=feature,
                num_hidden=num_bottleneck,
                no_bias=False
            )

        if faug:
            feature_before_faug = feature
            feature = feature + mx.symbol.Variable('faug')

        gesture_branch_kargs = {}
        if self.lstm and (lstm_last == 0 or lstm_last > 1):
            gesture_label = mx.symbol.Variable(name='gesture_softmax_label')
            gesture_label = mx.symbol.Reshape(mx.symbol.Concat(*[
                mx.symbol.Reshape(gesture_label, shape=(0, 1))
                for i in range(lstm_last or self.lstm_window)], dim=0), shape=(-1,))
            gesture_branch_kargs['label'] = gesture_label
            if lstm_grad_scale:
                gesture_branch_kargs['grad_scale'] = 1 / (lstm_last or self.lstm_window)

        gesture_softmax, gesture_fc = self.get_branch(
            name='gesture',
            data=feature,
            num_class=num_gesture,
            num_block=num_gesture_block,
            num_hidden=num_bottleneck,
            use_ignore=True,
            return_fc=True,
            # grad_scale=0.1 if soft_label else 1
            **gesture_branch_kargs
        )

        loss.insert(0, gesture_softmax)

        if soft_label:
            net = mx.symbol.SoftmaxActivation(gesture_fc / 10)
            net = mx.symbol.log(net)
            net = mx.symbol.broadcast_mul(net, mx.symbol.Variable('soft_label'))
            net = -symsum(data=net)
            net = mx.symbol.MakeLoss(data=net, grad_scale=0.1)
            loss.append(net)

        if (revgrad or tzeng) and num_subject > 0:
            assert not (confuse_conv and confuse_all)
            if confuse_conv:
                feature = features['conv2']
            if confuse_all:
                feature = mx.symbol.Concat(*[mx.symbol.Flatten(features[name]) for name in sorted(features)])

            if revgrad:
                feature = mx.symbol.Custom(data=feature, name='grl', op_type='GRL')
            else:
                feature = mx.symbol.Custom(data=feature, name='bottleneck_gradscale', op_type='GradScale')
            subject_softmax_loss, subject_fc = self.get_branch(
                name='subject',
                data=feature,
                num_class=num_subject,
                num_block=num_subject_block,
                num_hidden=num_bottleneck,
                # Ganin et al. use 0.1 in their code
                # https://github.com/ddtm/caffe/commit/7a2c1967b4ec54d771d1e746887be4b07a7a4975
                grad_scale=kargs.pop('subject_loss_weight', 0.1),
                use_ignore=True,
                return_fc=True,
                fc_attr={'wd_mult': str(subject_wd)} if subject_wd is not None else {}
            )
            loss.append(subject_softmax_loss)
            if tzeng:
                subject_fc = mx.symbol.Custom(data=subject_fc, name='subject_confusion_gradscale', op_type='GradScale')
                subject_softmax = mx.symbol.SoftmaxActivation(subject_fc)
                subject_confusion_loss = mx.symbol.MakeLoss(
                    data=-symsum(data=mx.symbol.log(subject_softmax + 1e-8)) / num_subject,
                    grad_scale=kargs.pop('subject_confusion_loss_weight', 0.1)
                )
                loss.append(subject_confusion_loss)
            # subject_softmax = mx.symbol.LogisticRegressionOutput(
                # name='subject_softmax',
                # data=subject_fc,
                # grad_scale=kargs.pop('subject_loss_weight', None) or 0.1
            # )

        target_gesture_loss_weight = kargs.get('target_gesture_loss_weight')
        if target_gesture_loss_weight is not None:
            loss.append(mx.symbol.SoftmaxOutput(
                name='target_gesture_softmax',
                data=gesture_fc,
                grad_scale=target_gesture_loss_weight,
                use_ignore=True
            ))

        if output is None:
            self.net = loss[0] if len(loss) == 1 else mx.sym.Group(loss)
        else:
            assert_equal(len(loss), 1)
            self.net = loss[0].get_internals()[output]

        if return_bottleneck:
            self.net = bottleneck

        self.net.num_semg_row = num_semg_row
        self.net.num_semg_col = num_semg_col
        self.net.num_presnet = num_presnet
        self.net.num_channel = num_channel
        if self.lstm:
            self.net.data_shape_1 = self.num_channel * self.lstm_window
        else:
            self.net.data_shape_1 = self.num_channel

        if self.lstm:
            self.net.num_feature = num_lstm_hidden
        else:
            self.net.num_feature = num_bottleneck

        if faug:
            self.net.feature_before_faug = feature_before_faug

        self.net.presnet_proj_type = presnet_proj_type

        if self.lstm:
            assert_equal(self.num_stream, 1)

            net = mx.symbol.Variable(name='data')
            net = self.get_bn('zscore', net)
            if lstm_shortcut:
                shortcut = net
                shortcut = mx.symbol.Reshape(shortcut, shape=(-1, constant.NUM_SEMG_POINT))
            feature = self.get_feature(
                data=net,
                num_filter=num_filter,
                num_pixel=num_pixel,
                num_block=num_feature_block,
                num_hidden=num_hidden,
                num_bottleneck=num_bottleneck,
                dropout=dropout,
                prefix=''
            )['bottleneck']
            if lstm_shortcut:
                feature = mx.symbol.Concat(shortcut, feature, dim=1)
            self.net.bottleneck = feature

            net = mx.symbol.Variable(name='data')
            from .lstm import lstm_unroll
            net = lstm_unroll(
                prefix='',
                data=net,
                num_lstm_layer=num_lstm_layer,
                seq_len=self.lstm_window,
                num_hidden=num_lstm_hidden,
                dropout=lstm_dropout,
                minibatch=self.minibatch,
                num_subject=self.num_subject,
                bn=self.lstm_bn
            )
            if lstm_last == 1:
                net = net[-1]
            elif lstm_last < 0:
                net = mx.sym.Pooling(
                    mx.sym.Concat(*[mx.sym.Reshape(h, shape=(0, 0, 1, 1)) for h in net[lstm_last:]], dim=3),
                    kernel=(1, 1),
                    global_pool=True,
                    pool_type='avg'
                )
            elif lstm_last:
                net = mx.sym.Concat(*net[-lstm_last:], dim=0)
            else:
                net = mx.sym.Concat(*net, dim=0)
            net = mx.symbol.FullyConnected(
                name='gesture_last_fc',
                data=net,
                num_hidden=num_gesture,
                no_bias=False
            )
            net = mx.symbol.SoftmaxActivation(net)
            if lstm_last == 0 or lstm_last > 1:
                net = mx.sym.Reshape(net, shape=((lstm_last or self.lstm_window), -1, num_gesture))
                net = mx.symbol.sum(net, axis=0) / (lstm_last or self.lstm_window)
                net = mx.sym.Reshape(net, shape=(-1, num_gesture))
            self.net.lstm = net
            #  self.net.lstm = self.get_branch(
                #  name='gesture',
                #  data=net,
                #  num_class=num_gesture,
                #  num_block=num_gesture_block,
                #  num_hidden=num_bottleneck,
                #  use_ignore=True
            #  )

            self.net.lstm_window = self.lstm_window
            self.net.num_lstm_hidden = num_lstm_hidden
            self.net.num_lstm_layer = num_lstm_layer
            self.net.lstm_bn = self.lstm_bn


def symsum(data):
    return mx.symbol.sum(data, axis=1)
    # return mx.symbol.FullyConnected(
        # name='sum',
        # data=data,
        # num_hidden=1,
        # no_bias=True,
        # attr={'lr_mult': '0', 'wd_mult': '0'}
    # )


def get_symbol(*args, **kargs):
    return Symbol(*args, **kargs).net


def Dropout(**kargs):
    p = kargs.pop('p')
    return kargs.pop('data') if p == 0 else mx.symbol.Dropout(p=p, **kargs)


def Convolution(*args, **kargs):
    kargs['cudnn_tune'] = 'fastest'
    return mx.symbol.Convolution(*args, **kargs)
