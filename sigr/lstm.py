from collections import namedtuple
import mxnet as mx


LSTMState = namedtuple("LSTMState", ["c", "h"])
#  LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_bias", "h2h_weight", "h2h_bias"])
LSTMParam = namedtuple("LSTMParam", ["i2h_weight", "i2h_gamma", "h2h_weight", "h2h_gamma",
                                     "beta", "c_gamma", "c_beta"])
LSTMModel = namedtuple("LSTMModel", ["rnn_exec", "symbol",
                                     "init_states", "last_states",
                                     "seq_data", "seq_labels", "seq_outputs",
                                     "param_blocks"])


class LSTM(object):

    def lstm_orig(self, prefix, num_hidden, indata, prev_state, param, seqidx, layeridx, dropout=0.):
        """LSTM Cell symbol"""
        if dropout > 0.:
            indata = mx.sym.Dropout(data=indata, p=dropout)
        i2h = mx.sym.FullyConnected(data=indata,
                                    weight=param.i2h_weight,
                                    bias=param.i2h_bias,
                                    num_hidden=num_hidden * 4,
                                    name=prefix + "t%d_l%d_i2h" % (seqidx, layeridx))
        h2h = mx.sym.FullyConnected(data=prev_state.h,
                                    weight=param.h2h_weight,
                                    bias=param.h2h_bias,
                                    num_hidden=num_hidden * 4,
                                    name=prefix + "t%d_l%d_h2h" % (seqidx, layeridx))
        gates = i2h + h2h
        slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                          name=prefix + "t%d_l%d_slice" % (seqidx, layeridx))
        in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
        in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
        forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
        out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")
        next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
        next_h = out_gate * mx.sym.Activation(next_c, act_type="tanh")
        return LSTMState(c=next_c, h=next_h)

    def lstm_not_share_beta_gamma(self, prefix, num_hidden, indata, prev_state, param, seqidx, layeridx, dropout=0.):
        """LSTM Cell symbol"""
        if dropout > 0.:
            indata = mx.sym.Dropout(data=indata, p=dropout)
        i2h = mx.sym.FullyConnected(data=indata,
                                    weight=param.i2h_weight,
                                    bias=param.i2h_bias,
                                    num_hidden=num_hidden * 4,
                                    name=prefix + "t%d_l%d_i2h" % (seqidx, layeridx))
        i2h = mx.sym.BatchNorm(
            name=prefix + "t%d_l%d_i2h_bn" % (seqidx, layeridx),
            data=i2h,
            fix_gamma=False,
            momentum=0.9,
            attr={'wd_mult': '0'}
        )
        h2h = mx.sym.FullyConnected(data=prev_state.h,
                                    weight=param.h2h_weight,
                                    bias=param.h2h_bias,
                                    num_hidden=num_hidden * 4,
                                    name=prefix + "t%d_l%d_h2h" % (seqidx, layeridx))
        h2h = mx.sym.BatchNorm(
            name=prefix + "t%d_l%d_h2h_bn" % (seqidx, layeridx),
            data=h2h,
            fix_gamma=False,
            momentum=0.9,
            attr={'wd_mult': '0'}
        )
        gates = i2h + h2h
        slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                          name=prefix + "t%d_l%d_slice" % (seqidx, layeridx))
        in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
        in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
        forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
        out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")
        next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
        next_h = out_gate * mx.sym.Activation(
            mx.symbol.BatchNorm(
                name=prefix + 't%d_l%d_c_bn' % (seqidx, layeridx),
                data=next_c,
                fix_gamma=False,
                momentum=0.9,
                attr={'wd_mult': '0'}
            ),
            act_type="tanh"
        )
        return LSTMState(c=next_c, h=next_h)

    def lstm(self, prefix, num_hidden, indata, prev_state, param, seqidx, layeridx, dropout=0.):
        """LSTM Cell symbol"""
        if dropout > 0.:
            indata = mx.sym.Dropout(data=indata, p=dropout)
        i2h = mx.sym.FullyConnected(data=indata,
                                    weight=param.i2h_weight,
                                    num_hidden=num_hidden * 4,
                                    no_bias=True,
                                    name=prefix + "t%d_l%d_i2h" % (seqidx, layeridx))
        i2h = self.BatchNorm(
            name=prefix + "t%d_l%d_i2h_bn" % (seqidx, layeridx),
            data=i2h,
            gamma=param.i2h_gamma,
            num_channel=num_hidden * 4
        )
        h2h = mx.sym.FullyConnected(data=prev_state.h,
                                    weight=param.h2h_weight,
                                    num_hidden=num_hidden * 4,
                                    no_bias=True,
                                    name=prefix + "t%d_l%d_h2h" % (seqidx, layeridx))
        h2h = self.BatchNorm(
            name=prefix + "t%d_l%d_h2h_bn" % (seqidx, layeridx),
            data=h2h,
            gamma=param.h2h_gamma,
            beta=param.beta,
            num_channel=num_hidden * 4
        )
        gates = i2h + h2h
        slice_gates = mx.sym.SliceChannel(gates, num_outputs=4,
                                          name=prefix + "t%d_l%d_slice" % (seqidx, layeridx))
        in_gate = mx.sym.Activation(slice_gates[0], act_type="sigmoid")
        in_transform = mx.sym.Activation(slice_gates[1], act_type="tanh")
        forget_gate = mx.sym.Activation(slice_gates[2], act_type="sigmoid")
        out_gate = mx.sym.Activation(slice_gates[3], act_type="sigmoid")
        next_c = (forget_gate * prev_state.c) + (in_gate * in_transform)
        next_h = out_gate * mx.sym.Activation(
            self.BatchNorm(
                name=prefix + 't%d_l%d_c_bn' % (seqidx, layeridx),
                data=next_c,
                gamma=param.c_gamma,
                beta=param.c_beta,
                num_channel=num_hidden
            ),
            act_type="tanh"
        )
        return LSTMState(c=next_c, h=next_h)

    def BatchNorm(self, name, data, gamma, beta=None, **kargs):
        net = data

        if not self.bn:
            return net

        if self.minibatch:
            num_channel = kargs.pop('num_channel')
            net = mx.symbol.Reshape(net, shape=(-1, self.num_subject * num_channel))
            net = mx.symbol.BatchNorm(
                name=name + '_norm',
                data=net,
                fix_gamma=True,
                momentum=0.9,
                attr={'wd_mult': '0', 'lr_mult': '0'}
            )
            net = mx.symbol.Reshape(data=net, shape=(-1, num_channel))
        else:
            net = mx.symbol.BatchNorm(
                name=name + '_norm',
                data=net,
                fix_gamma=True,
                momentum=0.9,
                attr={'wd_mult': '0', 'lr_mult': '0'}
            )
        net = mx.symbol.broadcast_mul(net, gamma)
        if beta is not None:
            net = mx.symbol.broadcast_plus(net, beta)
        return net

    def __init__(
        self,
        prefix,
        data,
        num_lstm_layer,
        seq_len,
        num_hidden,
        dropout=0.,
        minibatch=False,
        num_subject=0,
        bn=True,
    ):
        self.bn = bn
        self.minibatch = minibatch
        self.num_subject = num_subject
        if self.minibatch:
            assert self.num_subject > 0

        prefix += 'lstm_'

        param_cells = []
        last_states = []
        for i in range(num_lstm_layer):
            param_cells.append(LSTMParam(i2h_weight=mx.sym.Variable(prefix + "l%d_i2h_weight" % i),
                                         #  i2h_bias=mx.sym.Variable(prefix + "l%d_i2h_bias" % i),
                                         h2h_weight=mx.sym.Variable(prefix + "l%d_h2h_weight" % i),
                                         #  h2h_bias=mx.sym.Variable(prefix + "l%d_h2h_bias" % i)))
                                         i2h_gamma=mx.symbol.Variable(prefix + 'l%d_i2h_gamma' % i, shape=(1, num_hidden * 4), attr={'wd_mult': '0'}),
                                         h2h_gamma=mx.symbol.Variable(prefix + 'l%d_h2h_gamma' % i, shape=(1, num_hidden * 4), attr={'wd_mult': '0'}),
                                         beta=mx.symbol.Variable(prefix + 'l%d_beta' % i, shape=(1, num_hidden * 4), attr={'wd_mult': '0'}),
                                         c_gamma=mx.symbol.Variable(prefix + 'l%d_c_gamma' % i, shape=(1, num_hidden), attr={'wd_mult': '0'}),
                                         c_beta=mx.symbol.Variable(prefix + 'l%d_c_beta' % i, shape=(1, num_hidden), attr={'wd_mult': '0'})))
            state = LSTMState(c=mx.sym.Variable(prefix + "l%d_init_c" % i, attr={'lr_mult': '0'}),
                              h=mx.sym.Variable(prefix + "l%d_init_h" % i, attr={'lr_mult': '0'}))
            last_states.append(state)
        assert(len(last_states) == num_lstm_layer)

        wordvec = mx.sym.SliceChannel(data=data, num_outputs=seq_len, squeeze_axis=1)

        hidden_all = []
        for seqidx in range(seq_len):
            hidden = wordvec[seqidx]

            # stack LSTM
            for i in range(num_lstm_layer):
                if i == 0:
                    dp_ratio = 0.
                else:
                    dp_ratio = dropout
                next_state = self.lstm(prefix, num_hidden, indata=hidden,
                                       prev_state=last_states[i],
                                       param=param_cells[i],
                                       seqidx=seqidx, layeridx=i, dropout=dp_ratio)
                hidden = next_state.h
                last_states[i] = next_state

            # decoder
            if dropout > 0.:
                hidden = mx.sym.Dropout(data=hidden, p=dropout)
            hidden_all.append(hidden)

        self.net = hidden_all
        #  return mx.sym.Concat(*hidden_all, dim=1)
        #  return mx.sym.Pooling(mx.sym.Concat(*[mx.sym.Reshape(h, shape=(0, 0, 1, 1)) for h in hidden_all], dim=2), kernel=(1, 1), global_pool=True, pool_type='max')
        #  return mx.sym.Pooling(mx.sym.Concat(*[mx.sym.Reshape(h, shape=(0, 0, 1, 1)) for h in hidden_all], dim=2), kernel=(1, 1), global_pool=True, pool_type='avg')


def lstm_unroll(**kargs):
    return LSTM(**kargs).net
