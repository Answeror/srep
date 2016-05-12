/*!
 *  Copyright (c) 2016 by Contributors
 * \file elementwise_binary_broadcast_op-inl.h
 * \brief Function defintion of elementwise binary operators with broadcast
 *
 * For example,
 *
 *   [1, 2] + [1,  = [2, 3;
 *             2 ]    3, 4]
 *
 * The shapes broacast of the above example
 *
 *   A      (2d tensor):  1 x 2
 *   B      (1d tensor):  2 x 1
 *   Result (2d tensor):  2 x 2
 *
 * More examples
 *
 *   A      (3d tensor):  15 x 3 x 5
 *   B      (2d tensor):   1 x 1 x 5
 *   Result (3d tensor):  15 x 3 x 5
 *
 *   A      (3d tensor):  15 x 3 x 5
 *   B      (2d tensor):   1 x 3 x 1
 *   Result (3d tensor):  15 x 3 x 5
 *
 * Here are examples of shapes that do not broadcast:
 *
 *   A      (1d tensor):  3
 *   B      (1d tensor):  4 # trailing dimensions do not match
 *
 *   A      (2d tensor):  1 x 2 x 1
 *   B      (3d tensor):  8 x 4 x 3 # second from last dimensions mismatched
 *
 * When no broadcast is need, it falls back to elementwise_binary_op-inl.h
 */
#ifndef MXNET_OPERATOR_ELEMENTWISE_BINARY_BROADCAST_OP_INL_H_
#define MXNET_OPERATOR_ELEMENTWISE_BINARY_BROADCAST_OP_INL_H_

#include <mxnet/operator_util.h>
#include <algorithm>
#include <vector>
#include "./mshadow_op.h"
#include "./broadcast_reduce_op_common.h"

#if defined(__CUDACC__)
#define XPU gpu
#else
#define XPU cpu
#endif

namespace mxnet {
namespace op {
namespace broadcast_detail {

using namespace mshadow;
using namespace mshadow::expr;

template<typename TrueExp, typename FalseExp, typename DType>
struct ConditionalExp : Exp<ConditionalExp<TrueExp, FalseExp, DType>, DType, type::kChainer> {
  const TrueExp &true_exp_;
  const FalseExp &false_exp_;
  bool condition_;
  ConditionalExp(const TrueExp &true_exp, const FalseExp &false_exp, bool condition)
    : true_exp_(true_exp), false_exp_(false_exp), condition_(condition) {}
};

template<typename TrueExp, typename FalseExp, typename DType, int truetype, int falsetype>
inline ConditionalExp<TrueExp, FalseExp, DType>
conditional(const Exp<TrueExp, DType, truetype> &true_exp,
            const Exp<FalseExp, DType, falsetype> &false_exp,
            bool condition) {
  return ConditionalExp<TrueExp, FalseExp, DType>(true_exp.self(), false_exp.self(), condition);
}

}
}
}

namespace mshadow {
namespace expr {

template<typename TrueExp, typename FalseExp, typename DType>
struct Plan<mxnet::op::broadcast_detail::ConditionalExp<TrueExp, FalseExp, DType>, DType> {
 public:
  explicit Plan(const mxnet::op::broadcast_detail::ConditionalExp<TrueExp, FalseExp, DType> &e)
    : true_exp_(MakePlan(e.true_exp_)),
      false_exp_(MakePlan(e.false_exp_)),
      condition_(e.condition_) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    return condition_ ? true_exp_.Eval(i, j) : false_exp_.Eval(i, j);
  }

 private:
  Plan<TrueExp, DType> true_exp_;
  Plan<FalseExp, DType> false_exp_;
  bool condition_;
};

template<typename TrueExp, typename FalseExp, typename DType>
inline Plan<mxnet::op::broadcast_detail::ConditionalExp<TrueExp, FalseExp, DType>, DType>
MakePlan(const mxnet::op::broadcast_detail::ConditionalExp<TrueExp, FalseExp, DType> &e) {
  return Plan<mxnet::op::broadcast_detail::ConditionalExp<TrueExp, FalseExp, DType>, DType>(e);
}

template<typename TrueExp, typename FalseExp, typename DType>
struct ExpInfo<mxnet::op::broadcast_detail::ConditionalExp<TrueExp, FalseExp, DType> >
  : ExpInfo<TrueExp> {};

template<int dim, typename TrueExp, typename FalseExp, typename DType>
struct ShapeCheck<dim, mxnet::op::broadcast_detail::ConditionalExp<TrueExp, FalseExp, DType> > {
  inline static Shape<dim> Check(const mxnet::op::broadcast_detail::ConditionalExp<TrueExp, FalseExp, DType> &e) {
    return e.condition_ ? ShapeCheck<dim, TrueExp>::Check(e.true_exp_) : ShapeCheck<dim, FalseExp>::Check(e.false_exp_);
  }
};

}
}

namespace mxnet {
namespace op {

inline bool IsBroadcastNeeded_(const TShape& lhs,
                              const TShape& rhs) {
  // force ndim to be equal. do not smartly padding dims with 1s, which may confuse users
  CHECK_EQ(lhs.ndim(), rhs.ndim());
  for (index_t i = 0; i < lhs.ndim(); ++i) {
    if (lhs[i] != rhs[i]) return true;
  }
  return false;
}

inline TShape BinaryBroadcastShape_(const TShape& lhs,
                                    const TShape& rhs,
                                    const EnvArguments& env) {
  if (!IsBroadcastNeeded_(lhs, rhs)) return lhs;
  std::vector<index_t> ret(lhs.ndim());
  for (size_t i = 0; i < ret.size(); ++i) {
    ret[i] = std::max(lhs[i], rhs[i]);
  }
  return TShape(ret.begin(), ret.end());
}

inline void InferBroadcastNewShapes_(bool *do_opt,
  TShape *new_lhs_shape, TShape *new_rhs_shape, TShape *new_out_shape,
  const TShape &lhs_shape, const TShape &rhs_shape, const TShape &out_shape) {
  using namespace mshadow;
  using namespace mshadow::expr;
  CHECK((lhs_shape.ndim() == rhs_shape.ndim()) && (rhs_shape.ndim() == out_shape.ndim())) <<
    "ndim inconsistency, lhs_shape=" << lhs_shape << ", rhs_shape=" << rhs_shape <<
    ", out_shape=" << out_shape;
  *do_opt = false;
  TShape lhs_axes = GetBroadcastingAxes_(lhs_shape, out_shape);
  TShape rhs_axes = GetBroadcastingAxes_(rhs_shape, out_shape);
  bool lhs_contiguous, rhs_contiguous;
  index_t lhs_broadcasting_size, rhs_broadcasting_size;
  CheckContiguousAxes_(&lhs_contiguous, &lhs_broadcasting_size, lhs_axes, out_shape);
  CheckContiguousAxes_(&rhs_contiguous, &rhs_broadcasting_size, rhs_axes, out_shape);
  if (lhs_contiguous && rhs_contiguous && (lhs_axes.ndim() == 0 || rhs_axes.ndim() == 0)) {
    *do_opt = true;
    if (lhs_axes.ndim() == 0) {
      index_t leading =
        rhs_shape.ProdShape(0, rhs_axes[0]);
      index_t trailing =
        rhs_shape.ProdShape(rhs_axes[rhs_axes.ndim() - 1] + 1, rhs_shape.ndim());
      *new_lhs_shape = Shape3(leading, rhs_broadcasting_size, trailing);
      *new_rhs_shape = Shape3(leading, 1, trailing);
      *new_out_shape = Shape3(leading, rhs_broadcasting_size, trailing);
    } else {
      index_t leading =
        lhs_shape.ProdShape(0, lhs_axes[0]);
      index_t trailing =
        lhs_shape.ProdShape(lhs_axes[lhs_axes.ndim() - 1] + 1, lhs_shape.ndim());
      *new_lhs_shape = Shape3(leading, 1, trailing);
      *new_rhs_shape = Shape3(leading, lhs_broadcasting_size, trailing);
      *new_out_shape = Shape3(leading, lhs_broadcasting_size, trailing);
    }
  } else {
    *do_opt = false;
    CHECK(lhs_shape.ndim() <= MXNET_SPECIAL_MAX_NDIM)
      << "Only support input dimension up to " << MXNET_SPECIAL_MAX_NDIM
      << ", lhs_shape=" << lhs_shape << ", rhs_shape=" << rhs_shape
      << ", out_shape=" << out_shape;
    *new_lhs_shape = TShape(MXNET_SPECIAL_MAX_NDIM);
    *new_rhs_shape = TShape(MXNET_SPECIAL_MAX_NDIM);
    *new_out_shape = TShape(MXNET_SPECIAL_MAX_NDIM);
    for (index_t i = 0; i < lhs_shape.ndim(); i++) {
      (*new_lhs_shape)[i] = lhs_shape[i];
      (*new_rhs_shape)[i] = rhs_shape[i];
      (*new_out_shape)[i] = out_shape[i];
    }
  }
  CHECK(((*new_lhs_shape).Size() == lhs_shape.Size())
    && ((*new_rhs_shape).Size() == rhs_shape.Size())
    && ((*new_out_shape).Size() == out_shape.Size()))
    << "new_lhs_shape:" << *new_lhs_shape << ",lhs_shape:" << lhs_shape
    << "new_rhs_shape:" << *new_rhs_shape << ",rhs_shape:" << rhs_shape
    << "new_out_shape:" << *new_out_shape << ",out_shape:" << out_shape;
}

inline void GetBroadcastShape_(const TShape& lhs,
                               const TShape& rhs,
                               TShape* ret_reshaped,
                               int* lhs_broadcast_axis,
                               int* rhs_broadcast_axis) {
  TShape ret = BinaryBroadcastShape_(lhs, rhs, EnvArguments());
  int n = static_cast<int>(ret.ndim());
  int pos[4] = {0, n, n, n};
  for (int h = 0; h < 2; ++h) {
    const TShape& inp = h == 0 ? lhs : rhs;
    for (int i = 0; i < n; ++i) {
      if (inp[i] == ret[i]) {
        pos[h*2] = i; break;
      }
    }
    for (int i = n; i > 0; --i) {
      if (inp[i-1] == ret[i-1]) {
        pos[h*2+1] = i; break;
      }
    }
  }
  bool no_broadcast_lhs = pos[0] == 0 && pos[1] == n;
  bool no_broadcast_rhs = pos[2] == 0 && pos[3] == n;
  int pos_ordered[4] = {0, -1, -1, n};
  if (no_broadcast_lhs && no_broadcast_rhs) {
    // no broadcast
    LOG(FATAL) << "no broadcast is needed";
  } else if (no_broadcast_lhs && !no_broadcast_rhs) {
    // only broadcast rhs
    *rhs_broadcast_axis = 1;
    *lhs_broadcast_axis = -1;
    pos_ordered[1] = pos[2];
    pos_ordered[2] = pos[3];
  } else if (!no_broadcast_lhs && no_broadcast_rhs) {
    // only broadcast lhs
    *rhs_broadcast_axis = -1;
    *lhs_broadcast_axis = 1;
    pos_ordered[1] = pos[0];
    pos_ordered[2] = pos[1];
  } else {
    // broadcast both lhs and rhs
    int p;
    if (pos[0] <= pos[2]) {
      CHECK(pos[0] == 0 && pos[1] == pos[2] && pos[3] == n)
        << "broadcast shape error: lhs = " << lhs << "; rhs = " << rhs;
      *lhs_broadcast_axis = 0;
      *rhs_broadcast_axis = 1;
      p = pos[1];
    } else {
      CHECK(pos[2] == 0 && pos[3] == pos[0] && pos[1] == n)
        << "broadcast shape error: lhs = " << lhs << "; rhs = " << rhs;
      *lhs_broadcast_axis = 1;
      *rhs_broadcast_axis = 0;
      p = pos[0];
    }
    std::vector<index_t> dim(2, 1);
    for (int i = 0; i < p; ++i) dim[0] *= ret[i];
    for (int i = p; i < n; ++i) dim[1] *= ret[i];
    *ret_reshaped = TShape(dim.begin(), dim.end());
    return;
  }
  std::vector<index_t> dim(3, 1);
  for (int i = 0; i < 3; ++i) {
    for (int j = pos_ordered[i]; j < pos_ordered[i+1]; ++j) {
      dim[i] *= ret[j];
    }
  }
  *ret_reshaped = TShape(dim.begin(), dim.end());
}

// template<typename xpu, typename OP>
// void BinaryBroadcastForward_(const TBlob& lhs,
                             // const TBlob& rhs,
                             // const EnvArguments& env,
                             // TBlob *ret,
                             // OpReqType req,
                             // RunContext ctx) {
  // using namespace mshadow::expr;
  // using mshadow::Shape;
  // using mshadow::Shape1;
  // using mshadow::Tensor;
  // mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  // CHECK_EQ(ret->type_flag_, lhs.type_flag_)
    // << "Binary function only support input/output with the same type";
  // CHECK_EQ(ret->type_flag_, rhs.type_flag_)
    // << "Binary function only support input/output with the same type";

  // if (!IsBroadcastNeeded_(lhs.shape_, rhs.shape_)) {
    // // no broadcast
    // MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
        // Tensor<xpu, 2, DType> out = ret->FlatTo2D<xpu, DType>(s);
        // ASSIGN_DISPATCH(out, req,
                        // F<OP>(lhs.FlatTo2D<xpu, DType>(s),
                              // rhs.FlatTo2D<xpu, DType>(s)));
      // });
    // return;
  // }

  // TShape ret_reshaped;
  // int lhs_broadcast_axis;
  // int rhs_broadcast_axis;
  // GetBroadcastShape_(lhs.shape_, rhs.shape_, &ret_reshaped,
                     // &lhs_broadcast_axis, &rhs_broadcast_axis);
  // MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
      // if (lhs_broadcast_axis >= 0) {
        // // broadcast lhs
        // Tensor<xpu, 1, DType> mlhs =
            // lhs.get_with_shape<xpu, 1, DType>(Shape1(lhs.shape_.Size()), s);
        // if (rhs_broadcast_axis >= 0) {
          // // broadcast both
          // Tensor<xpu, 1, DType> mrhs =
              // rhs.get_with_shape<xpu, 1, DType>(Shape1(rhs.shape_.Size()), s);

          // Shape<2> ret_mshape = ret_reshaped.get<2>();
          // Tensor<xpu, 2, DType> out =
              // ret->get_with_shape<xpu, 2, DType>(ret_mshape, s);
          // if (lhs_broadcast_axis == 0) {
            // ASSIGN_DISPATCH(out, req,
                            // F<OP>(broadcast<0>(mlhs, ret_mshape),
                                  // broadcast<1>(mrhs, ret_mshape)));
          // } else {
            // ASSIGN_DISPATCH(out, req,
                            // F<OP>(broadcast<1>(mlhs, ret_mshape),
                                  // broadcast<0>(mrhs, ret_mshape)));
          // }
        // } else {
          // // only lhs
          // Shape<3> ret_mshape = ret_reshaped.get<3>();
          // Tensor<xpu, 3, DType> out =
              // ret->get_with_shape<xpu, 3, DType>(ret_mshape, s);
          // Tensor<xpu, 3, DType> mrhs =
              // rhs.get_with_shape<xpu, 3, DType>(ret_mshape, s);
          // if (lhs.shape_.Size() == 1) {
            // ASSIGN_DISPATCH(out, req,
                            // F<OP>(broadcast_scalar(mlhs, ret_mshape), mrhs));
          // } else {
            // ASSIGN_DISPATCH(out, req,
                            // F<OP>(broadcast<1>(mlhs, ret_mshape), mrhs));
          // }
        // }
      // } else {
        // Tensor<xpu, 1, DType> mrhs =
            // rhs.get_with_shape<xpu, 1, DType>(mshadow::Shape1(rhs.shape_.Size()), s);
        // if (rhs_broadcast_axis >= 0) {
          // // only rhs
          // Shape<3> ret_mshape = ret_reshaped.get<3>();
          // Tensor<xpu, 3, DType> out =
              // ret->get_with_shape<xpu, 3, DType>(ret_mshape, s);
          // Tensor<xpu, 3, DType> mlhs =
              // lhs.get_with_shape<xpu, 3, DType>(ret_mshape, s);
          // if (lhs.shape_.Size() == 1) {
            // ASSIGN_DISPATCH(out, req,
                            // F<OP>(mlhs, broadcast_scalar(mrhs, ret_mshape)));
          // } else {
            // ASSIGN_DISPATCH(out, req,
                            // F<OP>(mlhs, broadcast<1>(mrhs, ret_mshape)));
          // }
        // } else {
          // LOG(FATAL) << "no broadcast is needed";
        // }
      // }
    // });
// }

template<typename xpu, typename OP>
void BinaryBroadcastForward_(const TBlob& lhs,
                             const TBlob& rhs,
                             const EnvArguments& env,
                             TBlob *ret,
                             OpReqType req,
                             RunContext ctx) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(ret->type_flag_, lhs.type_flag_)
    << "Binary function only support input/output with the same type";
  CHECK_EQ(ret->type_flag_, rhs.type_flag_)
    << "Binary function only support input/output with the same type";
  CHECK_EQ(lhs.shape_.ndim(), rhs.shape_.ndim()) << "the ndim of lhs and rhs must be equal,"
    " shape of lhs=" << lhs.shape_ << " shape of rhs=" << rhs.shape_;
  if (!IsBroadcastNeeded_(lhs.shape_, rhs.shape_)) {
    // no broadcast
    MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
      mshadow::Tensor<xpu, 2, DType> out = ret->FlatTo2D<xpu, DType>(s);
      ASSIGN_DISPATCH(out, req,
        F<OP>(lhs.FlatTo2D<xpu, DType>(s),
        rhs.FlatTo2D<xpu, DType>(s)));
    });
    return;
  }
  bool do_opt;
  TShape lhs_new_shape_, rhs_new_shape_, out_new_shape_;
  InferBroadcastNewShapes_(&do_opt, &lhs_new_shape_, &rhs_new_shape_, &out_new_shape_,
    lhs.shape_, rhs.shape_, ret->shape_);
  MSHADOW_TYPE_SWITCH(ret->type_flag_, DType, {
    if (do_opt) {
      Shape<3> lhs_new_shape, rhs_new_shape, out_new_shape;
      for (index_t i = 0; i < 3; i++) {
        lhs_new_shape[i] = lhs_new_shape_[i];
        rhs_new_shape[i] = rhs_new_shape_[i];
        out_new_shape[i] = out_new_shape_[i];
      }
      Tensor<xpu, 3, DType> out = ret->get_with_shape<xpu, 3, DType>(out_new_shape, s);
      Tensor<xpu, 3, DType> mlhs = lhs.get_with_shape<xpu, 3, DType>(lhs_new_shape, s);
      Tensor<xpu, 3, DType> mrhs = rhs.get_with_shape<xpu, 3, DType>(rhs_new_shape, s);
      ASSIGN_DISPATCH(out, req,
        F<OP>(broadcast_detail::conditional(mlhs, broadcast_to(mlhs, out_new_shape_), lhs_new_shape == out_new_shape),
              broadcast_detail::conditional(mrhs, broadcast_to(mrhs, out_new_shape_), rhs_new_shape == out_new_shape)));
    } else {
      Shape<MXNET_SPECIAL_MAX_NDIM> lhs_new_shape, rhs_new_shape, out_new_shape;
      for (index_t i = 0; i < MXNET_SPECIAL_MAX_NDIM; i++) {
        lhs_new_shape[i] = lhs_new_shape_[i];
        rhs_new_shape[i] = rhs_new_shape_[i];
        out_new_shape[i] = out_new_shape_[i];
      }
      Tensor<xpu, MXNET_SPECIAL_MAX_NDIM, DType> out =
        ret->get_with_shape<xpu, MXNET_SPECIAL_MAX_NDIM, DType>(out_new_shape, s);
      Tensor<xpu, MXNET_SPECIAL_MAX_NDIM, DType> mlhs =
        lhs.get_with_shape<xpu, MXNET_SPECIAL_MAX_NDIM, DType>(lhs_new_shape, s);
      Tensor<xpu, MXNET_SPECIAL_MAX_NDIM, DType> mrhs =
        rhs.get_with_shape<xpu, MXNET_SPECIAL_MAX_NDIM, DType>(rhs_new_shape, s);
      ASSIGN_DISPATCH(out, req,
        F<OP>(broadcast_detail::conditional(mlhs, broadcast_to(mlhs, out_new_shape_), lhs_new_shape == out_new_shape),
              broadcast_detail::conditional(mrhs, broadcast_to(mrhs, out_new_shape_), rhs_new_shape == out_new_shape)));
    }
  });
}

// template<typename xpu, typename LHS_OP, typename RHS_OP>
// void BinaryBroadcastBackward_(const OutputGrad& out_grad,
                              // const EnvArguments& env,
                              // TBlob* lhs_grad,
                              // TBlob* rhs_grad,
                              // OpReqType req_lhs_grad,
                              // OpReqType req_rhs_grad,
                              // RunContext ctx) {
  // using namespace mshadow::expr;
  // using mshadow::Shape;
  // using mshadow::Shape1;
  // using mshadow::Shape2;
  // using mshadow::Tensor;
  // mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();

  // if (!IsBroadcastNeeded_(lhs_grad->shape_, rhs_grad->shape_)) {
    // // no broadcast
    // MSHADOW_TYPE_SWITCH(lhs_grad->type_flag_, DType, {
        // Tensor<xpu, 2, DType> mout_grad = out_grad.data.FlatTo2D<xpu, DType>(s);
        // Tensor<xpu, 2, DType> mlhs_grad = lhs_grad->FlatTo2D<xpu, DType>(s);
        // Tensor<xpu, 2, DType> mrhs_grad = rhs_grad->FlatTo2D<xpu, DType>(s);
        // ASSIGN_DISPATCH(mlhs_grad, req_lhs_grad, F<LHS_OP>(mout_grad));
        // ASSIGN_DISPATCH(mrhs_grad, req_rhs_grad, F<RHS_OP>(mout_grad));
      // });
    // return;
  // }

  // TShape ret_reshaped;
  // int lhs_broadcast_axis;
  // int rhs_broadcast_axis;
  // GetBroadcastShape_(lhs_grad->shape_, rhs_grad->shape_, &ret_reshaped,
                     // &lhs_broadcast_axis, &rhs_broadcast_axis);
  // index_t lhs_size = lhs_grad->shape_.Size();
  // index_t rhs_size = rhs_grad->shape_.Size();

  // MSHADOW_REAL_TYPE_SWITCH(lhs_grad->type_flag_, DType, {
      // if (lhs_broadcast_axis >= 0) {
        // Tensor<xpu, 1, DType> mlhs_grad =
            // lhs_grad->get_with_shape<xpu, 1, DType>(Shape1(lhs_size), s);
        // if (rhs_broadcast_axis >= 0) {
          // // broadcast both
          // Tensor<xpu, 2, DType> mout_grad =
              // out_grad.data.get_with_shape<xpu, 2, DType>(ret_reshaped.get<2>(), s);
          // Tensor<xpu, 1, DType> mrhs_grad =
              // rhs_grad->get_with_shape<xpu, 1, DType>(Shape1(rhs_size), s);
          // if (lhs_broadcast_axis == 0) {
            // ASSIGN_DISPATCH(
                // mlhs_grad, req_lhs_grad, sumall_except_dim<0>(F<LHS_OP>(mout_grad)));
            // ASSIGN_DISPATCH(
                // mrhs_grad, req_rhs_grad, sumall_except_dim<1>(F<RHS_OP>(mout_grad)));
          // } else {
            // ASSIGN_DISPATCH(
                // mlhs_grad, req_lhs_grad, sumall_except_dim<1>(F<LHS_OP>(mout_grad)));
            // ASSIGN_DISPATCH(
                // mrhs_grad, req_rhs_grad, sumall_except_dim<0>(F<RHS_OP>(mout_grad)));
          // }
        // } else {
          // // only broadcast lhs
          // Tensor<xpu, 3, DType> mout_grad =
              // out_grad.data.get_with_shape<xpu, 3, DType>(ret_reshaped.get<3>(), s);
          // Tensor<xpu, 3, DType> mrhs_grad =
              // rhs_grad->get_with_shape<xpu, 3, DType>(ret_reshaped.get<3>(), s);
          // ASSIGN_DISPATCH(
              // mlhs_grad, req_lhs_grad, sumall_except_dim<1>(F<LHS_OP>(mout_grad)));
          // ASSIGN_DISPATCH(mrhs_grad, req_rhs_grad, F<RHS_OP>(mout_grad));
        // }
      // } else {
        // if (rhs_broadcast_axis >= 0) {
          // // only broadcast rhs
          // Tensor<xpu, 3, DType> mlhs_grad =
              // lhs_grad->get_with_shape<xpu, 3, DType>(ret_reshaped.get<3>(), s);
          // Tensor<xpu, 1, DType> mrhs_grad =
              // rhs_grad->get_with_shape<xpu, 1, DType>(Shape1(rhs_size), s);
          // Tensor<xpu, 3, DType> mout_grad =
              // out_grad.data.get_with_shape<xpu, 3, DType>(ret_reshaped.get<3>(), s);
          // ASSIGN_DISPATCH(mlhs_grad, req_lhs_grad, F<LHS_OP>(mout_grad));
          // ASSIGN_DISPATCH(
              // mrhs_grad, req_rhs_grad, sumall_except_dim<1>(F<RHS_OP>(mout_grad)));
        // } else {
          // LOG(FATAL) << "no broadcast is needed";
        // }
      // }
    // });
// }

template<typename xpu, typename LHS_OP, typename RHS_OP>
void BinaryBroadcastBackward_(const OutputGrad& out_grad,
                              const EnvArguments& env,
                              TBlob* lhs_grad,
                              TBlob* rhs_grad,
                              OpReqType req_lhs_grad,
                              OpReqType req_rhs_grad,
                              RunContext ctx) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  CHECK_EQ(out_grad.data.type_flag_, lhs_grad->type_flag_)
    << "Binary function only support ingrad/outgrad with the same type";
  CHECK_EQ(out_grad.data.type_flag_, rhs_grad->type_flag_)
    << "Binary function only support ingrad/outgrad with the same type";
  CHECK_EQ(rhs_grad->shape_.ndim(), rhs_grad->shape_.ndim()) <<
    "the ndim of lhs_grad and rhs_grad must be equal,"
    " shape of lhs_grad=" << lhs_grad->shape_ << " shape of rhs_grad=" << rhs_grad->shape_;
  if (!IsBroadcastNeeded_(lhs_grad->shape_, rhs_grad->shape_)) {
    // no broadcast
    MSHADOW_TYPE_SWITCH(lhs_grad->type_flag_, DType, {
        Tensor<xpu, 2, DType> mout_grad = out_grad.data.FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> mlhs_grad = lhs_grad->FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> mrhs_grad = rhs_grad->FlatTo2D<xpu, DType>(s);
        ASSIGN_DISPATCH(mlhs_grad, req_lhs_grad, F<LHS_OP>(mout_grad));
        ASSIGN_DISPATCH(mrhs_grad, req_rhs_grad, F<RHS_OP>(mout_grad));
      });
    return;
  }
  bool do_opt;
  TShape lhs_new_shape_, rhs_new_shape_, out_new_shape_;
  InferBroadcastNewShapes_(&do_opt, &lhs_new_shape_, &rhs_new_shape_, &out_new_shape_,
    lhs_grad->shape_, rhs_grad->shape_, out_grad.data.shape_);
  MSHADOW_REAL_TYPE_SWITCH(lhs_grad->type_flag_, DType, {
    if (do_opt) {
      Shape<3> out_new_shape;
      for (index_t i = 0; i < 3; i++) {
        out_new_shape[i] = out_new_shape_[i];
      }
      Tensor<xpu, 3, DType> mout_grad =
        out_grad.data.get_with_shape<xpu, 3, DType>(out_new_shape, s);
      Tensor<xpu, 1, DType> mlhs_grad =
        lhs_grad->get_with_shape<xpu, 1, DType>(Shape1(lhs_grad->Size()), s);
      Tensor<xpu, 1, DType> mrhs_grad =
        rhs_grad->get_with_shape<xpu, 1, DType>(Shape1(rhs_grad->Size()), s);
      ReduceToAssign<red::sum>(mlhs_grad, req_lhs_grad, lhs_new_shape_, F<LHS_OP>(mout_grad));
      ReduceToAssign<red::sum>(mrhs_grad, req_rhs_grad, rhs_new_shape_, F<RHS_OP>(mout_grad));
    } else {
      Shape<MXNET_SPECIAL_MAX_NDIM> out_new_shape;
      for (index_t i = 0; i < MXNET_SPECIAL_MAX_NDIM; i++) {
        out_new_shape[i] = out_new_shape_[i];
      }
      Tensor<xpu, MXNET_SPECIAL_MAX_NDIM, DType> mout_grad =
        out_grad.data.get_with_shape<xpu, MXNET_SPECIAL_MAX_NDIM, DType>(out_new_shape, s);
      Tensor<xpu, 1, DType> mlhs_grad =
        lhs_grad->get_with_shape<xpu, 1, DType>(Shape1(lhs_grad->Size()), s);
      Tensor<xpu, 1, DType> mrhs_grad =
        rhs_grad->get_with_shape<xpu, 1, DType>(Shape1(rhs_grad->Size()), s);
      ReduceToAssign<red::sum>(mlhs_grad, req_lhs_grad, lhs_new_shape_, F<LHS_OP>(mout_grad));
      ReduceToAssign<red::sum>(mrhs_grad, req_rhs_grad, rhs_new_shape_, F<RHS_OP>(mout_grad));
    }
  });
}

template<typename xpu>
void BroadcastMulBackward_(const OutputGrad& out_grad,
                            const Input0& lhs,
                            const Input1& rhs,
                            const EnvArguments& env,
                            TBlob* lhs_grad,
                            TBlob* rhs_grad,
                            OpReqType req_lhs_grad,
                            OpReqType req_rhs_grad,
                            RunContext ctx) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  if (!IsBroadcastNeeded_(lhs_grad->shape_, rhs_grad->shape_)) {
    MSHADOW_TYPE_SWITCH(lhs_grad->type_flag_, DType, {
      mshadow::Tensor<xpu, 2, DType> mout_grad = out_grad.data.FlatTo2D<xpu, DType>(s);
      mshadow::Tensor<xpu, 2, DType> mlhs_data = lhs.data.FlatTo2D<xpu, DType>(s);
      mshadow::Tensor<xpu, 2, DType> mrhs_data = rhs.data.FlatTo2D<xpu, DType>(s);
      mshadow::Tensor<xpu, 2, DType> mlhs_grad = lhs_grad->FlatTo2D<xpu, DType>(s);
      mshadow::Tensor<xpu, 2, DType> mrhs_grad = rhs_grad->FlatTo2D<xpu, DType>(s);
      CHECK_NE(req_rhs_grad, kWriteInplace);
      ASSIGN_DISPATCH(mrhs_grad, req_rhs_grad, mlhs_data * mout_grad);
      ASSIGN_DISPATCH(mlhs_grad, req_lhs_grad, mrhs_data * mout_grad);
    });
    return;
  }
  bool do_opt;
  TShape lhs_new_shape_, rhs_new_shape_, out_new_shape_;
  InferBroadcastNewShapes_(&do_opt, &lhs_new_shape_, &rhs_new_shape_, &out_new_shape_,
    lhs_grad->shape_, rhs_grad->shape_, out_grad.data.shape_);
  MSHADOW_REAL_TYPE_SWITCH(lhs_grad->type_flag_, DType, {
    if (do_opt) {
      Shape<3> lhs_new_shape, rhs_new_shape, out_new_shape;
      for (index_t i = 0; i < 3; i++) {
        lhs_new_shape[i] = lhs_new_shape_[i];
        rhs_new_shape[i] = rhs_new_shape_[i];
        out_new_shape[i] = out_new_shape_[i];
      }
      mshadow::Tensor<xpu, 3, DType> mout_grad =
        out_grad.data.get_with_shape<xpu, 3, DType>(out_new_shape, s);
      mshadow::Tensor<xpu, 3, DType> mlhs_data =
        lhs.data.get_with_shape<xpu, 3, DType>(lhs_new_shape, s);
      mshadow::Tensor<xpu, 3, DType> mrhs_data =
        rhs.data.get_with_shape<xpu, 3, DType>(rhs_new_shape, s);
      mshadow::Tensor<xpu, 1, DType> mlhs_grad =
        lhs_grad->get_with_shape<xpu, 1, DType>(Shape1(lhs_grad->Size()), s);
      mshadow::Tensor<xpu, 1, DType> mrhs_grad =
        rhs_grad->get_with_shape<xpu, 1, DType>(Shape1(rhs_grad->Size()), s);
      if (lhs_new_shape == out_new_shape) {
        ReduceToAssign<red::sum>(mrhs_grad, req_rhs_grad, rhs_new_shape_,
          mlhs_data * mout_grad);
      } else {
        ReduceToAssign<red::sum>(mrhs_grad, req_rhs_grad, rhs_new_shape_,
          broadcast_to(mlhs_data, out_new_shape_) * mout_grad);
      }
      if (rhs_new_shape == out_new_shape) {
        ReduceToAssign<red::sum>(mlhs_grad, req_lhs_grad, lhs_new_shape_,
          mrhs_data * mout_grad);
      } else {
        ReduceToAssign<red::sum>(mlhs_grad, req_lhs_grad, lhs_new_shape_,
          broadcast_to(mrhs_data, out_new_shape_) * mout_grad);
      }
    } else {
      Shape<MXNET_SPECIAL_MAX_NDIM> lhs_new_shape, rhs_new_shape, out_new_shape;
      for (index_t i = 0; i < MXNET_SPECIAL_MAX_NDIM; i++) {
        lhs_new_shape[i] = lhs_new_shape_[i];
        rhs_new_shape[i] = rhs_new_shape_[i];
        out_new_shape[i] = out_new_shape_[i];
      }
      mshadow::Tensor<xpu, MXNET_SPECIAL_MAX_NDIM, DType> mout_grad =
        out_grad.data.get_with_shape<xpu, MXNET_SPECIAL_MAX_NDIM, DType>(out_new_shape, s);
      mshadow::Tensor<xpu, MXNET_SPECIAL_MAX_NDIM, DType> mlhs_data =
        lhs.data.get_with_shape<xpu, MXNET_SPECIAL_MAX_NDIM, DType>(lhs_new_shape, s);
      mshadow::Tensor<xpu, MXNET_SPECIAL_MAX_NDIM, DType> mrhs_data =
        rhs.data.get_with_shape<xpu, MXNET_SPECIAL_MAX_NDIM, DType>(rhs_new_shape, s);
      mshadow::Tensor<xpu, 1, DType> mlhs_grad =
        lhs_grad->get_with_shape<xpu, 1, DType>(Shape1(lhs_grad->Size()), s);
      mshadow::Tensor<xpu, 1, DType> mrhs_grad =
        rhs_grad->get_with_shape<xpu, 1, DType>(Shape1(rhs_grad->Size()), s);
      ReduceToAssign<red::sum>(mrhs_grad, req_rhs_grad, rhs_new_shape_,
        broadcast_to(mlhs_data, out_new_shape_) * mout_grad);
      ReduceToAssign<red::sum>(mlhs_grad, req_lhs_grad, lhs_new_shape_,
        broadcast_to(mrhs_data, out_new_shape_) * mout_grad);
    }
  });
}

template<typename xpu>
void BroadcastDivBackward_(const OutputGrad& out_grad,
  const Input0& lhs,
  const Input1& rhs,
  const EnvArguments& env,
  TBlob* lhs_grad,
  TBlob* rhs_grad,
  OpReqType req_lhs_grad,
  OpReqType req_rhs_grad,
  RunContext ctx) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  if (!IsBroadcastNeeded_(lhs_grad->shape_, rhs_grad->shape_)) {
    MSHADOW_TYPE_SWITCH(lhs_grad->type_flag_, DType, {
      mshadow::Tensor<xpu, 2, DType> mout_grad = out_grad.data.FlatTo2D<xpu, DType>(s);
      mshadow::Tensor<xpu, 2, DType> mlhs_data = lhs.data.FlatTo2D<xpu, DType>(s);
      mshadow::Tensor<xpu, 2, DType> mrhs_data = rhs.data.FlatTo2D<xpu, DType>(s);
      mshadow::Tensor<xpu, 2, DType> mlhs_grad = lhs_grad->FlatTo2D<xpu, DType>(s);
      mshadow::Tensor<xpu, 2, DType> mrhs_grad = rhs_grad->FlatTo2D<xpu, DType>(s);
      CHECK_NE(req_rhs_grad, kWriteInplace);
      ASSIGN_DISPATCH(mrhs_grad, req_rhs_grad,
        F<mshadow_op::negation>(mout_grad * mlhs_data) /
        F<mshadow_op::square>(mrhs_data));
      ASSIGN_DISPATCH(mlhs_grad, req_lhs_grad, mout_grad / mrhs_data);
    });
    return;
  }
  bool do_opt;
  TShape lhs_new_shape_, rhs_new_shape_, out_new_shape_;
  InferBroadcastNewShapes_(&do_opt, &lhs_new_shape_, &rhs_new_shape_, &out_new_shape_,
    lhs_grad->shape_, rhs_grad->shape_, out_grad.data.shape_);
  MSHADOW_REAL_TYPE_SWITCH(lhs_grad->type_flag_, DType, {
    if (do_opt) {
      Shape<3> lhs_new_shape, rhs_new_shape, out_new_shape;
      for (index_t i = 0; i < 3; i++) {
        lhs_new_shape[i] = lhs_new_shape_[i];
        rhs_new_shape[i] = rhs_new_shape_[i];
        out_new_shape[i] = out_new_shape_[i];
      }
      mshadow::Tensor<xpu, 3, DType> mout_grad =
        out_grad.data.get_with_shape<xpu, 3, DType>(out_new_shape, s);
      mshadow::Tensor<xpu, 3, DType> mlhs_data =
        lhs.data.get_with_shape<xpu, 3, DType>(lhs_new_shape, s);
      mshadow::Tensor<xpu, 3, DType> mrhs_data =
        rhs.data.get_with_shape<xpu, 3, DType>(rhs_new_shape, s);
      mshadow::Tensor<xpu, 1, DType> mlhs_grad =
        lhs_grad->get_with_shape<xpu, 1, DType>(Shape1(lhs_grad->Size()), s);
      mshadow::Tensor<xpu, 1, DType> mrhs_grad =
        rhs_grad->get_with_shape<xpu, 1, DType>(Shape1(rhs_grad->Size()), s);
      ReduceToAssign<red::sum>(mrhs_grad, req_rhs_grad, rhs_new_shape_,
        F<mshadow_op::negation>(mout_grad * broadcast_to(mlhs_data, out_new_shape_)) /
        F<mshadow_op::square>(broadcast_to(mrhs_data, out_new_shape_)));
      ReduceToAssign<red::sum>(mlhs_grad, req_lhs_grad, lhs_new_shape_, mout_grad /
        broadcast_to(mrhs_data, out_new_shape_));
    } else {
      Shape<MXNET_SPECIAL_MAX_NDIM> lhs_new_shape, rhs_new_shape, out_new_shape;
      for (index_t i = 0; i < MXNET_SPECIAL_MAX_NDIM; i++) {
        lhs_new_shape[i] = lhs_new_shape_[i];
        rhs_new_shape[i] = rhs_new_shape_[i];
        out_new_shape[i] = out_new_shape_[i];
      }
      mshadow::Tensor<xpu, MXNET_SPECIAL_MAX_NDIM, DType> mout_grad =
        out_grad.data.get_with_shape<xpu, MXNET_SPECIAL_MAX_NDIM, DType>(out_new_shape, s);
      mshadow::Tensor<xpu, MXNET_SPECIAL_MAX_NDIM, DType> mlhs_data =
        lhs.data.get_with_shape<xpu, MXNET_SPECIAL_MAX_NDIM, DType>(lhs_new_shape, s);
      mshadow::Tensor<xpu, MXNET_SPECIAL_MAX_NDIM, DType> mrhs_data =
        rhs.data.get_with_shape<xpu, MXNET_SPECIAL_MAX_NDIM, DType>(rhs_new_shape, s);
      mshadow::Tensor<xpu, 1, DType> mlhs_grad =
        lhs_grad->get_with_shape<xpu, 1, DType>(Shape1(lhs_grad->Size()), s);
      mshadow::Tensor<xpu, 1, DType> mrhs_grad =
        rhs_grad->get_with_shape<xpu, 1, DType>(Shape1(rhs_grad->Size()), s);
      ReduceToAssign<red::sum>(mrhs_grad, req_rhs_grad, rhs_new_shape_,
        F<mshadow_op::negation>(mout_grad * broadcast_to(mlhs_data, out_new_shape_)) /
        F<mshadow_op::square>(broadcast_to(mrhs_data, out_new_shape_)));
      ReduceToAssign<red::sum>(mlhs_grad, req_lhs_grad, lhs_new_shape_, mout_grad /
        broadcast_to(mrhs_data, out_new_shape_));
    }
  });
}

template<typename xpu>
void BroadcastPowerBackward_(const OutputGrad& out_grad,
  const Input0& lhs,
  const Input1& rhs,
  const EnvArguments& env,
  TBlob* lhs_grad,
  TBlob* rhs_grad,
  OpReqType req_lhs_grad,
  OpReqType req_rhs_grad,
  RunContext ctx) {
  using namespace mshadow;
  using namespace mshadow::expr;
  Stream<xpu> *s = ctx.get_stream<xpu>();
  if (!IsBroadcastNeeded_(lhs_grad->shape_, rhs_grad->shape_)) {
    MSHADOW_TYPE_SWITCH(lhs_grad->type_flag_, DType, {
      mshadow::Tensor<xpu, 2, DType> mout_grad = out_grad.data.FlatTo2D<xpu, DType>(s);
      mshadow::Tensor<xpu, 2, DType> mlhs_data = lhs.data.FlatTo2D<xpu, DType>(s);
      mshadow::Tensor<xpu, 2, DType> mrhs_data = rhs.data.FlatTo2D<xpu, DType>(s);
      mshadow::Tensor<xpu, 2, DType> mlhs_grad = lhs_grad->FlatTo2D<xpu, DType>(s);
      mshadow::Tensor<xpu, 2, DType> mrhs_grad = rhs_grad->FlatTo2D<xpu, DType>(s);
      CHECK_NE(req_rhs_grad, kWriteInplace);
      ASSIGN_DISPATCH(mrhs_grad, req_rhs_grad,
        F<mshadow_op::log>(mlhs_data) *
        F<mshadow_op::power>(mlhs_data, mrhs_data) * mout_grad);
      ASSIGN_DISPATCH(mlhs_grad, req_lhs_grad,
        mrhs_data *
        F<mshadow_op::power>(mlhs_data, mrhs_data - scalar<DType>(1)) *
        mout_grad);
    });
    return;
  }
  bool do_opt;
  TShape lhs_new_shape_, rhs_new_shape_, out_new_shape_;
  InferBroadcastNewShapes_(&do_opt, &lhs_new_shape_, &rhs_new_shape_, &out_new_shape_,
    lhs_grad->shape_, rhs_grad->shape_, out_grad.data.shape_);
  MSHADOW_REAL_TYPE_SWITCH(lhs_grad->type_flag_, DType, {
    if (do_opt) {
      Shape<3> lhs_new_shape, rhs_new_shape, out_new_shape;
      for (index_t i = 0; i < 3; i++) {
        lhs_new_shape[i] = lhs_new_shape_[i];
        rhs_new_shape[i] = rhs_new_shape_[i];
        out_new_shape[i] = out_new_shape_[i];
      }
      mshadow::Tensor<xpu, 3, DType> mout_grad =
        out_grad.data.get_with_shape<xpu, 3, DType>(out_new_shape, s);
      mshadow::Tensor<xpu, 3, DType> mlhs_data =
        lhs.data.get_with_shape<xpu, 3, DType>(lhs_new_shape, s);
      mshadow::Tensor<xpu, 3, DType> mrhs_data =
        rhs.data.get_with_shape<xpu, 3, DType>(rhs_new_shape, s);
      mshadow::Tensor<xpu, 1, DType> mlhs_grad =
        lhs_grad->get_with_shape<xpu, 1, DType>(Shape1(lhs_grad->Size()), s);
      mshadow::Tensor<xpu, 1, DType> mrhs_grad =
        rhs_grad->get_with_shape<xpu, 1, DType>(Shape1(rhs_grad->Size()), s);
      ReduceToAssign<red::sum>(mrhs_grad, req_rhs_grad, rhs_new_shape_,
        F<mshadow_op::log>(broadcast_to(mlhs_data, out_new_shape_)) *
        F<mshadow_op::power>(broadcast_to(mlhs_data, out_new_shape_),
                             broadcast_to(mrhs_data, out_new_shape_)) * mout_grad);
      ReduceToAssign<red::sum>(mlhs_grad, req_lhs_grad, lhs_new_shape_,
        broadcast_to(mrhs_data, out_new_shape_) *
        F<mshadow_op::power>(broadcast_to(mlhs_data, out_new_shape_),
                             broadcast_to(mrhs_data, out_new_shape_) - scalar<DType>(1)) *
        mout_grad);
    } else {
      Shape<MXNET_SPECIAL_MAX_NDIM> lhs_new_shape, rhs_new_shape, out_new_shape;
      for (index_t i = 0; i < MXNET_SPECIAL_MAX_NDIM; i++) {
        lhs_new_shape[i] = lhs_new_shape_[i];
        rhs_new_shape[i] = rhs_new_shape_[i];
        out_new_shape[i] = out_new_shape_[i];
      }
      mshadow::Tensor<xpu, MXNET_SPECIAL_MAX_NDIM, DType> mout_grad =
        out_grad.data.get_with_shape<xpu, MXNET_SPECIAL_MAX_NDIM, DType>(out_new_shape, s);
      mshadow::Tensor<xpu, MXNET_SPECIAL_MAX_NDIM, DType> mlhs_data =
        lhs.data.get_with_shape<xpu, MXNET_SPECIAL_MAX_NDIM, DType>(lhs_new_shape, s);
      mshadow::Tensor<xpu, MXNET_SPECIAL_MAX_NDIM, DType> mrhs_data =
        rhs.data.get_with_shape<xpu, MXNET_SPECIAL_MAX_NDIM, DType>(rhs_new_shape, s);
      mshadow::Tensor<xpu, 1, DType> mlhs_grad =
        lhs_grad->get_with_shape<xpu, 1, DType>(Shape1(lhs_grad->Size()), s);
      mshadow::Tensor<xpu, 1, DType> mrhs_grad =
        rhs_grad->get_with_shape<xpu, 1, DType>(Shape1(rhs_grad->Size()), s);
      ReduceToAssign<red::sum>(mrhs_grad, req_rhs_grad, rhs_new_shape_,
        F<mshadow_op::log>(broadcast_to(mlhs_data, out_new_shape_)) *
        F<mshadow_op::power>(broadcast_to(mlhs_data, out_new_shape_),
        broadcast_to(mrhs_data, out_new_shape_)) * mout_grad);
      ReduceToAssign<red::sum>(mlhs_grad, req_lhs_grad, lhs_new_shape_,
        broadcast_to(mrhs_data, out_new_shape_) *
        F<mshadow_op::power>(broadcast_to(mlhs_data, out_new_shape_),
        broadcast_to(mrhs_data, out_new_shape_) - scalar<DType>(1)) *
        mout_grad);
    }
  });
}


MXNET_REGISTER_SIMPLE_OP(broadcast_plus, XPU)
.set_shape_function(BinaryBroadcastShape_)
.set_function(XPU::kDevMask, BinaryBroadcastForward_<
              XPU, mshadow::op::plus>, kNoInplace, kRegisterSymbolic)
.set_gradient(XPU::kDevMask, BinaryBroadcastBackward_<
              XPU, mshadow_op::identity, mshadow_op::identity>, kNoInplace)
.describe("lhs add rhs with broadcast");

MXNET_REGISTER_SIMPLE_OP(broadcast_minus, XPU)
.set_shape_function(BinaryBroadcastShape_)
.set_function(XPU::kDevMask, BinaryBroadcastForward_<
              XPU, mshadow::op::minus>, kNoInplace, kRegisterSymbolic)
.set_gradient(XPU::kDevMask, BinaryBroadcastBackward_<
              XPU, mshadow_op::identity, mshadow_op::negation>, kNoInplace)
.describe("lhs minus rhs with broadcast");

MXNET_REGISTER_SIMPLE_OP(broadcast_mul, XPU)
.set_shape_function(BinaryBroadcastShape_)
.set_function(XPU::kDevMask, BinaryBroadcastForward_<
              XPU, mshadow::op::mul>, kNoInplace, kRegisterSymbolic)
.set_gradient(XPU::kDevMask, BroadcastMulBackward_<XPU>, kNoInplace)
.describe("lhs multiple rhs with broadcast");

MXNET_REGISTER_SIMPLE_OP(broadcast_div, XPU)
.set_shape_function(BinaryBroadcastShape_)
.set_function(XPU::kDevMask, BinaryBroadcastForward_<
              XPU, mshadow::op::div>, kNoInplace, kRegisterSymbolic)
.set_gradient(XPU::kDevMask, BroadcastDivBackward_<XPU>, kNoInplace)
.describe("lhs divide rhs with broadcast");

MXNET_REGISTER_SIMPLE_OP(broadcast_power, XPU)
.set_shape_function(BinaryBroadcastShape_)
.set_function(XPU::kDevMask, BinaryBroadcastForward_<
              XPU, mshadow_op::power>, kNoInplace, kRegisterSymbolic)
.set_gradient(XPU::kDevMask, BroadcastPowerBackward_<XPU>, kNoInplace)
.describe("lhs power rhs with broadcast");

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_ELEMENTWISE_BINARY_BROADCAST_OP_INL_H_
