// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "gru_sequence_cudnn_components.hpp"

#include <error.hpp>
#include <gsl/gsl_assert>

#include "converters.hpp"

namespace CUDAPlugin::RNN::Details {

GRUSequenceParamsCuDnn::GRUSequenceParamsCuDnn(const GRUSequenceParams& params)
    : element_type_{convertDataType<cudnnDataType_t>(params.element_type_)},
      element_type_cuda_{convertDataType<cudaDataType_t>(params.element_type_)},
      element_size_{CUDAPlugin::elementSize(element_type_)},
      direction_{CUDNN_UNIDIRECTIONAL},
      linear_before_reset_{params.linear_before_reset_},
      batch_size_{static_cast<int32_t>(params.batch_size_)},
      max_seq_length_{static_cast<int32_t>(params.max_seq_length_)},
      input_size_{static_cast<int32_t>(params.input_size_)},
      hidden_size_{static_cast<int32_t>(params.hidden_size_)},
      w_host_buffers_{params.w_host_buffers_},
      r_host_buffers_{params.r_host_buffers_},
      b_host_buffers_{params.b_host_buffers_} {
    if (params.direction_ == ngraph::op::RecurrentSequenceDirection::REVERSE) {
        throwIEException("Currently GRUSequence cuDNN implementation doesn't support REVERSE direction");
    }
    if (params.direction_ == ngraph::op::RecurrentSequenceDirection::BIDIRECTIONAL) {
        throwIEException("Currently GRUSequence cuDNN implementation doesn't support BIDIRECTIONAL direction");
    }

    if (input_size_ == 1 && hidden_size_ == 1) {
        throwIEException(
            "Currently GRUSequence cuDNN implementation doesn't support combination of "
            "input_size == 1 and hidden_size == 1 simultaneously");
    }

    const auto supported_activations = std::vector<std::string>{"sigmoid", "tanh"};
    if (params.activations_ != supported_activations) {
        throwIEException(
            "Currently GRUSequence cuDNN implementation supports only "
            "\"sigmoid\", \"tanh\"");
    }

    const bool is_clipped = (params.clip_ != 0.0f) && !std::isinf(params.clip_);
    if (is_clipped) {
        throwIEException("Currently GRUSequence cuDNN implementation doesn't support clipping");
    }

    static_assert(sizeof(int32_t) == sizeof(int));
    seq_length_array_.resize(batch_size_, max_seq_length_);
}

GRUSequenceDescriptorsCuDnn::GRUSequenceDescriptorsCuDnn(const CreationContext& context,
                                                         const GRUSequenceParamsCuDnn& params,
                                                         const Config& config)
    : params_{params}, config_{config} {
    createRNNDescriptor(context);
    createXDescriptor();
    createYDescriptor();
    createHDescriptor();

    CUDA::DnnHandle dnn_handle{};
    weight_space_size_ = 0;
    throwIfError(cudnnGetRNNWeightSpaceSize(dnn_handle.get(), rnn_desc_.get(), &weight_space_size_));
    Ensures(weight_space_size_ >= params_.w_host_buffers_.size_bytes() + params_.r_host_buffers_.size_bytes() +
                                      params_.b_host_buffers_.size_bytes());

    work_space_size_ = 0;
    size_t reserve_space_size = 0;
    throwIfError(cudnnGetRNNTempSpaceSizes(
        dnn_handle.get(), rnn_desc_.get(), dnnForwardMode(), x_desc_.get(), &work_space_size_, &reserve_space_size));
    // the returned size of the reserve space buffer will be zero when the fMode argument is CUDNN_FWD_MODE_INFERENCE
    Ensures(reserve_space_size == 0);
}

void GRUSequenceDescriptorsCuDnn::createRNNDescriptor(const CreationContext& context) {
    const auto rnn_algo = CUDNN_RNN_ALGO_STANDARD;
    const auto rnn_mode = CUDNN_GRU;
    const auto bias_mode = params_.linear_before_reset_ ? CUDNN_RNN_DOUBLE_BIAS : CUDNN_RNN_SINGLE_INP_BIAS;
    const auto input_mode = CUDNN_LINEAR_INPUT;
    const bool can_use_half = (params_.element_type_ == CUDNN_DATA_HALF) && CUDA::isHalfSupported(context.device());
    const auto math_prec =
        ((params_.element_type_ == CUDNN_DATA_DOUBLE) || can_use_half) ? params_.element_type_ : CUDNN_DATA_FLOAT;
    const auto numLayers = 1;

    // Possible optimization: down type conversion can be forced with CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION option
    // to utilize Tensor Cores on the supported devices at the price of precision
    const auto math_type = (params_.element_type_ == CUDNN_DATA_DOUBLE || params_.element_type_ == CUDNN_DATA_FLOAT)
                               ? CUDNN_DEFAULT_MATH
                               : CUDNN_TENSOR_OP_MATH;

    // A single layer network will have no dropout applied. Dropout is used in the training mode only.
    const cudnnDropoutDescriptor_t drop_out_desc = nullptr;

    const uint32_t aux_flags =
        (config_.rnn_data_layout == CUDNN_RNN_DATA_LAYOUT_SEQ_MAJOR_PACKED) ? 0 : CUDNN_RNN_PADDED_IO_ENABLED;
    rnn_desc_.set(rnn_algo,
                  rnn_mode,
                  bias_mode,
                  params_.direction_,
                  input_mode,
                  params_.element_type_,
                  math_prec,
                  math_type,
                  params_.input_size_,
                  params_.hidden_size_,
                  params_.projSize(),
                  numLayers,
                  drop_out_desc,
                  aux_flags);

    // TODO: If cuDNN starts supporting similar clipping to OpenVINO, apply clipping here:
    //  rnn_desc_.setClip(...);
}

void GRUSequenceDescriptorsCuDnn::createXDescriptor() {
    const auto x_vector_size = params_.input_size_;
    x_desc_.set(params_.element_type_,
                config_.rnn_data_layout,
                params_.max_seq_length_,
                params_.batch_size_,
                x_vector_size,
                params_.seq_length_array_.data(),
                nullptr);
}

void GRUSequenceDescriptorsCuDnn::createYDescriptor() {
    const auto y_vector_size = params_.numDirections() * params_.projSize();
    y_desc_.set(params_.element_type_,
                                         config_.rnn_data_layout,
                                         params_.max_seq_length_,
                                         params_.batch_size_,
                                         y_vector_size,
                                         params_.seq_length_array_.data(),
                                         nullptr);
}

void GRUSequenceDescriptorsCuDnn::createHDescriptor() {
    const size_t nbDims = 3;
    const int h_dim_a[nbDims] = {params_.numDirections(), params_.batch_size_, params_.projSize()};
    const int h_stride_a[nbDims] = {params_.batch_size_ * params_.projSize(), params_.projSize(), 1};
    h_desc_.set(params_.element_type_, nbDims, h_dim_a, h_stride_a);
}

void GRUSequenceDescriptorsCuDnn::initDevSeqLengthArray(DevPtr buffer) {
    CUDA::DefaultStream::stream().upload(buffer, params_.seq_length_array_.data(), seqLengthArraySizeBytes());
}

void GRUSequenceDescriptorsCuDnn::initWeightSpace(DevPtr buffer) {
    calculateWeightBuffers(buffer);

    const int num_pseudo_layers = params_.numDirections();
    const auto dev_buffers_count = GRUSequenceParams::lin_layer_count * num_pseudo_layers;

    const auto w_host_buffer_size = params_.w_host_buffers_.size_bytes() / dev_buffers_count;
    const auto r_host_buffer_size = params_.r_host_buffers_.size_bytes() / dev_buffers_count;
    const auto b1_host_layer_size = params_.hidden_size_ * params_.element_size_;

    const uint8_t* w_host_addr = params_.w_host_buffers_.data();
    const uint8_t* r_host_addr = params_.r_host_buffers_.data();
    const uint8_t* b1_host_addr = params_.b_host_buffers_.data();

    const auto& stream = CUDA::DefaultStream::stream();

    for (int i = 0; i < dev_buffers_count; ++i) {
        // OpenVINO: linear layer indices are ZRH gates (Z states for update, R for reset and H for output hidden)
        //      https://docs.openvino.ai/latest/openvino_docs_ops_sequence_GRUSequence_5.html?highlight=grusequence
        // In cuDNN they are RZH
        //      https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnGetRNNWeightParams
        //
        // So we swap the corresponding buffers:
        int j = i;
        j = (j == 0) ? 1 : ((j == 1) ? 0 : j);

        Expects(w_host_buffer_size == w_dev_buffers_[j].size_bytes());
        stream.upload(DevPtr{w_dev_buffers_[j].data()}, w_host_addr, w_host_buffer_size);
        w_host_addr += w_host_buffer_size;

        Expects(b1_host_layer_size == b1_dev_buffers_[j].size_bytes());
        stream.upload(DevPtr{b1_dev_buffers_[j].data()}, b1_host_addr, b1_host_layer_size);
        b1_host_addr += b1_host_layer_size;

        Expects(r_host_buffer_size == r_dev_buffers_[j].size_bytes());
        stream.upload(DevPtr{r_dev_buffers_[j].data()}, r_host_addr, r_host_buffer_size);
        r_host_addr += r_host_buffer_size;

        if (i < b2_dev_buffers_.size()) {
            if (params_.linear_before_reset_ && (j == 2)) {
                // the tensor shape is [4 * hidden_size], get the fourth element from an array
                const size_t offset = b1_host_layer_size * (j + 1);
                Ensures(offset + b1_host_layer_size <= params_.b_host_buffers_.size_bytes());
                const uint8_t* b2_host_addr = params_.b_host_buffers_.data() + offset;
                stream.upload(DevPtr{b2_dev_buffers_[j].data()}, b2_host_addr, b1_host_layer_size);
            } else {
                stream.memset(DevPtr{b2_dev_buffers_[j].data()}, 0, b2_dev_buffers_[j].size_bytes());
            }
        }
    }
}

bool GRUSequenceDescriptorsCuDnn::weightBuffersFit(DevPtr buffer) const {
    auto weight_space = buffer.get();
    Expects(weight_space);

    DevBuffers all_weights;
    all_weights.reserve(w_dev_buffers_.size() + r_dev_buffers_.size() + b1_dev_buffers_.size() +
                        b2_dev_buffers_.size());
    for (const auto v : w_dev_buffers_) {
        all_weights.push_back(v);
    }
    for (const auto v : r_dev_buffers_) {
        all_weights.push_back(v);
    }
    for (const auto v : b1_dev_buffers_) {
        all_weights.push_back(v);
    }
    for (const auto v : b2_dev_buffers_) {
        all_weights.push_back(v);
    }

    std::sort(all_weights.begin(), all_weights.end(), [](DevBuffers::value_type a, DevBuffers::value_type b) {
        return a.data() < b.data();
    });

    const auto size = all_weights.size();
    uint8_t* addr_a = static_cast<uint8_t*>(all_weights[0].data());
    size_t size_a;
    uint8_t* addr_b = static_cast<uint8_t*>(buffer.get());

    if (addr_a < addr_b) {
        return false;
    }

    addr_a = static_cast<uint8_t*>(all_weights[size - 1].data());
    size_a = all_weights[size - 1].size_bytes();
    addr_b = static_cast<uint8_t*>(buffer.get());

    if (addr_a + size_a > addr_b + weightSpaceSize()) {
        return false;
    }

    for (int i = 1; i < size; ++i) {
        addr_a = static_cast<uint8_t*>(all_weights[i - 1].data());
        size_a = all_weights[i - 1].size_bytes();
        addr_b = static_cast<uint8_t*>(all_weights[i].data());
        if (addr_a + size_a > addr_b) {
            return false;
        }
    }

    return true;
}

void GRUSequenceDescriptorsCuDnn::calculateWeightBuffers(DevPtr buffer) {
    auto weight_space = buffer.get();
    Expects(weight_space);

    const auto data_type = params_.element_type_;
    const auto input_size = params_.input_size_;
    const auto hidden_size = params_.hidden_size_;
    const auto element_size = CUDAPlugin::elementSize(params_.element_type_);

    w_dev_buffers_.clear();
    r_dev_buffers_.clear();
    b1_dev_buffers_.clear();
    b2_dev_buffers_.clear();
    size_t w_total_bytes = 0;
    size_t r_total_bytes = 0;
    size_t b1_total_bytes = 0;
    size_t b2_total_bytes = 0;

    CUDA::DnnHandle dnn_handle{};
    int32_t lin_layer_id = 0;
    const int wb_nb_dims_requested = 3;
    cudnnDataType_t wb_data_type{};
    int wb_nb_dims = 0;
    int wb_dim_a[wb_nb_dims_requested] = {};
    int wb_stride_a[wb_nb_dims_requested] = {};
    size_t tensor_size_bytes = 0;

    const int num_pseudo_layers = params_.numDirections();
    for (int32_t pseudo_layer = 0; pseudo_layer < num_pseudo_layers; ++pseudo_layer) {
        const auto lin_layer_count = GRUSequenceParams::lin_layer_count;
        for (int i = 0; i < lin_layer_count; ++i) {
            lin_layer_id = i;
            CUDA::DnnTensorDescriptor w_desc{};
            CUDA::DnnTensorDescriptor b1_desc{};
            void* w_addr = nullptr;
            void* b1_addr = nullptr;
            throwIfError(cudnnGetRNNWeightParams(dnn_handle.get(),
                                                 rnn_desc_.get(),
                                                 pseudo_layer,
                                                 weight_space_size_,
                                                 weight_space,
                                                 lin_layer_id,
                                                 w_desc.get(),
                                                 &w_addr,
                                                 b1_desc.get(),
                                                 &b1_addr));
            Ensures(w_addr);
            w_desc.getTensorNdDescriptor(wb_nb_dims_requested, wb_data_type, wb_nb_dims, wb_dim_a, wb_stride_a);
            Ensures(wb_nb_dims == wb_nb_dims_requested);
            Ensures(wb_data_type == data_type);
            Ensures(wb_dim_a[0] == 1 && wb_dim_a[1] == hidden_size && wb_dim_a[2] == input_size);
            Ensures(wb_stride_a[0] == hidden_size * input_size && wb_stride_a[1] == input_size && wb_stride_a[2] == 1);
            tensor_size_bytes = w_desc.getTensorSizeInBytes();
            Ensures(tensor_size_bytes >= hidden_size * input_size * element_size);
            w_dev_buffers_.emplace_back(static_cast<uint8_t*>(w_addr), tensor_size_bytes);
            w_total_bytes += tensor_size_bytes;

            Ensures(b1_addr);
            b1_desc.getTensorNdDescriptor(wb_nb_dims_requested, wb_data_type, wb_nb_dims, wb_dim_a, wb_stride_a);
            Ensures(wb_nb_dims == wb_nb_dims_requested);
            Ensures(wb_data_type == data_type);
            Ensures(wb_dim_a[0] == 1 && wb_dim_a[1] == hidden_size && wb_dim_a[2] == 1);
            Ensures(wb_stride_a[0] == hidden_size && wb_stride_a[1] == 1 && wb_stride_a[2] == 1);
            tensor_size_bytes = b1_desc.getTensorSizeInBytes();
            Ensures(tensor_size_bytes >= hidden_size * element_size);
            b1_dev_buffers_.emplace_back(static_cast<uint8_t*>(b1_addr), tensor_size_bytes);
            b1_total_bytes += tensor_size_bytes;

            lin_layer_id = i + lin_layer_count;
            CUDA::DnnTensorDescriptor r_desc{};
            CUDA::DnnTensorDescriptor b2_desc{};
            void* r_addr = nullptr;
            void* b2_addr = nullptr;
            throwIfError(cudnnGetRNNWeightParams(dnn_handle.get(),
                                                 rnn_desc_.get(),
                                                 pseudo_layer,
                                                 weight_space_size_,
                                                 weight_space,
                                                 lin_layer_id,
                                                 r_desc.get(),
                                                 &r_addr,
                                                 b2_desc.get(),
                                                 &b2_addr));
            Ensures(r_addr);
            r_desc.getTensorNdDescriptor(wb_nb_dims_requested, wb_data_type, wb_nb_dims, wb_dim_a, wb_stride_a);
            Ensures(wb_nb_dims == wb_nb_dims_requested);
            Ensures(wb_data_type == data_type);
            Ensures(wb_dim_a[0] == 1 && wb_dim_a[1] == hidden_size && wb_dim_a[2] == hidden_size);
            Ensures(wb_stride_a[0] == hidden_size * hidden_size && wb_stride_a[1] == hidden_size &&
                    wb_stride_a[2] == 1);
            tensor_size_bytes = r_desc.getTensorSizeInBytes();
            Ensures(tensor_size_bytes >= hidden_size * hidden_size * element_size);
            r_dev_buffers_.emplace_back(static_cast<uint8_t*>(r_addr), tensor_size_bytes);
            r_total_bytes += tensor_size_bytes;

            if (b2_addr) {
                b2_desc.getTensorNdDescriptor(wb_nb_dims_requested, wb_data_type, wb_nb_dims, wb_dim_a, wb_stride_a);
                tensor_size_bytes = b2_desc.getTensorSizeInBytes();
                b2_dev_buffers_.emplace_back(static_cast<uint8_t*>(b2_addr), tensor_size_bytes);
                b2_total_bytes += tensor_size_bytes;
            }
        }
    }

    Ensures(weightBuffersFit(buffer));
    Ensures(weight_space_size_ >= w_total_bytes + r_total_bytes + b1_total_bytes + b2_total_bytes);
    Ensures(w_total_bytes >= params_.w_host_buffers_.size_bytes() &&
            r_total_bytes >= params_.r_host_buffers_.size_bytes());
}

}  // namespace CUDAPlugin::RNN::Details