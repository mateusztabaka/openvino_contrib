// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_layout.hpp"
#include "opset/opset.hpp"

#include <openvino/pass/pattern/op/wrap_type.hpp>
#include "transformations/utils/utils.hpp"

using namespace ov;

namespace ArmPlugin {
namespace pass {

static std::vector<int> nchw_to_nhwc{0, 2, 3, 1};
static std::vector<int> nhwc_to_nchw{0, 3, 1, 2};

static std::vector<int> ncdhw_to_ndhwc{0, 2, 3, 4, 1};
static std::vector<int> ndhwc_to_ncdhw{0, 4, 1, 2, 3};

static std::shared_ptr<ArmPlugin::opset::Transpose> transpose_on_input(const Output<Node>& input, size_t rank) {
    switch (rank) {
    case 4:
	return std::make_shared<ArmPlugin::opset::Transpose>(input,
		ArmPlugin::opset::Constant::create(element::i32, Shape{nchw_to_nhwc.size()}, nchw_to_nhwc));
    case 5:
	return std::make_shared<ArmPlugin::opset::Transpose>(input,
		ArmPlugin::opset::Constant::create(element::i32, Shape{ncdhw_to_ndhwc.size()}, ncdhw_to_ndhwc));
    default:
	IE_THROW() << "ConvertLayout: unsupported rank";
    }
}

std::shared_ptr<ArmPlugin::opset::Transpose> transpose_on_output(const Output<Node>& input, size_t rank) {
    switch (rank) {
    case 4:
	return std::make_shared<ArmPlugin::opset::Transpose>(input,
		ArmPlugin::opset::Constant::create(element::i32, Shape{nhwc_to_nchw.size()}, nhwc_to_nchw));
    case 5:
	return std::make_shared<ArmPlugin::opset::Transpose>(input,
		ArmPlugin::opset::Constant::create(element::i32, Shape{ndhwc_to_ncdhw.size()}, ndhwc_to_ncdhw));
    default:
	IE_THROW() << "ConvertLayout: unsupported rank";
    }
}

class ConvertArmConvolutionLayout : public ov::pass::MatcherPass {
    public:
	OPENVINO_RTTI("ConvertArmConvolutionLayout", "0");

	ConvertArmConvolutionLayout() {
	    auto root = ov::pass::pattern::wrap_type<opset::ArmConvolution>(ov::pass::pattern::has_static_rank());

	    matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
		auto node = m.get_match_root();
		auto conv = ov::as_type_ptr<opset::ArmConvolution>(node);
		if (!conv) {
		    return false;
		}
		size_t rank = conv->get_output_partial_shape(0).size();
		auto activations_transpose = transpose_on_input(conv->input_value(0), rank);
		auto weights_transpose = transpose_on_input(conv->input_value(1), rank);
		std::shared_ptr<opset::ArmConvolution> new_conv;
		if (conv->get_input_size() > 2) {
		    new_conv = std::make_shared<opset::ArmConvolution>(activations_transpose, weights_transpose, conv->input_value(2),
			    conv->get_strides(),
			    conv->get_pads_begin(),
			    conv->get_pads_end(),
			    conv->get_dilations(),
			    conv->get_auto_pad());
		} else {
		    new_conv = std::make_shared<opset::ArmConvolution>(activations_transpose, weights_transpose,
			    conv->get_strides(),
			    conv->get_pads_begin(),
			    conv->get_pads_end(),
			    conv->get_dilations(),
			    conv->get_auto_pad());
		}
		new_conv->set_friendly_name(conv->get_friendly_name());
		auto transpose = transpose_on_output(new_conv, rank);
		copy_runtime_info(conv, {new_conv, transpose});
		replace_node(conv, transpose);

		return true;
	    };

	    auto m = std::make_shared<ov::pass::pattern::Matcher>(root, "ConvertLayout");
	    register_matcher(m, callback);
	}
};

ConvertLayout::ConvertLayout() {
    add_matcher<ConvertArmConvolutionLayout>();
}

} // pass
} // ArmPlugin

