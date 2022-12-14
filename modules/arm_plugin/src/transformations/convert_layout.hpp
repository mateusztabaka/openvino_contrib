// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <openvino/pass/graph_rewrite.hpp>

namespace ArmPlugin {
namespace pass {

class ConvertLayout: public ov::pass::GraphRewrite {
public:
    OPENVINO_RTTI("ConvertLayout", "0");
    ConvertLayout();
};

}  // namespace pass
}  // namespace ArmPlugin
