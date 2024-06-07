# VIVANTE(R) CORPORATION END USER LICENSE AGREEMENT (EULA)
# 
# Copyright 2019 VIVANTE CORPORATION
# 
# NOTICE TO USER: PLEASE READ THIS CONTRACT CAREFULLY. BY DOWNLOADING OR INSTALLING
# THE SOFTWARE, YOU ("USER") AGREE TO BE BOUND BY THE FOLLOWING TERMS AND CONDITIONS.
# YOU AGREE THAT THIS AGREEMENT IS ENFORCEABLE LIKE ANY WRITTEN NEGOTIATED AGREEMENT
# SIGNED BY YOU. IF YOU DO NOT AGREE, DO NOT USE THIS SOFTWARE. IF YOU ACQUIRED THE
# SOFTWARE ON TANGIBLE MEDIA (e.g., CD-ROM) WITHOUT AN OPPORTUNITY TO REVIEW THIS
# LICENSE, AND YOU DO NOT ACCEPT THIS AGREEMENT, YOU MAY NOT USE THE SOFTWARE.
# 
# The package available for download or installation to User includes only binary,
# machine-executable instructions ("Software").
# 
# VIVANTE owns the Software and makes it available to User only under the terms and
# conditions set forth in this Agreement.
# 
# License: Subject to the terms of this Agreement, VIVANTE hereby grants to User a
# non-exclusive, non-transferable, royalty-free license to possess and to use the Software.
# User agrees not to disassemble, decompile or reverse engineer the Software. User
# acknowledges that certain parts of the Software may contain third party components
# that may be subject to restrictions, and expressly agrees not to attempt to modify
# or distribute these without first receiving consent from VIVANTE and/or the respective
# third party.
# 
# Government End Users: If you are acquiring the Software on behalf of any unit or
# agency of the United States Government, the following provisions apply. The Government
# agrees the Software was developed at private expense and is provided with "RESTRICTED RIGHTS".
# Use, duplication, or disclosure by the Government is subject to restrictions as set forth
# in DFARS 227.7202-1(a) and 227.7202-3(a) (1995), DFARS 252.227-7013(c)(1)(ii) (Oct 1988),
# FAR 12.212(a) (1995), FAR 52.227-19, (June 1987) or FAR 52.227-14(ALT III) (June 1987),
# as amended from time to time. In the event that this License, or any part thereof, is deemed
# inconsistent with the minimum rights identified in the Restricted Rights provisions, the
# minimum rights shall prevail.
# 
# No Other License: No rights or licenses are granted by VIVANTE under this License, expressly
# or by implication, with respect to any proprietary information or patent, copyright, trade
# secret or other intellectual property right owned or controlled by VIVANTE, except as expressly
# provided in this License.
# 
# Term: Vivante has the right to terminate this Agreement immediately upon written notice, which
# may include email, if User fails to comply with any term of this Agreement. Upon any such
# termination, User must give up all rights to the Software.
# 
# Support: VIVANTE has no obligation to support or to continue providing or updating any of the Software.
# 
# NO WARRANTY: THE SOFTWARE AND ANY OTHER MATERIALS PROVIDED BY VIVANTE TO USER HEREUNDER ARE
# PROVIDED "AS IS." VIVANTE DISCLAIMS ALL WARRANTIES, EXPRESS, IMPLIED OR STATUTORY, INCLUDING,
# WITHOUT LIMITATION, THE IMPLIED WARRANTIES OF TITLE, MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. VIVANTE LICENSES THE SOFTWARE TO USER ON AN "AS IS" BASIS AND
# WITHOUT WARRANTY OF ANY KIND. VIVANTE AND ITS SUPPLIERS DO NOT AND CANNOT WARRANT THE PERFORMANCE
# OR RESULTS YOU MAY OBTAIN BY USING THE SOFTWARE. EXCEPT FOR ANY WARRANTY, CONDITION, REPRESENTATION
# OR TERM TO THE EXTENT TO WHICH THE SAME CANNOT OR MAY NOT BE EXCLUDED OR LIMITED BY LAW APPLICABLE
# TO YOU IN YOUR JURISDICTION, VIVANTE AND ITS SUPPLIERS MAKE NO WARRANTIES, CONDITIONS, REPRESENTATIONS
# OR TERMS, EXPRESS OR IMPLIED, WHETHER BY STATUTE, COMMON LAW, CUSTOM, USAGE OR OTHERWISE AS TO THE
# MATERIALS, SOFTWARE, OR ANY COMPONENT THEREOF, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT OF THIRD
# PARTY RIGHTS, INTEGRATION, MERCHANTABILITY, SATISFACTORY QUALITY OR FITNESS FOR ANY PARTICULAR PURPOSE.
# LIMITATION OF LIABILITY: IN NO EVENT WILL VIVANTE OR ITS SUPPLIERS BE LIABLE TO USER, USER'S CUSTOMERS,
# OR ANY OTHER PERSON OR ENTITY FOR ANY DAMAGES, CLAIMS OR COSTS WHATSOEVER ARISING FROM THIS AGREEMENT
# AND/OR YOUR USE OF THE SOFTWARE AND MATERIALS OR ANY COMPONENT THEREOF, INCLUDING WITHOUT LIMITATION
# ANY CONSEQUENTIAL, DIRECT, SPECIAL, PUNITIVE, INCIDENTAL, INDIRECT DAMAGES (WHETHER IN AN ACTION IN
# CONTRACT, TORT OR BASED ON WARRANTY), OR ANY LOST PROFITS OR LOST SAVINGS, EVEN IF A VIVANTE
# REPRESENTATIVE HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH LOSS, DAMAGES, CLAIMS OR COSTS OR FOR ANY
# CLAIM BY ANY THIRD PARTY. THE FOREGOING LIMITATIONS AND EXCLUSIONS APPLY TO THE EXTENT PERMITTED BY
# APPLICABLE LAW IN YOUR JURISDICTION. IN NO EVENT SHALL VIVANTE'S AGGREGATE LIABILITY TO USER OR ANY
# OTHER PERSON OR ENTITY CLAIMING THROUGH OR UNDER USER EXCEED THE AMOUNT OF MONEY ACTUALLY PAID BY USER
# TO VIVANTE FOR THE SOFTWARE OR ANY OTHER MATERIALS.

import json
import sys
import os
import dill
import math
#support build-in functions:
'''
    def build_port(self, ly_str, pt=0):
    def have_const_in_inputs(self, tensor):
    def have_single_tensor_in_inputs(self, tensor):
    def input_port_is_const(self, tensor, pt):
    def shape_pick(self, tensor):
    def attr_pick(self, tensor, key, default=0):
    def array_layout(self, array, layout):
    def tensor_to_numpy(self, tensor_name, trans=None):
    def squeeze_shapes(self, squeeze_dims, input_shape):
    def fc_weight(self, in_tensor, node):
    def split_slice_cale(self, slice):
    def map_pad_value(self, node):
    def reducex_axis_list(self, node, input_shape)
'''

ruler_list = list()

def rule_pyfunc_def(func):
    def _wrap_func(*args, **kwargs):
        src = dill.source.getsource(func)
        src = src.replace('\r\n', '\n')
        src = src.replace('@rule_pyfunc_def\n', '')
        src = src.split('\n')
        return ['__rule_func_additional_args = ' + json.dumps(kwargs)] + src if len(kwargs) > 0 else src
    return _wrap_func

@rule_pyfunc_def
def r_softmax_get_sf_axis(self, node, tensor):
    axis = self.attr_pick(node['Softmax'], 'axis', None)
    if axis is None:
        shape = self.shape_pick(tensor['I:out0'])
        if len(shape) == 4 and self.minor_version < 13:
            axis = 1
        else:
            axis = -1
    return axis

@rule_pyfunc_def
def r_softmax_get_log_sf_axis(self, node, tensor):
    axis = self.attr_pick(node['LogSoftmax'], 'axis', None)
    if axis is None:
        shape = self.shape_pick(tensor['I:out0'])
        if len(shape) == 4 and self.minor_version < 13:
            axis = 1
        else:
            axis = -1
    return axis

@rule_pyfunc_def
def r_batchnormal_v6_cond(self, node, tensor, name0, name1, name2, name3):
    shape0 = self.shape_pick(tensor[name0])
    shape1 = self.shape_pick(tensor[name1])
    shape2 = self.shape_pick(tensor[name2])
    shape3 = self.shape_pick(tensor[name3])
    s0=1
    s1=1
    s2=1
    s3=1
    for i in shape0:
        if i != 0:
            s0 *= i
    for i in shape1:
        if i != 0:
            s1 *= i
    for i in shape2:
        if i != 0:
            s2 *= i
    for i in shape3:
        if i != 0:
            s3 *= i
    return s0 == s1 and s1 == s2 and s2 == s3

@rule_pyfunc_def
def r_softmax_pre_cond(self, node, tensor):
    axis = self.attr_pick(node['Softmax'], 'axis', -1)
    input_rank = len(self.shape_pick(tensor['I:out0']))
    if axis < 0:
        axis = input_rank + axis
    if self.minor_version < 13:
        if input_rank > 2:
            if axis == input_rank - 1 or axis == 1:
                return True
        elif input_rank == 2:
            return True
        return False
    return True

@rule_pyfunc_def
def r_logsoftmax_pre_cond(self, node, tensor):
    axis = self.attr_pick(node['LogSoftmax'], 'axis', -1)
    input_rank = len(self.shape_pick(tensor['I:out0']))
    if axis < 0:
        axis = input_rank + axis
    if self.minor_version < 13:
        if input_rank > 2:
            if axis == input_rank - 1:
                return True
        elif input_rank == 2:
            return True
        return False
    return True

@rule_pyfunc_def
def r_slice_get_size(self, node, tensor):
    starts = self.attr_pick(node['Slice'], 'starts', None)
    ends = self.attr_pick(node['Slice'], 'ends', None)
    axes = self.attr_pick(node['Slice'], 'axes', None)
    in_shape = self.shape_pick(tensor['I:out0'])
    out_shape = self.shape_pick(tensor['Slice:out0'])

    import numpy as np
    import copy
    INT_MAX = np.iinfo(np.int64).max
    in_shape = copy.deepcopy(in_shape)
    ends = copy.deepcopy(ends)
    size = copy.deepcopy(in_shape)
    for i in range(len(axes)):
        if starts[i] < 0:
            starts[i] = in_shape[axes[i]] + starts[i]
        if ends[i] == INT_MAX:
            ends[i] = in_shape[axes[i]]
        elif ends[i] < 0:
            ends[i] = in_shape[axes[i]] + ends[i]
        size[axes[i]] = ends[i] - starts[i]
    return size

@rule_pyfunc_def
def r_slice_get_begin(self, node, tensor):
    in_shape = self.shape_pick(tensor['I:out0'])
    starts = self.attr_pick(node['Slice'], 'starts', None)
    axes = self.attr_pick(node['Slice'], 'axes', None)
    begin = [0] * len(in_shape)
    for i in range(len(axes)):
        if starts[i] < 0:
            starts[i] = in_shape[axes[i]] + starts[i]
        begin[axes[i]] = starts[i]
    return begin

@rule_pyfunc_def
def r_slice_pre_cond(self, node, tensor, steps_tensor):
    steps = self.tensor_to_numpy(tensor[steps_tensor]).tolist()
    for step in steps:
        if step != 1:
            return False
    return True

@rule_pyfunc_def
def r_get_deconv_weights(self, node, tensor, weight):
    in_channel = self.shape_pick(tensor[weight])[1]
    group = self.attr_pick(node['ConvTranspose'], 'group', 1)
    weights = in_channel * group
    return weights

@rule_pyfunc_def
def r_group_conv1d_pre_condition(self, node, tensor):
    ret = False
    if len(self.shape_pick(tensor['Constant_0:out0'])) == 3:
        in_shape = self.shape_pick(tensor['I:out0'])
        group_number = self.attr_pick(node['Conv'], 'group', 1)
        if group_number > 1:
            ret = True
    return ret

@rule_pyfunc_def
def r_depthwise_conv1d_pre_condition(self, node, tensor):
    ret = False
    if len(self.shape_pick(tensor['Constant_0:out0'])) == 3:
        in_shape = self.shape_pick(tensor['I:out0'])
        group_number = self.attr_pick(node['Conv'], 'group', 1)
        if group_number > 1 and group_number == in_shape[1]:
            ret = True
    return ret

@rule_pyfunc_def
def r_pad_value_map(self, node, tensor):
    pad_np = self.tensor_to_numpy(tensor['Constant:out0'])
    axes = tensor.get('Constant_2:out0', None)
    pads = list(pad_np)
    pads = [int(p) for p in pads]
    pads_array = list()
    if axes is not None:
        axes = self.tensor_to_numpy(tensor['Constant_2:out0'])
        in_rank = len(self.shape_pick(tensor["I_0:out0"]))
        pad_dims = len(axes)
        pad_index = 0
        for dim in range(in_rank):
            if dim not in axes:
                pad = [0, 0]
            else:
                pad = [pads[pad_index], pads[pad_dims + pad_index]]
                pad_index += 1
            pads_array.append(pad)
    else:
        dims = len(pads) // 2
        for id in range(dims):
            pad = [pads[id], pads[dims + id]]
            pads_array.append(pad)
    return pads_array

@rule_pyfunc_def
def r_pad_padding_const_map(self, node, tensor):
    import sys, numpy
    padding_const = self.tensor_to_numpy(tensor['Constant_1:out0'])
    if isinstance(padding_const, numpy.ndarray):
        padding_const = padding_const.tolist()[0]
    if padding_const < -sys.maxsize - 1:
        padding_const = -sys.maxsize - 1
    return padding_const

@rule_pyfunc_def
def r_pad_padding_const_condition(self, node, tensor):
    import sys
    padding_const = self.tensor_to_numpy(tensor['Constant_1:out0'])
    if padding_const < -sys.maxsize - 1:
        return False
    return True

@rule_pyfunc_def
def r_dconv_get_kernel_shape(self, node, tensor, weight, dim):
    kernel_shape = self.attr_pick(node['ConvTranspose'], 'kernel_shape')
    if not kernel_shape:
        kernel = self.tensor_to_numpy(tensor[weight])
        kernel_shape = kernel.shape
        ksize_h = kernel_shape[2]
        ksize_w = kernel_shape[3]
        if len(kernel_shape) == 5:
            ksize_d = kernel_shape[2]
            ksize_h = kernel_shape[3]
            ksize_w = kernel_shape[4]
    else:
        ksize_h = kernel_shape[0]
        ksize_w = kernel_shape[1]
        if len(kernel_shape) == 3:
            ksize_d = kernel_shape[0]
            ksize_h = kernel_shape[1]
            ksize_w = kernel_shape[2]

    if dim == 'height':
        return ksize_h
    if dim == 'width':
        return ksize_w
    if dim == 'depth':
        return ksize_d

@rule_pyfunc_def
def r_conv1d_get_kernel_shape(self, node, tensor, kernel_name):
    kernel_shape = self.attr_pick(node['Conv'], 'kernel_shape', None)
    if kernel_shape is None:
        kernel = self.tensor_to_numpy(tensor[kernel_name])
        kernel_shape = kernel.shape
        return kernel_shape[2]

    return kernel_shape[0]

@rule_pyfunc_def
def r_permute_value(self, node, tensor):
    in_shape = self.shape_pick(tensor['I:out0'])
    perm = self.attr_pick(node['Transpose'], 'perm', None)
    if perm is None:
        perm = list()
        for idx in range(len(in_shape)):
            perm.append(idx)
        perm.reverse()
    _perm = " ".join([str(x) for x in perm])
    return _perm

@rule_pyfunc_def
def r_resize_10_check(self, node, tensor):
    in_shape = self.shape_pick(tensor["I:out0"])
    out_shape = self.shape_pick(tensor["Resize:out0"])

    # acuity only support 3D or 4D resize
    if len(in_shape) < 2 or len(in_shape) > 4:
        return False
    # acuity only support resize width or height
    if in_shape[0] != out_shape[0] or in_shape[1] != out_shape[1]:
        return False

    return True

@rule_pyfunc_def
def r_resize_check(self, node, tensor):
    in_shape = self.shape_pick(tensor["I:out0"])
    out_shape = self.shape_pick(tensor["Resize:out0"])
    # acuity only support resize width or height
    if in_shape[0] != out_shape[0] or in_shape[1] != out_shape[1]:
        return False

    unsuppored_trans_mode = [
        #'pytorch_half_pixel',
        #for pytorch_half_pixel, we assue that length_resized will always > 1,
        #in this condition, it equals to 'half_pixel',
        #but if length_resized == 1, there will be some precision issue
        'tf_half_piexl_for_nn',
        'tf_crop_and_resize'
    ]
    trans_mode = self.attr_pick(node['Resize'], 'coordinate_transformation_mode', 'half_pixel')
    if trans_mode in unsuppored_trans_mode:
        return False

    mode = self.attr_pick(node['Resize'], 'mode', 'nearest')
    nearest_mode = self.attr_pick(node['Resize'], 'nearest_mode', 'round_prefer_floor')
    if mode == 'nearest' and 'ceil' in nearest_mode:
        return False

    # pytorch coeff_a is -0.75
    # tf coeff_a is -0.5, we only support this coeff_a
    coeff_a = self.attr_pick(node['Resize'], 'cubic_coeff_a', -0.75)
    if mode == 'cubic' and coeff_a != -0.5:
        return False

    return True

@rule_pyfunc_def
def r_resize_get_new_size(self, node, tensor):
    out_shape = self.shape_pick(tensor["Resize:out0"])
    new_size = out_shape[2:] # [batch, channel, height, width] or [batch, channel, width]
    return new_size

@rule_pyfunc_def
def r_resize_get_type(self, node, tensor):
    mode = self.attr_pick(node['Resize'], 'mode', 'nearest').lower()
    _mode_map = {
        "nearest": "nearest",
        "linear": "bilinear",
        "cubic": "bicubic"
    }

    _maped_mode = "nearest"
    if mode in _mode_map.keys():
        _maped_mode = _mode_map[mode]
    return _maped_mode

@rule_pyfunc_def
def r_resize_get_align_corners(self, node, tensor):
    trans_mode = self.attr_pick(node['Resize'], 'coordinate_transformation_mode', 'half_pixel')
    if trans_mode == 'align_corners':
        return True
    return False

@rule_pyfunc_def
def r_resize_get_half_pixel(self, node, tensor):
    trans_mode = self.attr_pick(node['Resize'], 'coordinate_transformation_mode', 'half_pixel')
    # for pytorch_half_pixel, we assue that length_resized will always > 1,
    # in this condition, it equals to 'half_pixel',
    # but if length_resized == 1, there will be some precision issue
    if trans_mode in ['half_pixel', 'pytorch_half_pixel']:
        return True
    return False

@rule_pyfunc_def
def r_mm_check_wb(self, node, tensor, in_tensor, weight, bias = None):
    input_shape = self.shape_pick(tensor[in_tensor])
    weight_shape = self.shape_pick(tensor[weight])
    if len(input_shape) != 2 or len(weight_shape) != 2:
        return False

    if bias is not None:
        bias_shape = self.shape_pick(tensor[bias])
        weights = self.shape_pick(tensor[weight])[1]
        if len(bias_shape) != 1 or weights != bias_shape[0]:
            return False
    return True

@rule_pyfunc_def
def r_upsample_to_resize_check(self, node, tensor, img_in, img_out, expand_in, expand_out):
    img_in_shape = self.shape_pick(tensor[img_in])
    img_out_shape = self.shape_pick(tensor[img_out])
    expand_in_shape = self.shape_pick(tensor[expand_in])
    expand_out_shape = self.shape_pick(tensor[expand_out])
    if len(img_in_shape) == 4 and \
        len(img_out_shape) == 4 and \
        len(expand_in_shape) == 6 and \
        len(expand_out_shape) == 6 and \
        img_in_shape[0] == img_out_shape[0] and \
        img_in_shape[1] == img_out_shape[1] and \
        expand_in_shape[2] == expand_out_shape[2] and \
        expand_in_shape[4] == expand_out_shape[4]:
        return True
    return False

@rule_pyfunc_def
def r_pool_padding(self, node, tensor, pool_type):
    assert pool_type in ['AveragePool', 'MaxPool']
    auto_pad = self.attr_pick(node[pool_type], 'auto_pad', None)
    if auto_pad in ['SAME_UPPER']:
        return 'SAME'
    elif auto_pad in ['SAME_LOWER']:
        return 'SAME_LOWER'
    else:
        return 'VALID'

@rule_pyfunc_def
def r_expand_broadcast_shape(self, node, tensor, in_tensor):
    import copy
    shape = self.tensor_to_numpy(tensor['Constant:out0']).tolist()
    in_shape = self.shape_pick(tensor[in_tensor])
    rank0 = len(shape)
    rank1 = len(in_shape)
    rank = max(rank0, rank1)
    if rank0 != rank1:
        shape0 = copy.deepcopy(shape)
        shape1 = copy.deepcopy(in_shape)
        shape0.reverse()
        shape1.reverse()
        dim0 = len(shape0)
        dim1 = len(shape1)
        braodcast_shape = list()
        for idx in range(rank):
            if idx < dim0:
                s0 = shape0[idx]
            else:
                s0 = 1
            if idx < dim1:
                s1 = shape1[idx]
            else:
                s1 = 1
            s = max(s0, s1)
            braodcast_shape.append(s)
        braodcast_shape.reverse()
        return braodcast_shape
    else:
        return shape

@rule_pyfunc_def
def r_center_crop_pad_shape(self, node, tensor, in_tensor):
    import copy
    shape = self.tensor_to_numpy(tensor['Constant:out0']).tolist()
    in_shape = self.shape_pick(tensor[in_tensor])
    axes = self.attr_pick(node['CenterCropPad'], 'axes', None)
    if axes:
        out_shape = copy.deepcopy(in_shape)
        for i, axis in enumerate(axes):
            out_shape[axis] = shape[i]
    else:
        out_shape = shape
    return out_shape


@rule_pyfunc_def
def r_col2im_pads(self, node, tensor):
    pads = self.attr_pick(node['Col2Im'], 'pads', None)
    if pads:
        pads_value = [[pads[0], pads[2]], [pads[1], pads[3]]]
    else:
        pads_value = [[0, 0], [0, 0]]
    return pads_value


r_variable = {
"ruler_name": "r_variable",
"src_ops_alias": ["Constant"],
"src_inter_flow": [],
"src_in_anchor": [],
"src_out_tensor": ["Constant:out0"],
"acu_lys_alias": ["variable"],
"src_acu_in_tensor_map": [],
"src_acu_out_tensor_map": [["Constant:out0", "variable:out0"]],
"param_map": {"variable": {'shape': ['ORIGIN', 'CODE', "self.shape_pick(tensor['Constant:out0'])"],
                           'is_scalar': ['BOOL', 'CODE',
                           "True if len(self.tensor_to_numpy_without_convert_0darry(tensor['Constant:out0']).shape) "
                           "== 0 else False "],
                           'type': ["STRING", "CODE", "self.dtype_pick(tensor['Constant:out0'])"],
                           }},
"blob_map": {"variable": {'data':
                              ['CODE',
                               "np.array([self.tensor_to_numpy(tensor['Constant:out0'])]) "\
                               " if self.tensor_to_numpy(tensor['Constant:out0']).shape == () "\
                               "else self.tensor_to_numpy(tensor['Constant:out0'])"],}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_variable)

r_rsp_mm_add = {
"ruler_name": "r_rsp_mm_add",
"src_ops_alias": ["Reshape", "MatMul", "Add", "Constant_0", "Constant_1"],
"src_inter_flow":
    [["Reshape:out0", "MatMul:in0"], ["MatMul:out0", "Add:in0"], ["Constant_0:out0", "MatMul:in1"],
     ["Constant_1:out0", "Add:in1"]],
"src_in_anchor": [["I:out0", "Reshape:in0"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["fullconnect"],
"src_acu_in_tensor_map": [["I:out0", "fullconnect:in0"]],
"src_acu_out_tensor_map": [["Add:out0", "fullconnect:out0"]],
"param_map": {"fullconnect":
                  {"weights": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[1]"],
                   "bias": ["BOOL", "VALUE", True]}},
"blob_map": {"fullconnect":
                 {"weight": ["CODE", "self.matmul_weight(tensor['Constant_0:out0'])"],
                  "bias": ["CODE", "self.tensor_to_numpy(tensor['Constant_1:out0'])"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": r_mm_check_wb(in_tensor='Reshape:out0', weight='Constant_0:out0', bias='Constant_1:out0'),
"src_ops_main_version": None,
"src_ops_minior_version": [1, 4]}
# ruler_list.append(r_rsp_mm_add)

r_qlinearmatmul_qlinearadd_to_fc = {
"ruler_name": "qlinearmatmul_qlinearadd_to_fc",
"src_ops_alias": ["QLinearAdd", "QLinearMatMul", "Constant", "Constant_1", "Constant_2",
                  "Constant_3", "Constant_4", "Constant_5", "Constant_6", "Constant_7",
                  "Constant_8", "Constant_9", "Constant_10", "Constant_11"],
"src_inter_flow": [["QLinearMatMul:out0", "QLinearAdd:in0"], ["Constant:out0", "QLinearAdd:in1"],
                   ["Constant_1:out0", "QLinearAdd:in2"], ["Constant_2:out0", "QLinearAdd:in3"],
                   ["Constant_3:out0", "QLinearAdd:in4"], ["Constant_4:out0", "QLinearAdd:in5"],
                   ["Constant_5:out0", "QLinearAdd:in6"], ["Constant_6:out0", "QLinearAdd:in7"],
                   ["Constant_7:out0", "QLinearMatMul:in1"], ["Constant_8:out0", "QLinearMatMul:in2"],
                   ["Constant_9:out0", "QLinearMatMul:in3"], ["Constant_10:out0", "QLinearMatMul:in4"],
                   ["Constant_11:out0", "QLinearMatMul:in5"], ["Constant:out0", "QLinearMatMul:in6"],
                   ["Constant_1:out0", "QLinearMatMul:in7"]],
"src_in_anchor": [["I_0:out0", "QLinearMatMul:in0"]],
"src_out_tensor": ["QLinearAdd:out0"],
"acu_lys_alias": ["fullconnect"],
"src_acu_in_tensor_map": [["I_0:out0", "fullconnect:in0"]],
"src_acu_out_tensor_map": [["QLinearAdd:out0", "fullconnect:out0"]],
"acu_inter_flow": [],
"param_map": {"fullconnect": {"bias": ["BOOL", "VALUE", True],
                              "weights": ["INT", "CODE", "self.tensor_to_numpy(tensor['Constant_7:out0']).shape[-1]"],
                              "axis": ["INT", "VALUE", 1]
                              }},
"blob_map": {"fullconnect": {"weight": ["CODE", "self.dequant_matmul_weight("
                                        "tensor['Constant_7:out0'],"
                                        "self.tensor_to_numpy(tensor['Constant_10:out0']),"
                                        "self.tensor_to_numpy(tensor['Constant_11:out0']))"],
                             "bias": ["CODE", "self.dequant_matmul_weight("
                                        "tensor['Constant_0:out0'],"
                                        "self.tensor_to_numpy(tensor['Constant_3:out0']),"
                                        "self.tensor_to_numpy(tensor['Constant_4:out0']))"]}},
"extension": [
    ["CODE", "self.qnt_coef_tensor("
             "'weight', acu_ly['fullconnect'],"
             "tensor['Constant_10:out0'],"
             "tensor['Constant_11:out0'])"],
    ["CODE", "self.qnt_coef_tensor("
             "'bias', acu_ly['fullconnect'],"
             "tensor['Constant_3:out0'],"
             "tensor['Constant_4:out0'])"],
    ["CODE", "self.qnt_out_tensor(acu_ly['fullconnect'], tensor['Constant_5:out0'], tensor['Constant_6:out0'], 0)"],
    ["CODE", "self.qnt_in_tensor(acu_ly['fullconnect'], tensor['Constant:out0'], tensor['Constant_1:out0'], 0)"]
],
"priority_tip": 0,
"pre_condition": "self.tensor_to_numpy(tensor['Constant_3:out0']).shape[0] > 1",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
# ruler_list.append(r_qlinearmatmul_qlinearadd_to_fc)

r_qlinearmatmul_to_fc = {
"ruler_name": "qlinearmatmul_to_fc",
"src_ops_alias": ["QLinearMatMul", "Constant", "Constant_1", "Constant_2", "Constant_3",
                  "Constant_4", "Constant_5", "Constant_6"],
"src_inter_flow": [["Constant:out0", "QLinearMatMul:in1"], ["Constant_1:out0", "QLinearMatMul:in2"],
                   ["Constant_2:out0", "QLinearMatMul:in3"], ["Constant_3:out0", "QLinearMatMul:in4"],
                   ["Constant_4:out0", "QLinearMatMul:in5"], ["Constant_5:out0", "QLinearMatMul:in6"],
                   ["Constant_6:out0", "QLinearMatMul:in7"]],
"src_in_anchor": [["I_0:out0", "QLinearMatMul:in0"]],
"src_out_tensor": ["QLinearMatMul:out0"],
"acu_lys_alias": ["fullconnect"],
"src_acu_in_tensor_map": [["I_0:out0", "fullconnect:in0"]],
"src_acu_out_tensor_map": [["QLinearMatMul:out0", "fullconnect:out0"]],
"acu_inter_flow": [],
"param_map": {"fullconnect": {"bias": ["BOOL", "VALUE", False],
                              "weights": ["INT", "CODE", "self.tensor_to_numpy(tensor['Constant_2:out0']).shape[-1]"],
                              "axis": ["INT", "VALUE", 1]
                              }},
"blob_map": {"fullconnect": {"weight": ["CODE", "self.dequant_matmul_weight("
                                        "tensor['Constant_2:out0'],"
                                        "self.tensor_to_numpy(tensor['Constant_3:out0']),"
                                        "self.tensor_to_numpy(tensor['Constant_4:out0']))"],
                            }},
"extension": [
    ["CODE", "self.qnt_coef_tensor("
             "'weight', acu_ly['fullconnect'],"
             "tensor['Constant_3:out0'],"
             "tensor['Constant_4:out0'])"],
    ["CODE", "self.qnt_out_tensor(acu_ly['fullconnect'], tensor['Constant_5:out0'], tensor['Constant_6:out0'], 0)"],
    ["CODE", "self.qnt_in_tensor(acu_ly['fullconnect'], tensor['Constant:out0'], tensor['Constant_1:out0'], 0)"]
],
"priority_tip": 0,
"pre_condition": "self.tensor_to_numpy(tensor['Constant_3:out0']).shape[0] > 1",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
# QLinearMatMul:QLinearMatMul_Gemm_104_MatMul_quant;
# Constant:Initializer_464_scale;
# Constant_1:Initializer_464_zero_point;
# Constant_2:Initializer_classifier.1.weight_quantized;Constant_3:Initializer_classifier.1.weight_scale;
# Constant_4:Initializer_classifier.1.weight_zero_point;Constant_5:Initializer_output_MatMul_scale;
# Constant_6:Initializer_output_MatMul_zero_point
ruler_list.append(r_qlinearmatmul_to_fc)

r_rsp_mm_add_v5 = {
"ruler_name": "r_rsp_mm_add_v5",
"src_ops_alias": ["Reshape", "MatMul", "Add", "Constant_0", "Constant_1", "Constant_2"],
"src_inter_flow":
    [["Reshape:out0", "MatMul:in0"], ["MatMul:out0", "Add:in0"], ["Constant_0:out0", "MatMul:in1"],
     ["Constant_1:out0", "Add:in1"], ["Constant_2:out0", "Reshape:in1"]],
"src_in_anchor": [["I:out0", "Reshape:in0"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["fullconnect"],
"src_acu_in_tensor_map": [["I:out0", "fullconnect:in0"]],
"src_acu_out_tensor_map": [["Add:out0", "fullconnect:out0"]],
"param_map":
    {"fullconnect":
         {"weights": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[1]"],
          "bias": ["BOOL", "VALUE", True]}},
"blob_map":
    {"fullconnect":
         {"weight": ["CODE", "self.matmul_weight(tensor['Constant_0:out0'])"],
          "bias": ["CODE", "self.tensor_to_numpy(tensor['Constant_1:out0'])"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": r_mm_check_wb(in_tensor='Reshape:out0', weight='Constant_0:out0', bias='Constant_1:out0'),
"src_ops_main_version": None,
"src_ops_minior_version": [5, -1]}
# ruler_list.append(r_rsp_mm_add_v5)

r_mm_add = {
"ruler_name": "r_mm_add",
"src_ops_alias": ["MatMul", "Add", "Constant_0", "Constant_1"],
"src_inter_flow": [["MatMul:out0", "Add:in0"], ["Constant_0:out0", "MatMul:in1"], ["Constant_1:out0", "Add:in1"]],
"src_in_anchor": [["I:out0", "MatMul:in0"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["fullconnect"],
"src_acu_in_tensor_map": [["I:out0", "fullconnect:in0"]],
"src_acu_out_tensor_map": [["Add:out0", "fullconnect:out0"]],
"param_map": {"fullconnect":
                  {"weights": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[1]"],
                   "bias": ["BOOL", "VALUE", True]}},
"blob_map": {"fullconnect":
                 {"weight": ["CODE", "self.matmul_weight(tensor['Constant_0:out0'])"],
                  "bias": ["CODE", "self.tensor_to_numpy(tensor['Constant_1:out0'])"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": r_mm_check_wb(in_tensor='I:out0', weight='Constant_0:out0', bias='Constant_1:out0'),
"src_ops_main_version": None,
"src_ops_minior_version": None}
ruler_list.append(r_mm_add)

r_mm = {
"ruler_name": "r_mm",
"src_ops_alias": ["MatMul", "Constant_0"],
"src_inter_flow": [["Constant_0:out0", "MatMul:in1"]],
"src_in_anchor": [["I:out0", "MatMul:in0"]],
"src_out_tensor": ["MatMul:out0"],
"acu_lys_alias": ["fullconnect"],
"src_acu_in_tensor_map": [["I:out0", "fullconnect:in0"]],
"src_acu_out_tensor_map": [["MatMul:out0", "fullconnect:out0"]],
"param_map": {"fullconnect":
                  {"weights": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[1]"],
                   "bias": ["BOOL", "VALUE", False]}},
"blob_map": {"fullconnect":
                 {"weight": ["CODE", "self.matmul_weight(tensor['Constant_0:out0'])"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": r_mm_check_wb(in_tensor='I:out0', weight='Constant_0:out0'),
"src_ops_main_version": None,
"src_ops_minior_version": None}
ruler_list.append(r_mm)

r_gemm = {
"ruler_name": "r_gemm",
"src_ops_alias": ["Gemm"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Gemm:in0"], ['I_1:out0', "Gemm:in1"]],
"src_out_tensor": ["Gemm:out0"],
"acu_lys_alias": ["matmul"],
"src_acu_in_tensor_map":[["I:out0", "matmul:in0"], ['I_1:out0', "matmul:in1"]],
"src_acu_out_tensor_map": [["Gemm:out0", "matmul:out0"]],
"param_map":{
    "matmul":{
        'transpose_a': ['BOOL', 'CODE', "False if self.attr_pick(node['Gemm'], 'transA', 0) == 0 else True"],
        'transpose_b': ['BOOL', 'CODE', "False if self.attr_pick(node['Gemm'], 'transB', 0) == 0 else True"],
    }
},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_gemm)

r_gemm_3_inputs = {
"ruler_name": "r_gemm_3_inputs",
"src_ops_alias": ["Gemm", "Constant"],
"src_inter_flow": [["Constant:out0", "Gemm:in1"]],
"src_in_anchor": [["I_0:out0", "Gemm:in0"], ["I_1:out0", "Gemm:in2"]],
"src_out_tensor": ["Gemm:out0"],
"acu_lys_alias": ["matmul", "add", "variable"],
"src_acu_in_tensor_map":[["I_0:out0", "matmul:in0"], ['I_1:out0', "add:in1"]],
"src_acu_out_tensor_map": [["Gemm:out0", "add:out0"]],
"acu_inter_flow": [["variable:out0", "matmul:in1"], ["matmul:out0", "add:in0"]],
"param_map":{
    "matmul":{
        'transpose_a': ['BOOL', 'CODE', "False if self.attr_pick(node['Gemm'], 'transA', 0) == 0 else True"],
        'transpose_b': ['BOOL', 'CODE', "False if self.attr_pick(node['Gemm'], 'transB', 0) == 0 else True"],
    },
    "variable": {
        'shape': ['ORIGIN', 'CODE', "self.shape_pick(tensor['Constant:out0'])"],
    }
},
"blob_map": {
    "variable": {
        'data': ["CODE", "self.tensor_to_numpy(tensor['Constant:out0']) * self.attr_pick(node['Gemm'], 'alpha', 1)"]
    }
},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_gemm_3_inputs)

r_gemm_bias = {
"ruler_name": "r_gemm_bias",
"src_ops_alias": ["Gemm", "Constant"],
"src_inter_flow": [["Constant:out0", "Gemm:in2"]],
"src_in_anchor": [["I_0:out0", "Gemm:in0"], ["I_1:out0", "Gemm:in1"]],
"src_out_tensor": ["Gemm:out0"],
"acu_lys_alias": ["matmul", "add", "variable"],
"src_acu_in_tensor_map": [["I_0:out0", "matmul:in0"], ["I_1:out0", "matmul:in1"]],
"src_acu_out_tensor_map": [["Gemm:out0", "add:out0"]],
"acu_inter_flow": [["matmul:out0", "add:in0"], ["variable:out0", "add:in1"]],
"param_map": {
    "matmul": {
        'transpose_a': ['BOOL', 'CODE', "False if self.attr_pick(node['Gemm'], 'transA', 0) == 0 else True"],
        'transpose_b': ['BOOL', 'CODE', "False if self.attr_pick(node['Gemm'], 'transB', 0) == 0 else True"],
    },
    "variable": {
        'shape': ['ORIGIN', 'CODE', "self.shape_pick(tensor['Constant:out0'])"],
    },
    "add": {}
},
"blob_map": {
    "matmul": {},
    "add": {},
    "variable": {
        'data': ["CODE", "self.tensor_to_numpy(tensor['Constant:out0']) * self.attr_pick(node['Gemm'], 'beta', 0)"]
    }
},
"priority_tip": 0,
"pre_condition": "self.attr_pick(node['Gemm'], 'alpha', 0) == 1 and self.attr_pick(node['Gemm'], 'beta', 0) != 0",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_gemm_bias)

r_gemm_2_matmul = {
"ruler_name": "r_gemm_2_matmul",
"src_ops_alias": ["Gemm", "Constant"],
"src_inter_flow": [["Constant:out0", "Gemm:in2"]],
"src_in_anchor": [["I:out0", "Gemm:in0"], ['I_1:out0', "Gemm:in1"]],
"src_out_tensor": ["Gemm:out0"],
"acu_lys_alias": ["matmul"],
"src_acu_in_tensor_map":[["I:out0", "matmul:in0"], ['I_1:out0', "matmul:in1"]],
"src_acu_out_tensor_map": [["Gemm:out0", "matmul:out0"]],
"param_map":{
    "matmul":{
        'transpose_a': ['BOOL', 'CODE', "False if self.attr_pick(node['Gemm'], 'transA', 0) == 0 else True"],
        'transpose_b': ['BOOL', 'CODE', "False if self.attr_pick(node['Gemm'], 'transB', 0) == 0 else True"],
    }
},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "self.attr_pick(node['Gemm'], 'alpha', 0) == 1 and "
                 "self.attr_pick(node['Gemm'], 'beta', 0) == 0",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_gemm_2_matmul)

r_gemm_2_fc = {
"ruler_name": "r_gemm_2_fc",
"src_ops_alias": ["Gemm", "Constant_0"],
"src_inter_flow": [["Constant_0:out0", "Gemm:in1"]],
"src_in_anchor": [["I:out0", "Gemm:in0"]],
"src_out_tensor": ["Gemm:out0"],
"acu_lys_alias": ["fullconnect"],
"src_acu_in_tensor_map": [["I:out0", "fullconnect:in0"]],
"src_acu_out_tensor_map": [["Gemm:out0", "fullconnect:out0"]],
"param_map": {"fullconnect":
                  {"weights": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[0]"],
                   "bias": ["BOOL", "VALUE", "False"]}},
"blob_map": {"fullconnect": {"weight": ["CODE", "self.fc_weight(tensor['Constant_0:out0'], node['Gemm'])"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition":
    "len(self.query_inputs(node['Gemm'])) == 2 and "\
    "self.attr_pick(node['Gemm'], 'transA', 0) == 0 and "\
    "self.attr_pick(node['Gemm'], 'transB', 0) == 1",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_gemm_2_fc)

r_gemm_2_fc_wb = {
"ruler_name": "r_gemm_2_fc_wb",
"src_ops_alias": ["Gemm", "Constant_0", "Constant_1"],
"src_inter_flow": [["Constant_0:out0", "Gemm:in1"], ["Constant_1:out0", "Gemm:in2"]],
"src_in_anchor": [["I:out0", "Gemm:in0"]],
"src_out_tensor": ["Gemm:out0"],
"acu_lys_alias": ["fullconnect"],
"src_acu_in_tensor_map": [["I:out0", "fullconnect:in0"]],
"src_acu_out_tensor_map": [["Gemm:out0", "fullconnect:out0"]],
"param_map": {"fullconnect":
                  {"weights": ["INT", "CODE",
                               "self.gemm_weights_param(tensor['Constant_0:out0'], node['Gemm'], 'transB')"],
                   "bias": ["BOOL", "VALUE", True]}},
"blob_map":
    {"fullconnect":
         {"weight": ["CODE",
                     "self.gemm_weight_blob(tensor['Constant_0:out0'], node['Gemm'], 'transB')"],
          "bias":
              ["CODE",
               "self.tensor_to_numpy(tensor['Constant_1:out0']) * self.attr_pick(node['Gemm'], 'beta', 1.0)"]
          }
     },
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition":
    "len(self.query_inputs(node['Gemm'])) == 3 and "\
    "self.attr_pick(node['Gemm'], 'transA', 0) == 0 and "\
    "self.attr_pick(node['Gemm'], 'transB', 0) == 1",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_gemm_2_fc_wb)

r_gemm_2_fc_wb_notranspose = {
"ruler_name": "gemm_2_fc_notranspose",
"src_ops_alias": ["Gemm", "Constant", "Constant_1"],
"src_inter_flow": [["Constant:out0", "Gemm:in1"], ["Constant_1:out0", "Gemm:in2"]],
"src_in_anchor": [["I_0:out0", "Gemm:in0"]],
"src_out_tensor": ["Gemm:out0"],
"acu_lys_alias": ["fullconnect"],
"src_acu_in_tensor_map": [["I_0:out0", "fullconnect:in0"]],
"src_acu_out_tensor_map": [["Gemm:out0", "fullconnect:out0"]],
"acu_inter_flow": [],
"param_map": {
        "fullconnect":{
            "weights": ["INT", "CODE", "self.gemm_weights_param(tensor['Constant:out0'], node['Gemm'], 'transB')"],
            "bias": ["BOOL", "VALUE", True]
        }
    },
"blob_map":
    {"fullconnect":
         {"weight":
              ["CODE",
               "self.gemm_weight_blob(tensor['Constant:out0'], node['Gemm'], 'transB')"],
          "bias":
              ["CODE",
               "self.tensor_to_numpy(tensor['Constant_1:out0']) * self.attr_pick(node['Gemm'], 'beta', 1.0)"]
          }
     },
"priority_tip": 0,
"pre_condition": "self.attr_pick(node['Gemm'], 'transA', 0) == 0 and"\
    " self.attr_pick(node['Gemm'], 'transB', 0) == 0",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_gemm_2_fc_wb_notranspose)

r_gemm_2_fc_4d_wb_notranspose = {
"ruler_name": "r_gemm_2_fc_4d_wb_notranspose",
"src_ops_alias": ["Gemm", "Reshape", "Reshape_1", "Constant", "Constant_1"],
"src_inter_flow": [["Reshape:out0", "Gemm:in0"], ["Reshape_1:out0", "Gemm:in1"], ["Constant:out0", "Gemm:in2"],
                   ["Constant_1:out0", "Reshape_1:in0"]],
"src_in_anchor": [["I_0:out0", "Reshape:in0"]],
"src_out_tensor": ["Gemm:out0"],
"acu_lys_alias": ["fullconnect"],
"src_acu_in_tensor_map": [["I_0:out0", "fullconnect:in0"]],
"src_acu_out_tensor_map": [["Gemm:out0", "fullconnect:out0"]],
"acu_inter_flow": [],
"param_map": {"fullconnect":
                  {"weights": ["INT", "CODE",
                               "self.gemm_weights_param(tensor['Constant:out0'], node['Gemm'], 'transB')"],
                   "bias": ["BOOL", "VALUE", True]}},
"blob_map":
    {"fullconnect":
         {"weight":
              ["CODE",
               "self.gemm_weight_blob(tensor['Reshape_1:out0'], node['Gemm'], 'transB')"],
          "bias":
              ["CODE",
               "self.tensor_to_numpy(tensor['Constant:out0']) * self.attr_pick(node['Gemm'], 'beta', 1.0)"]
          }
     },
"priority_tip": 0,
"pre_condition": "self.attr_pick(node['Gemm'], 'transA', 0) == 0 and"\
    " self.attr_pick(node['Gemm'], 'transB', 0) == 0",
"src_ops_main_version": None,
"src_ops_minior_version": [1, 5]}
ruler_list.append(r_gemm_2_fc_4d_wb_notranspose)

r_gemm_2_fc_4d_wb_notranspose_v5 = {
"ruler_name": "r_gemm_2_fc_4d_wb_notranspose_v5",
"src_ops_alias": ["Gemm", "Reshape", "Reshape_1", "Constant", "Constant_1", "Constant_2", "Constant_3"],
"src_inter_flow": [["Reshape:out0", "Gemm:in0"], ["Reshape_1:out0", "Gemm:in1"], ["Constant:out0", "Gemm:in2"],
                   ["Constant_1:out0", "Reshape:in1"], ["Constant_2:out0", "Reshape_1:in0"],
                   ["Constant_3:out0", "Reshape_1:in1"]],
"src_in_anchor": [["I_0:out0", "Reshape:in0"]],
"src_out_tensor": ["Gemm:out0"],
"acu_lys_alias": ["fullconnect"],
"src_acu_in_tensor_map": [["I_0:out0", "fullconnect:in0"]],
"src_acu_out_tensor_map": [["Gemm:out0", "fullconnect:out0"]],
"acu_inter_flow": [],
"param_map": {"fullconnect":
                  {"weights": ["INT", "CODE",
                               "self.gemm_weights_param(tensor['Reshape_1:out0'], node['Gemm'], 'transB')"],
                   "bias": ["BOOL", "VALUE", True]}},
"blob_map":
    {"fullconnect":
         {"weight":
              ["CODE",
               "self.gemm_weight_blob(tensor['Constant_2:out0'], node['Gemm'], 'transB', \
                    self.shape_pick(tensor['Reshape_1:out0']))"],
          "bias":
              ["CODE",
               "self.tensor_to_numpy(tensor['Constant:out0']) * self.attr_pick(node['Gemm'], 'beta', 1.0)"]
          }
     },
"priority_tip": 0,
"pre_condition": "self.attr_pick(node['Gemm'], 'transA', 0) == 0 and"\
    " self.attr_pick(node['Gemm'], 'transB', 0) == 0",
"src_ops_main_version": None,
"src_ops_minior_version": [5, -1]}
ruler_list.append(r_gemm_2_fc_4d_wb_notranspose_v5)

r_gemm_2_fc_wb_bc = {
"ruler_name": "r_gemm_2_fc_wb_bc",
"src_ops_alias": ["Gemm", "Constant_0", "Constant_1"],
"src_inter_flow": [["Constant_0:out0", "Gemm:in1"], ["Constant_1:out0", "Gemm:in2"]],
"src_in_anchor": [["I:out0", "Gemm:in0"]],
"src_out_tensor": ["Gemm:out0"],
"acu_lys_alias": ["fullconnect"],
"src_acu_in_tensor_map": [["I:out0", "fullconnect:in0"]],
"src_acu_out_tensor_map": [["Gemm:out0", "fullconnect:out0"]],
"param_map": {"fullconnect":
                  {"weights": ["INT", "CODE",
                               "self.gemm_weights_param(tensor['Constant_0:out0'], node['Gemm'], 'transB')"],
                   "bias": ["BOOL", "VALUE", True]}},
"blob_map":
    {"fullconnect":
         {"weight": ["CODE", "self.gemm_weight_blob(tensor['Constant_0:out0'], node['Gemm'], 'transB')"],
          "bias":
              ["CODE",
               "np.ones(self.shape_pick(tensor['Constant_0:out0'])[0], dtype=np.float32)*\
               self.tensor_to_numpy(tensor['Constant_1:out0']) * self.attr_pick(node['Gemm'], 'beta', 1.0)"]
          }
     },
"acu_inter_flow": [],
"priority_tip": 1,
"pre_condition":
    "len(self.query_inputs(node['Gemm'])) == 3 and "\
    "self.attr_pick(node['Gemm'], 'transA', 0) == 0 and "\
    "self.attr_pick(node['Gemm'], 'transB', 0) == 1 and "\
    "self.shape_pick(tensor['Constant_1:out0'])[0] == 1 ",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_gemm_2_fc_wb_bc)

r_fullconnect_with_wbrsp = {
"ruler_name": "r_fullconnect_with_wbrsp",
"src_ops_alias": ["Gemm", "Reshape", "Constant", "Constant_1"],
"src_inter_flow": [["Reshape:out0", "Gemm:in1"], ["Constant:out0", "Gemm:in2"], ["Constant_1:out0", "Reshape:in0"]],
"src_in_anchor": [["I_0:out0", "Gemm:in0"]],
"src_out_tensor": ["Gemm:out0"],
"acu_lys_alias": ["fullconnect"],
"src_acu_in_tensor_map": [["I_0:out0", "fullconnect:in0"]],
"src_acu_out_tensor_map": [["Gemm:out0", "fullconnect:out0"]],
"acu_inter_flow": [],
"param_map": {"fullconnect":
                  {"weights": ["INT", "CODE",
                               "self.gemm_weights_param(tensor['Reshape:out0'], node['Gemm'], 'transB')"],
                   "bias": ["BOOL", "VALUE", True]}},
"blob_map":
    {"fullconnect":
         {"weight":
              ["CODE",
               "self.gemm_weight_blob(tensor['Constant_1:out0'], node['Gemm'], 'transB', \
                    self.shape_pick(tensor['Reshape:out0']))"
               ],
          "bias":
              ["CODE",
               "self.tensor_to_numpy(tensor['Constant:out0']) * self.attr_pick(node['Gemm'], 'beta', 1.0)"]
          }
     },
"priority_tip": 0,
"pre_condition":
    "len(self.query_inputs(node['Gemm'])) == 3 and "\
    "self.attr_pick(node['Gemm'], 'transA', 0) == 0 and "\
    "self.attr_pick(node['Gemm'], 'transB', 0) == 1",
"src_ops_main_version": None,
"src_ops_minior_version": [1, 4]}
#Gemm:Gemm_141;Reshape:Reshape_140;Constant:Initializer_115;Constant_1:Initializer_114
ruler_list.append(r_fullconnect_with_wbrsp)

r_fullconnect_with_weight_at_in0_with_reshape = {
"ruler_name": "fc_w@in0_t@in1_with_reshape",
"src_ops_alias": ["Gemm", "Reshape", "Constant", "Constant_1", "Constant_2"],
"src_inter_flow": [["Reshape:out0", "Gemm:in1"], ["Constant:out0", "Gemm:in2"],
                   ["Constant_1:out0", "Reshape:in0"], ["Constant_2:out0", "Reshape:in1"]],
"src_in_anchor": [["I_0:out0", "Gemm:in0"]],
"src_out_tensor": ["Gemm:out0"],
"acu_lys_alias": ["fullconnect"],
"src_acu_in_tensor_map": [["I_0:out0", "fullconnect:in0"]],
"src_acu_out_tensor_map": [["Gemm:out0", "fullconnect:out0"]],
"acu_inter_flow": [],
"param_map": {"fullconnect":
                  {"weights": ["INT", "CODE",
                               "self.gemm_weights_param(tensor['Reshape:out0'], node['Gemm'], 'transB')"],
                   "bias": ["BOOL", "VALUE", True]}},
"blob_map":
    {"fullconnect":
         {"weight":
              ["CODE",
               "self.gemm_weight_blob(tensor['Constant_1:out0'], node['Gemm'], 'transB', \
                    self.shape_pick(tensor['Reshape:out0']))"
               ],
          "bias":
              ["CODE",
               "self.tensor_to_numpy(tensor['Constant:out0']) * self.attr_pick(node['Gemm'], 'beta', 1.0)"]
          }
     },
"priority_tip": 0,
"pre_condition":
    "len(self.query_inputs(node['Gemm'])) == 3 and "\
    "self.attr_pick(node['Gemm'], 'transA', 0) == 0 and "\
    "self.attr_pick(node['Gemm'], 'transB', 0) == 1",
"src_ops_main_version": None,
"src_ops_minior_version": [5, -1]}
#Gemm:Gemm_142;Reshape:Reshape_141;Constant:Initializer_114;Constant_1:Initializer_115;Constant_2:Initializer_117
ruler_list.append(r_fullconnect_with_weight_at_in0_with_reshape)


r_tanh = {
"ruler_name": "r_tanh",
"src_ops_alias": ["Tanh"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Tanh:in0"]],
"src_out_tensor": ["Tanh:out0"],
"acu_lys_alias": ["tanh"],
"src_acu_in_tensor_map": [["I:out0", "tanh:in0"]],
"src_acu_out_tensor_map": [["Tanh:out0", "tanh:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": None}
ruler_list.append(r_tanh)

r_atan = {
"ruler_name": "r_atan",
"src_ops_alias": ["Atan"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Atan:in0"]],
"src_out_tensor": ["Atan:out0"],
"acu_lys_alias": ["atan"],
"src_acu_in_tensor_map": [["I:out0", "atan:in0"]],
"src_acu_out_tensor_map": [["Atan:out0", "atan:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": None}
ruler_list.append(r_atan)

r_atanh = {
"ruler_name": "r_atanh",
"src_ops_alias": ["Atanh"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Atanh:in0"]],
"src_out_tensor": ["Atanh:out0"],
"acu_lys_alias": ["atanh"],
"src_acu_in_tensor_map": [["I:out0", "atanh:in0"]],
"src_acu_out_tensor_map": [["Atanh:out0", "atanh:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": None}
ruler_list.append(r_atanh)

r_relu = {
"ruler_name": "r_relu",
"src_ops_alias": ["Relu"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Relu:in0"]],
"src_out_tensor": ["Relu:out0"],
"acu_lys_alias": ["relu"],
"src_acu_in_tensor_map": [["I:out0", "relu:in0"]],
"src_acu_out_tensor_map": [["Relu:out0", "relu:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": None}
ruler_list.append(r_relu)

r_elu = {
"ruler_name": "r_elu",
"src_ops_alias": ["Elu"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Elu:in0"]],
"src_out_tensor": ["Elu:out0"],
"acu_lys_alias": ["elu"],
"src_acu_in_tensor_map": [["I:out0", "elu:in0"]],
"src_acu_out_tensor_map": [["Elu:out0", "elu:out0"]],
"param_map": {"elu": {"alpha": ["FLOAT", "CODE", "self.attr_pick(node['Elu'], 'alpha', 1.0)"]}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [6, -1]}
ruler_list.append(r_elu)

r_celu = {
"ruler_name": "r_celu",
"src_ops_alias": ["Celu"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Celu:in0"]],
"src_out_tensor": ["Celu:out0"],
"acu_lys_alias": ["celu"],
"src_acu_in_tensor_map": [["I:out0", "celu:in0"]],
"src_acu_out_tensor_map": [["Celu:out0", "celu:out0"]],
"param_map": {"celu": {"alpha": ["FLOAT", "CODE", "self.attr_pick(node['Celu'], 'alpha', 1.0)"]}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [12, -1]}
ruler_list.append(r_celu)

r_inverse_sigmoid = {
"ruler_name": "r_inverse_sigmoid",
"src_ops_alias": ["Log", "Div", "Clip", "Clip_1", "Clip_2", "Constant", "Sub", "Constant_1", "Constant_2"],
"src_inter_flow": [["Div:out0", "Log:in0"], ["Clip:out0", "Div:in0"], ["Clip_1:out0", "Div:in1"],
                   ["Clip_2:out0", "Clip:in0"], ["Constant:out0", "Clip:in1"], ["Sub:out0", "Clip_1:in0"],
                   ["Constant_1:out0", "Clip_1:in1"], ["Constant_2:out0", "Sub:in0"], ["Clip_2:out0", "Sub:in1"]],
"src_in_anchor": [["I_0:out0", "Clip_2:in0"], ["I_1:out0", "Clip_2:in1"], ["I_2:out0", "Clip_2:in2"]],
"src_out_tensor": ["Log:out0"],
"acu_lys_alias": ["inverse_sigmoid"],
"src_acu_in_tensor_map": [["I_0:out0", "inverse_sigmoid:in0"]],
"src_acu_out_tensor_map": [["Log:out0", "inverse_sigmoid:out0"]],
"acu_inter_flow": [],
"param_map": {"inverse_sigmoid": {
    'eps': ["FLOAT", "CODE", "self.tensor_to_numpy(tensor['Constant:out0'])"]
}},
"blob_map": {"inverse_sigmoid": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_inverse_sigmoid)

r_inverse_sigmoid_1 = {
"ruler_name": "r_inverse_sigmoid_1",
"src_ops_alias": ["Log", "Div", "Clip", "Clip_1", "Clip_2", "Constant", "Sub", "Constant_1", "Constant_2"],
"src_inter_flow": [["Div:out0", "Log:in0"], ["Clip:out0", "Div:in0"], ["Clip_1:out0", "Div:in1"],
                   ["Clip_2:out0", "Clip:in0"], ["Constant:out0", "Clip:in1"], ["Sub:out0", "Clip_1:in0"],
                   ["Constant:out0", "Clip_1:in1"], ["Constant_1:out0", "Clip_2:in1"],
                   ["Constant_2:out0", "Clip_2:in2"], ["Constant_2:out0", "Sub:in0"], ["Clip_2:out0", "Sub:in1"]],
"src_in_anchor": [["I_0:out0", "Clip_2:in0"]],
"src_out_tensor": ["Log:out0"],
"acu_lys_alias": ["inverse_sigmoid"],
"src_acu_in_tensor_map": [["I_0:out0", "inverse_sigmoid:in0"]],
"src_acu_out_tensor_map": [["Log:out0", "inverse_sigmoid:out0"]],
"acu_inter_flow": [],
"param_map": {"inverse_sigmoid": {
    'eps': ["FLOAT", "CODE", "self.tensor_to_numpy(tensor['Constant:out0'])"]}
},
"blob_map": {"inverse_sigmoid": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_inverse_sigmoid_1)

r_sigmoid = {
"ruler_name": "r_sigmoid",
"src_ops_alias": ["Sigmoid"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Sigmoid:in0"]],
"src_out_tensor": ["Sigmoid:out0"],
"acu_lys_alias": ["Sigmoid"],
"src_acu_in_tensor_map": [["I:out0", "Sigmoid:in0"]],
"src_acu_out_tensor_map": [["Sigmoid:out0", "Sigmoid:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": None}
ruler_list.append(r_sigmoid)

r_hard_sigmoid = {
"ruler_name": "r_hard_sigmoid",
"src_ops_alias": ["HardSigmoid"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "HardSigmoid:in0"]],
"src_out_tensor": ["HardSigmoid:out0"],
"acu_lys_alias": ["hard_sigmoid"],
"src_acu_in_tensor_map": [["I:out0", "hard_sigmoid:in0"]],
"src_acu_out_tensor_map": [["HardSigmoid:out0", "hard_sigmoid:out0"]],
"param_map": {
    "hard_sigmoid":{
        "alpha": ["FLOAT", "CODE", "self.attr_pick(node['HardSigmoid'], 'alpha', 0.2)"],
        "beta": ["FLOAT", "CODE", "self.attr_pick(node['HardSigmoid'], 'beta', 0.5)"],
    }
},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": None}
ruler_list.append(r_hard_sigmoid)

r_hard_swish = {
"ruler_name": "r_hard_swish",
"src_ops_alias": ["HardSwish"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "HardSwish:in0"]],
"src_out_tensor": ["HardSwish:out0"],
"acu_lys_alias": ["hard_swish"],
"src_acu_in_tensor_map": [["I:out0", "hard_swish:in0"]],
"src_acu_out_tensor_map": [["HardSwish:out0", "hard_swish:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [14, -1]}
ruler_list.append(r_hard_swish)

r_hard_swish_1 = {
"ruler_name": "r_hard_swish_1",
"src_ops_alias": ["Div", "Mul", "Constant", "Clip", "Add", "Constant_1", "Constant_2"],
"src_inter_flow": [["Mul:out0", "Div:in0"], ["Constant:out0", "Div:in1"], ["Clip:out0", "Mul:in1"],
                   ["Add:out0", "Clip:in0"], ["Constant_1:out0", "Clip:in1"], ["Constant:out0", "Clip:in2"],
                   ["Constant_2:out0", "Add:in1"]],
"src_in_anchor": [["I_0:out0", "Mul:in0"], ["I_0:out0", "Add:in0"]],
"src_out_tensor": ["Div:out0"],
"acu_lys_alias": ["hard_swish"],
"src_acu_in_tensor_map": [["I_0:out0", "hard_swish:in0"]],
"src_acu_out_tensor_map": [["Div:out0", "hard_swish:out0"]],
"acu_inter_flow": [],
"param_map": {"hard_swish": {}},
"blob_map": {"hard_swish": {}},
"priority_tip": 0,
"pre_condition": "(self.tensor_to_numpy(tensor['Constant:out0']) == 6.0).all() and "\
                "(self.tensor_to_numpy(tensor['Constant_1:out0']) == 0.0).all() and "\
                "(self.tensor_to_numpy(tensor['Constant_2:out0']) == 3.0).all()",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_hard_swish_1)

r_silu = {
"ruler_name": "r_silu",
"src_ops_alias": ["SiLU"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "SiLU:in0"]],
"src_out_tensor": ["SiLU:out0"],
"acu_lys_alias": ["swish"],
"src_acu_in_tensor_map": [["I:out0", "swish:in0"]],
"src_acu_out_tensor_map": [["HardSwish:out0", "swish:out0"]],
"param_map": {"swish":{'beta':["INT", 'VALUE', 1]}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_silu)

r_leakrelu = {
"ruler_name": "r_leakrelu",
"src_ops_alias": ["LeakyRelu"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "LeakyRelu:in0"]],
"src_out_tensor": ["LeakyRelu:out0"],
"acu_lys_alias": ["leakyrelu"],
"src_acu_in_tensor_map": [["I:out0", "leakyrelu:in0"]],
"src_acu_out_tensor_map": [["LeakyRelu:out0", "leakyrelu:out0"]],
"param_map": {"leakyrelu": {"leaky_ratio": ["FLOAT", "CODE", "self.attr_pick(node['LeakyRelu'], 'alpha', 0.01)"]}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": None}
ruler_list.append(r_leakrelu)

r_qlinearleakrelu = {
"ruler_name": "r_leakrelu",
"src_ops_alias": ["QLinearLeakyRelu", "Constant", "Constant_1", "Constant_2", "Constant_3"],
"src_inter_flow": [["Constant:out0", "QLinearLeakyRelu:in1"],
                   ["Constant_1:out0", "QLinearLeakyRelu:in2"], ["Constant_2:out0", "QLinearLeakyRelu:in3"],
                   ["Constant_3:out0", "QLinearLeakyRelu:in4"]],
"src_in_anchor": [["I:out0", "QLinearLeakyRelu:in0"]],
"src_out_tensor": ["QLinearLeakyRelu:out0"],
"acu_lys_alias": ["leakyrelu"],
"src_acu_in_tensor_map": [["I:out0", "leakyrelu:in0"]],
"src_acu_out_tensor_map": [["QLinearLeakyRelu:out0", "leakyrelu:out0"]],
"param_map": {"leakyrelu": {"leaky_ratio": ["FLOAT", "CODE", \
                                            "self.attr_pick(node['QLinearLeakyRelu'], 'alpha', 0.01)"]}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": None}
ruler_list.append(r_qlinearleakrelu)

r_qlinearleakrelu = {
"ruler_name": "r_leakrelu",
"src_ops_alias": ["QLinearLeakyRelu", "Constant", "Constant_1", "Constant_2", "Constant_3"],
"src_inter_flow": [["Constant:out0", "QLinearLeakyRelu:in1"],
                   ["Constant_1:out0", "QLinearLeakyRelu:in2"], ["Constant_2:out0", "QLinearLeakyRelu:in3"],
                   ["Constant_3:out0", "QLinearLeakyRelu:in4"]],
"src_in_anchor": [["I:out0", "QLinearLeakyRelu:in0"]],
"src_out_tensor": ["QLinearLeakyRelu:out0"],
"acu_lys_alias": ["leakyrelu"],
"src_acu_in_tensor_map": [["I:out0", "leakyrelu:in0"]],
"src_acu_out_tensor_map": [["QLinearLeakyRelu:out0", "leakyrelu:out0"]],
"param_map": {"leakyrelu": {"leaky_ratio": ["FLOAT", "CODE", \
                                            "self.attr_pick(node['QLinearLeakyRelu'], 'alpha', 0.01)"]}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": None}
ruler_list.append(r_qlinearleakrelu)

r_prelu_with_reshape = {
"ruler_name": "r_prelu_with_reshape",
"src_ops_alias": ["PRelu", "Reshape", "Constant", "Constant_1"],
"src_inter_flow": [["Reshape:out0", "PRelu:in1"],
                   ["Constant:out0", "Reshape:in0"],
                   ["Constant_1:out0", "Reshape:in1"]],
"src_in_anchor": [["I_0:out0", "PRelu:in0"]],
"src_out_tensor": ["PRelu:out0"],
"acu_lys_alias": ["prelu"],
"src_acu_in_tensor_map": [["I_0:out0", "prelu:in0"]],
"src_acu_out_tensor_map": [["PRelu:out0", "prelu:out0"]],
"acu_inter_flow": [],
"param_map": {"prelu": {}},
"blob_map": {"prelu": {"a":
              ["CODE",
               "self.prelu_alpha(tensor['I_0:out0'], tensor['Reshape:out0'])"]
          }},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": None}
ruler_list.append(r_prelu_with_reshape)

r_prelu = {
"ruler_name": "r_prelu",
"src_ops_alias": ["PRelu", "Constant_0"],
"src_inter_flow": [ ["Constant_0:out0", "PRelu:in1"]],
"src_in_anchor": [["I:out0", "PRelu:in0"]],
"src_out_tensor": ["PRelu:out0"],
"acu_lys_alias": ["prelu"],
"src_acu_in_tensor_map": [["I:out0", "prelu:in0"]],
"src_acu_out_tensor_map": [["PRelu:out0", "prelu:out0"]],
"param_map": {},
"blob_map": {"prelu": {"a": ["CODE", "self.prelu_alpha(tensor['I:out0'], tensor['Constant_0:out0'])"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": None}
ruler_list.append(r_prelu)

r_prelu_unsqueeze = {
"ruler_name": "r_prelu_unsqueeze",
"src_ops_alias": ["PRelu", "Constant", "Unsqueeze"],
"src_inter_flow": [["Constant:out0","Unsqueeze:in0"], ["Unsqueeze:out0", "PRelu:in1"]],
"src_in_anchor": [["I:out0", "PRelu:in0"]],
"src_out_tensor": ["PRelu:out0"],
"acu_lys_alias": ["prelu"],
"src_acu_in_tensor_map": [["I:out0", "prelu:in0"]],
"src_acu_out_tensor_map": [["PRelu:out0", "prelu:out0"]],
"param_map":  {},
"blob_map": {"prelu": {"a": ["CODE", "self.prelu_alpha(tensor['I:out0'], tensor['Unsqueeze:out0'])"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": None}
ruler_list.append(r_prelu_unsqueeze)

r_reciprocal = {
"ruler_name": "r_reciprocal",
"src_ops_alias": ["Reciprocal"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "Reciprocal:in0"]],
"src_out_tensor": ["Reciprocal:out0"],
"acu_lys_alias": ["variable", "Divide"],
"src_acu_in_tensor_map": [["I_0:out0", "Divide:in1"]],
"src_acu_out_tensor_map": [["Reciprocal:out0", "Divide:out0"]],
"acu_inter_flow": [["variable:out0", "Divide:in0"]],
"param_map": {"variable": {}},
"blob_map": {"variable": {'data': ['CODE', "np.array([1], dtype=np.float32)"]}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_reciprocal)

r_pow = {
"ruler_name": "r_pow",
"src_ops_alias": ["Pow"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Pow:in0"], ["I_1:out0", "Pow:in1"]],
"src_out_tensor": ["Pow:out0"],
"acu_lys_alias": ["pow"],
"src_acu_in_tensor_map": [["I:out0", "pow:in0"], ["I_1:out0", "pow:in1"]],
"src_acu_out_tensor_map": [["Pow:out0", "pow:out0"]],
"param_map": {"pow": {}},
"blob_map": {"pow": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": None
}
ruler_list.append(r_pow)

r_equal = {
"ruler_name": "r_equal",
"src_ops_alias": ["Equal"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Equal:in0"], ["I_1:out0", "Equal:in1"]],
"src_out_tensor": ["Equal:out0"],
"acu_lys_alias": ["equal"],
"src_acu_in_tensor_map": [["I:out0", "equal:in0"], ["I_1:out0", "equal:in1"]],
"src_acu_out_tensor_map": [["Equal:out0", "equal:out0"]],
"param_map": {"equal": {}},
"blob_map": {"equal": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": None
}
ruler_list.append(r_equal)

r_less = {
"ruler_name": "r_less",
"src_ops_alias": ["Less"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Less:in0"], ["I_1:out0", "Less:in1"]],
"src_out_tensor": ["Less:out0"],
"acu_lys_alias": ["less"],
"src_acu_in_tensor_map": [["I:out0", "less:in0"], ["I_1:out0", "less:in1"]],
"src_acu_out_tensor_map": [["Less:out0", "less:out0"]],
"param_map": {"less": {}},
"blob_map": {"less": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": None
}
ruler_list.append(r_less)

r_less_equal = {
"ruler_name": "r_less_equal",
"src_ops_alias": ["LessOrEqual"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "LessOrEqual:in0"], ["I_1:out0", "LessOrEqual:in1"]],
"src_out_tensor": ["LessOrEqual:out0"],
"acu_lys_alias": ["less_equal"],
"src_acu_in_tensor_map": [["I:out0", "less_equal:in0"], ["I_1:out0", "less_equal:in1"]],
"src_acu_out_tensor_map": [["LessOrEqual:out0", "less_equal:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [12, -1]
}
ruler_list.append(r_less_equal)

r_conv1d = {
"ruler_name": "r_conv1d",
"src_ops_alias": ["Conv", "Constant_0", "Constant_1"],
"src_inter_flow": [["Constant_0:out0", "Conv:in1"], ["Constant_1:out0", "Conv:in2"]],
"src_in_anchor": [["I:out0", "Conv:in0"]],
"src_out_tensor": ["Conv:out0"],
"acu_lys_alias": ["conv1d"],
"src_acu_in_tensor_map": [["I:out0", "conv1d:in0"]],
"src_acu_out_tensor_map": [["Conv:out0", "conv1d:out0"]],
"param_map":
{
"conv1d":
{
"weights": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[0]"],
"bias": ["BOOL", "VALUE", True],
"pad_method": ["STRING", "CODE", "'auto' if self.attr_pick(node['Conv'], 'pads', None)"\
               "== None else 'padding_const'"],
"ksize": ["INT", "PYFUNC", r_conv1d_get_kernel_shape(kernel_name='Constant_0:out0')],
"group_number": ["INT", "CODE", "self.attr_pick(node['Conv'], 'group', 1)"],
"stride": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1])[0]"],
"dilation":
[
  "INT",
  "CODE",
  "self.attr_pick(node['Conv'], 'dilations')"\
  " if isinstance(self.attr_pick(node['Conv'], 'dilations'), int)"\
  " else self.attr_pick(node['Conv'], 'dilations')[0]"
],
"padding":
[
  "STRING",
  "CODE",
  "'SAME' if self.attr_pick(node['Conv'], 'auto_pad', 'NOTSET') "\
  "in ['SAME_UPPER', 'SAME_LOWER'] else 'VALID' "
],
"pad":
[
  "INTS",
  "CODE",
  "[p for p in self.array_layout(self.attr_pick(node['Conv'], 'pads', [0, 0]), [0, 1])]"
]
}
},
"blob_map":
{
  "conv1d":
  {
    "weight": ["CODE", "self.tensor_to_numpy(tensor['Constant_0:out0'])"],
    "bias": ["CODE", "self.tensor_to_numpy(tensor['Constant_1:out0'])"]
  }
},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "len(self.shape_pick(tensor['Constant_0:out0'])) == 3",
"src_ops_main_version": None,
"src_ops_minior_version": None
}
ruler_list.append(r_conv1d)

r_group_conv1d = {
"ruler_name": "r_group_conv1d",
"src_ops_alias": ["Conv", "Constant_0", "Constant_1"],
"src_inter_flow": [["Constant_0:out0", "Conv:in1"], ["Constant_1:out0", "Conv:in2"]],
"src_in_anchor": [["I:out0", "Conv:in0"]],
"src_out_tensor": ["Conv:out0"],
"acu_lys_alias": ["group_conv1d"],
"src_acu_in_tensor_map": [["I:out0", "group_conv1d:in0"]],
"src_acu_out_tensor_map": [["Conv:out0", "group_conv1d:out0"]],
"param_map":
{
"group_conv1d":
{
"weights": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[0]"],
"bias": ["BOOL", "VALUE", True],
"pad_method": ["STRING", "CODE", "'auto' if self.attr_pick(node['Conv'], 'pads', None)"\
               "== None else 'padding_const'"],
"ksize": ["INT", "CODE", "self.attr_pick(node['Conv'], 'kernel_shape')[0]"],
"group_number": ["INT", "CODE", "self.attr_pick(node['Conv'], 'group', 1)"],
"stride": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1])[0]"],
"dilation":
[
  "INT",
  "CODE",
  "self.attr_pick(node['Conv'], 'dilations')"\
  " if isinstance(self.attr_pick(node['Conv'], 'dilations'), int)"\
  " else self.attr_pick(node['Conv'], 'dilations')[0]"
],
"padding":
[
  "STRING",
  "CODE",
  "'SAME' if self.attr_pick(node['Conv'], 'auto_pad', 'NOTSET') "\
  "in ['SAME_UPPER', 'SAME_LOWER'] else 'VALID' "
],
"pad":
[
  "INTS",
  "CODE",
  "[p for p in self.array_layout(self.attr_pick(node['Conv'], 'pads', [0, 0]), [0, 1])]"
]
}
},
"blob_map":
{
  "group_conv1d":
  {
    "weight": ["CODE", "self.tensor_to_numpy(tensor['Constant_0:out0'])"],
    "bias": ["CODE", "self.tensor_to_numpy(tensor['Constant_1:out0'])"]
  }
},
"acu_inter_flow": [],
"priority_tip": 1,
"pre_condition": r_group_conv1d_pre_condition(),
"src_ops_main_version": None,
"src_ops_minior_version": None
}
ruler_list.append(r_group_conv1d)

r_depthwise_conv1d = {
"ruler_name": "r_depthwise_conv1d",
"src_ops_alias": ["Conv", "Constant_0", "Constant_1"],
"src_inter_flow": [["Constant_0:out0", "Conv:in1"], ["Constant_1:out0", "Conv:in2"]],
"src_in_anchor": [["I:out0", "Conv:in0"]],
"src_out_tensor": ["Conv:out0"],
"acu_lys_alias": ["depthwise_conv1d"],
"src_acu_in_tensor_map": [["I:out0", "depthwise_conv1d:in0"]],
"src_acu_out_tensor_map": [["Conv:out0", "depthwise_conv1d:out0"]],
"param_map":
{
"depthwise_conv1d":
{
"weights": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[0]"],
"bias": ["BOOL", "VALUE", True],
"pad_method": ["STRING", "CODE", "'auto' if self.attr_pick(node['Conv'], 'pads', None)"\
               "== None else 'padding_const'"],
"ksize": ["INT", "CODE", "self.attr_pick(node['Conv'], 'kernel_shape')[0]"],
"group_number": ["INT", "CODE", "self.attr_pick(node['Conv'], 'group', 1)"],
"multiplier": ["INT", "CODE",
               "int(self.shape_pick(tensor['Constant_0:out0'])[0]/self.shape_pick(tensor['I:out0'])[1])"],
"stride": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1])[0]"],
"dilation":
[
  "INT",
  "CODE",
  "self.attr_pick(node['Conv'], 'dilations')"\
  " if isinstance(self.attr_pick(node['Conv'], 'dilations'), int)"\
  " else self.attr_pick(node['Conv'], 'dilations')[0]"
],
"padding":
[
  "STRING",
  "CODE",
  "'SAME' if self.attr_pick(node['Conv'], 'auto_pad', 'NOTSET') "\
  "in ['SAME_UPPER', 'SAME_LOWER'] else 'VALID' "
],
"pad":
[
  "INTS",
  "CODE",
  "[p for p in self.array_layout(self.attr_pick(node['Conv'], 'pads', [0, 0]), [0, 1])]"
]
}
},
"blob_map":
{
  "depthwise_conv1d":
  {
    "weight": ["CODE", "self.tensor_to_numpy(tensor['Constant_0:out0'])"],
    "bias": ["CODE", "self.tensor_to_numpy(tensor['Constant_1:out0'])"]
  }
},
"acu_inter_flow": [],
"priority_tip": 2,
"pre_condition": r_depthwise_conv1d_pre_condition(),
"src_ops_main_version": None,
"src_ops_minior_version": None
}
ruler_list.append(r_depthwise_conv1d)

r_conv1d_no_bias = {
"ruler_name": "r_conv1d_no_bias",
"src_ops_alias": ["Conv", "Constant_0"],
"src_inter_flow": [["Constant_0:out0", "Conv:in1"]],
"src_in_anchor": [["I:out0", "Conv:in0"]],
"src_out_tensor": ["Conv:out0"],
"acu_lys_alias": ["conv1d"],
"src_acu_in_tensor_map": [["I:out0", "conv1d:in0"]],
"src_acu_out_tensor_map": [["Conv:out0", "conv1d:out0"]],
"param_map":
{
"conv1d":
{
"weights": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[0]"],
"bias": ["BOOL", "VALUE", False],
"pad_method": ["STRING", "CODE",
"'auto' if self.attr_pick(node['Conv'], 'pads', None) == None else 'padding_const'"],
"ksize": ["INT", "PYFUNC", r_conv1d_get_kernel_shape(kernel_name='Constant_0:out0')],
"group_number": ["INT", "CODE", "self.attr_pick(node['Conv'], 'group', 1)"],
"stride": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1])[0]"],
"dilation":
[
  "INT",
  "CODE",
  "self.attr_pick(node['Conv'], 'dilations')"\
  " if isinstance(self.attr_pick(node['Conv'], 'dilations'), int)"\
  " else self.attr_pick(node['Conv'], 'dilations')[0]"
],
"padding":
[
  "STRING",
  "CODE",
  "'SAME' if self.attr_pick(node['Conv'], 'auto_pad', 'NOTSET') "\
  "in ['SAME_UPPER', 'SAME_LOWER'] else 'VALID' "
],
"pad":
[
  "INTS",
  "CODE",
  "[p for p in self.array_layout(self.attr_pick(node['Conv'], 'pads', [0, 0]), [0, 1])]"
]
}
},
"blob_map":
{
  "conv1d":
  {
    "weight": ["CODE", "self.tensor_to_numpy(tensor['Constant_0:out0'])"]
  }
},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "len(self.shape_pick(tensor['Constant_0:out0'])) == 3",
"src_ops_main_version": None,
"src_ops_minior_version": None
}
ruler_list.append(r_conv1d_no_bias)

r_group_conv1d_no_bias = {
"ruler_name": "r_group_conv1d_no_bias",
"src_ops_alias": ["Conv", "Constant_0"],
"src_inter_flow": [["Constant_0:out0", "Conv:in1"]],
"src_in_anchor": [["I:out0", "Conv:in0"]],
"src_out_tensor": ["Conv:out0"],
"acu_lys_alias": ["group_conv1d"],
"src_acu_in_tensor_map": [["I:out0", "group_conv1d:in0"]],
"src_acu_out_tensor_map": [["Conv:out0", "group_conv1d:out0"]],
"param_map":
{
"group_conv1d":
{
"weights": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[0]"],
"bias": ["BOOL", "VALUE", False],
"pad_method": ["STRING", "CODE",
"'auto' if self.attr_pick(node['Conv'], 'pads', None) == None else 'padding_const'"],
"ksize": ["INT", "CODE", "self.attr_pick(node['Conv'], 'kernel_shape')[0]"],
"group_number": ["INT", "CODE", "self.attr_pick(node['Conv'], 'group', 1)"],
"stride": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1])[0]"],
"dilation":
[
  "INT",
  "CODE",
  "self.attr_pick(node['Conv'], 'dilations')"\
  " if isinstance(self.attr_pick(node['Conv'], 'dilations'), int)"\
  " else self.attr_pick(node['Conv'], 'dilations')[0]"
],
"padding":
[
  "STRING",
  "CODE",
  "'SAME' if self.attr_pick(node['Conv'], 'auto_pad', 'NOTSET') "\
  "in ['SAME_UPPER', 'SAME_LOWER'] else 'VALID' "
],
"pad":
[
  "INTS",
  "CODE",
  "[p for p in self.array_layout(self.attr_pick(node['Conv'], 'pads', [0, 0]), [0, 1])]"
]
}
},
"blob_map":
{
  "group_conv1d":
  {
    "weight": ["CODE", "self.tensor_to_numpy(tensor['Constant_0:out0'])"]
  }
},
"acu_inter_flow": [],
"priority_tip": 1,
"pre_condition": r_group_conv1d_pre_condition(),
"src_ops_main_version": None,
"src_ops_minior_version": None
}
ruler_list.append(r_group_conv1d_no_bias)

r_depthwise_conv1d_no_bias = {
"ruler_name": "r_depthwise_conv1d_no_bias",
"src_ops_alias": ["Conv", "Constant_0"],
"src_inter_flow": [["Constant_0:out0", "Conv:in1"]],
"src_in_anchor": [["I:out0", "Conv:in0"]],
"src_out_tensor": ["Conv:out0"],
"acu_lys_alias": ["depthwise_conv1d"],
"src_acu_in_tensor_map": [["I:out0", "depthwise_conv1d:in0"]],
"src_acu_out_tensor_map": [["Conv:out0", "depthwise_conv1d:out0"]],
"param_map":
{
"depthwise_conv1d":
{
"weights": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[0]"],
"bias": ["BOOL", "VALUE", False],
"pad_method": ["STRING", "CODE",
"'auto' if self.attr_pick(node['Conv'], 'pads', None) == None else 'padding_const'"],
"ksize": ["INT", "CODE", "self.attr_pick(node['Conv'], 'kernel_shape')[0]"],
"group_number": ["INT", "CODE", "self.attr_pick(node['Conv'], 'group', 1)"],
"multiplier": ["INT", "CODE",
               "int(self.shape_pick(tensor['Constant_0:out0'])[0]/self.shape_pick(tensor['I:out0'])[1])"],
"stride": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1])[0]"],
"dilation":
[
  "INT",
  "CODE",
  "self.attr_pick(node['Conv'], 'dilations')"\
  " if isinstance(self.attr_pick(node['Conv'], 'dilations'), int)"\
  " else self.attr_pick(node['Conv'], 'dilations')[0]"
],
"padding":
[
  "STRING",
  "CODE",
  "'SAME' if self.attr_pick(node['Conv'], 'auto_pad', 'NOTSET') "\
  "in ['SAME_UPPER', 'SAME_LOWER'] else 'VALID' "
],
"pad":
[
  "INTS",
  "CODE",
  "[p for p in self.array_layout(self.attr_pick(node['Conv'], 'pads', [0, 0]), [0, 1])]"
]
}
},
"blob_map":
{
  "depthwise_conv1d":
  {
    "weight": ["CODE", "self.tensor_to_numpy(tensor['Constant_0:out0'])"]
  }
},
"acu_inter_flow": [],
"priority_tip": 2,
"pre_condition": r_depthwise_conv1d_pre_condition(),
"src_ops_main_version": None,
"src_ops_minior_version": None
}
ruler_list.append(r_depthwise_conv1d_no_bias)

r_conv = {
"ruler_name": "r_conv",
"src_ops_alias": ["Conv", "Constant_0", "Constant_1"],
"src_inter_flow": [["Constant_0:out0", "Conv:in1"], ["Constant_1:out0", "Conv:in2"]],
"src_in_anchor": [["I:out0", "Conv:in0"]],
"src_out_tensor": ["Conv:out0"],
"acu_lys_alias": ["convolution"],
"src_acu_in_tensor_map": [["I:out0", "convolution:in0"]],
"src_acu_out_tensor_map": [["Conv:out0", "convolution:out0"]],
"param_map":
{"convolution":
{"weights": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[0]"],
"pad_method":
   ["STRING", "CODE", "'auto' if self.attr_pick(node['Conv'], 'pads', None) == None else 'padding_const'"],
"bias": ["BOOL", "VALUE", True],
"ksize_w": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[3]"],
"group_number": ["INT", "CODE", "self.attr_pick(node['Conv'], 'group', 1)"],
"ksize_h": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[2]"],
"stride_w": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1, 1])[1]"],
"stride_h": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1, 1])[0]"],
"dilation": ["INTS", "CODE", "self.conv_dilation(node['Conv'])"],
"padding":
   ["STRING",
    "CODE",
    "'SAME' if self.attr_pick(node['Conv'], 'auto_pad', 'NOTSET') in ['SAME_UPPER', 'SAME_LOWER'] else 'VALID' "],
"pad":
   ["INTS",
    "CODE",
    "[p for p in self.array_layout(self.attr_pick(node['Conv'], 'pads', [ 0, 0, 0, 0]), [0, 2, 1, 3])]"]
}
},
"blob_map": {"convolution":
                 {"weight": ["CODE", "self.tensor_to_numpy(tensor['Constant_0:out0'])"],
                  "bias": ["CODE", "self.tensor_to_numpy(tensor['Constant_1:out0'])"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "len(self.shape_pick(tensor['Constant_0:out0'])) == 4",
"src_ops_main_version": None,
"src_ops_minior_version": None}
ruler_list.append(r_conv)

r_conv2d_op = {
"ruler_name": "r_conv2d_op",
"src_ops_alias": ["Conv"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "Conv:in0"], ["I_1:out0", "Conv:in1"]],
"src_out_tensor": ["Conv:out0"],
"acu_lys_alias": ["conv2d_op"],
"src_acu_in_tensor_map": [["I_0:out0", "conv2d_op:in0"], ["I_1:out0", "conv2d_op:in1"]],
"src_acu_out_tensor_map": [["Conv:out0", "conv2d_op:out0"]],
"acu_inter_flow": [],
"param_map": {
    "conv2d_op": {
    "padding":
        ["STRING",
         "CODE",
         "'SAME' if self.attr_pick(node['Conv'], 'auto_pad', 'NOTSET') in ['SAME_UPPER', 'SAME_LOWER'] else 'VALID' "],
    "pad":
        ["INTS",
         "CODE",
         "[p for p in self.array_layout(self.attr_pick(node['Conv'], 'pads', [ 0, 0, 0, 0]), [0, 2, 1, 3])]"],
    "group_number": ["INT", "CODE", "self.attr_pick(node['Conv'], 'group', 1)"],
    "stride_w": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1, 1])[1]"],
    "stride_h": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1, 1])[0]"],
    "dilation":["INTS", "CODE", "self.conv_dilation(node['Conv'])"],
}},
"blob_map": {"conv2d_op": {}},
"priority_tip": 0,
"pre_condition": "len(self.shape_pick(tensor['I_1:out0'])) == 4",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_conv2d_op)

r_conv2d_bias_op = {
"ruler_name": "r_conv2d_op",
"src_ops_alias": ["Conv"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "Conv:in0"], ["I_1:out0", "Conv:in1"], ["I_2:out0", "Conv:in2"]],
"src_out_tensor": ["Conv:out0"],
"acu_lys_alias": ["conv2d_op"],
"src_acu_in_tensor_map": [["I_0:out0", "conv2d_op:in0"], ["I_1:out0", "conv2d_op:in1"], ["I_2:out0", "conv2d_op:in2"]],
"src_acu_out_tensor_map": [["Conv:out0", "conv2d_op:out0"]],
"acu_inter_flow": [],
"param_map": {
    "conv2d_op": {
    "padding":
        ["STRING",
         "CODE",
         "'SAME' if self.attr_pick(node['Conv'], 'auto_pad', 'NOTSET') in ['SAME_UPPER', 'SAME_LOWER'] else 'VALID' "],
    "pad":
        ["INTS",
         "CODE",
         "[p for p in self.array_layout(self.attr_pick(node['Conv'], 'pads', [ 0, 0, 0, 0]), [0, 2, 1, 3])]"],
    "group_number": ["INT", "CODE", "self.attr_pick(node['Conv'], 'group', 1)"],
    "stride_w": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1, 1])[1]"],
    "stride_h": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1, 1])[0]"],
    "dilation": ["INTS", "CODE", "self.conv_dilation(node['Conv'])"],
}},
"blob_map": {"conv2d_op": {}},
"priority_tip": 0,
"pre_condition": "len(self.shape_pick(tensor['I_1:out0'])) == 4",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_conv2d_bias_op)

r_depthwise_conv2d_op = {
"ruler_name": "r_depthwise_conv2d_op",
"src_ops_alias": ["Conv"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "Conv:in0"], ["I_1:out0", "Conv:in1"]],
"src_out_tensor": ["Conv:out0"],
"acu_lys_alias": ["depthwise_conv2d_op"],
"src_acu_in_tensor_map": [["I_0:out0", "depthwise_conv2d_op:in0"], ["I_1:out0", "depthwise_conv2d_op:in1"]],
"src_acu_out_tensor_map": [["Conv:out0", "depthwise_conv2d_op:out0"]],
"acu_inter_flow": [],
"param_map": {
    "depthwise_conv2d_op": {
    "padding":
        ["STRING",
         "CODE",
         "'SAME' if self.attr_pick(node['Conv'], 'auto_pad', 'NOTSET') in ['SAME_UPPER', 'SAME_LOWER'] else 'VALID' "],
    "pad":
        ["INTS",
         "CODE",
         "[p for p in self.array_layout(self.attr_pick(node['Conv'], 'pads', [ 0, 0, 0, 0]), [0, 2, 1, 3])]"],
    "pad_method":
        ["STRING", "CODE", "'auto' if self.attr_pick(node['Conv'], 'pads', None) == None else 'padding_const'"],
    "ksize_w": ["INT", "CODE", "self.attr_pick(node['Conv'], 'kernel_shape')[1]"],
    "group_number": ["INT", "CODE", "self.attr_pick(node['Conv'], 'group', 1)"],
    "ksize_h": ["INT", "CODE", "self.attr_pick(node['Conv'], 'kernel_shape')[0]"],
    "stride_w": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1, 1])[1]"],
    "stride_h": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1, 1])[0]"],
    "dilation": ["INTS", "CODE", "self.conv_dilation(node['Conv'])"],
}},
"blob_map": {"depthwise_conv2d_op": {}},
"priority_tip": 1,
"pre_condition": "self.attr_pick(node['Conv'], 'group', 1) == self.shape_pick(tensor['I_0:out0'])[1]",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_depthwise_conv2d_op)

r_conv_nchw_squeeze = {
"ruler_name": "r_conv_nchw_squeeze",
"src_ops_alias": ["Conv", "Constant", "Squeeze", "Constant_1"],
"src_inter_flow": [["Constant:out0", "Conv:in1"], ["Squeeze:out0", "Conv:in2"], ["Constant_1:out0", "Squeeze:in0"]],
"src_in_anchor": [["I_0:out0", "Conv:in0"]],
"src_out_tensor": ["Conv:out0"],
"acu_lys_alias": ["convolution"],
"src_acu_in_tensor_map": [["I_0:out0", "convolution:in0"]],
"src_acu_out_tensor_map": [["Conv:out0", "convolution:out0"]],
"acu_inter_flow": [],
"param_map":
{"convolution":
{"weights": ["INT", "CODE", "self.shape_pick(tensor['Squeeze:out0'])[0]"],
"pad_method":
   ["STRING", "CODE", "'auto' if self.attr_pick(node['Conv'], 'pads', None) == None else 'padding_const'"],
"bias": ["BOOL", "VALUE", True],
"ksize_w": ["INT", "CODE", "self.attr_pick(node['Conv'], 'kernel_shape')[1]"],
"group_number": ["INT", "CODE", "self.attr_pick(node['Conv'], 'group', 1)"],
"ksize_h": ["INT", "CODE", "self.attr_pick(node['Conv'], 'kernel_shape')[0]"],
"stride_w": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1, 1])[1]"],
"stride_h": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1, 1])[0]"],
"dilation":
     ['INT',
      'CODE',
      "self.attr_pick(node['Conv'], 'dilations')"\
      " if isinstance(self.attr_pick(node['Conv'], 'dilations'), int)"\
      " else self.attr_pick(node['Conv'], 'dilations')[0]"],
"padding":
   ["STRING",
    "CODE",
    "'SAME' if self.attr_pick(node['Conv'], 'auto_pad', 'NOTSET') in ['SAME_UPPER', 'SAME_LOWER'] else 'VALID' "],
"pad":
   ["INTS",
    "CODE",
    "[p for p in self.array_layout(self.attr_pick(node['Conv'], 'pads', [ 0, 0, 0, 0]), [0, 2, 1, 3])]"]
}
},
"blob_map": {"convolution":
                 {"weight": ["CODE", "self.tensor_to_numpy(tensor['Constant:out0'])"],
                  "bias": ["CODE", "self.tensor_to_numpy(tensor['Squeeze:out0'])"]}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [6, -1]}
ruler_list.append(r_conv_nchw_squeeze)

r_conv_no_bias = {
"ruler_name": "r_conv_no_bias",
"src_ops_alias": ["Conv", "Constant_0"],
"src_inter_flow": [["Constant_0:out0", "Conv:in1"]],
"src_in_anchor": [["I:out0", "Conv:in0"]],
"src_out_tensor": ["Conv:out0"],
"acu_lys_alias": ["convolution"],
"src_acu_in_tensor_map": [["I:out0", "convolution:in0"]],
"src_acu_out_tensor_map": [["Conv:out0", "convolution:out0"]],
"param_map":
{"convolution":
{"weights": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[0]"],
"pad_method":
   ["STRING", "CODE", "'auto' if self.attr_pick(node['Conv'], 'pads', None) == None else 'padding_const'"],
"bias": ["BOOL", "VALUE", False],
"ksize_w": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[3]"],
"group_number": ["INT", "CODE", "self.attr_pick(node['Conv'], 'group', 1)"],
"ksize_h": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[2]"],
"stride_w": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1, 1])[1]"],
"stride_h": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1, 1])[0]"],
"dilation":
     ['INT',
      'CODE',
      "self.attr_pick(node['Conv'], 'dilations')"\
      " if isinstance(self.attr_pick(node['Conv'], 'dilations'), int)"\
      " else self.attr_pick(node['Conv'], 'dilations')[0]"],
"padding":
   ["STRING",
    "CODE",
    "'SAME' if self.attr_pick(node['Conv'], 'auto_pad', 'NOTSET') in ['SAME_UPPER', 'SAME_LOWER'] else 'VALID' "],
"pad":
   ["INTS",
    "CODE",
    "[p for p in self.array_layout(self.attr_pick(node['Conv'], 'pads', [ 0, 0, 0, 0]), [0, 2, 1, 3])]"]
}
},
"blob_map": {"convolution":
                 {"weight": ["CODE", "self.tensor_to_numpy(tensor['Constant_0:out0'])"],}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": None}
ruler_list.append(r_conv_no_bias)

r_conv3d_no_bias = {
"ruler_name": "r_conv3d_no_bias",
"src_ops_alias": ["Conv", "Constant_0"],
"src_inter_flow": [["Constant_0:out0", "Conv:in1"]],
"src_in_anchor": [["I:out0", "Conv:in0"]],
"src_out_tensor": ["Conv:out0"],
"acu_lys_alias": ["conv3d"],
"src_acu_in_tensor_map": [["I:out0", "conv3d:in0"]],
"src_acu_out_tensor_map": [["Conv:out0", "conv3d:out0"]],
"param_map":
{"conv3d":
{"weights": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[0]"],
"pad_method":
   ["STRING", "CODE", "'auto' if self.attr_pick(node['Conv'], 'pads', None) == None else 'padding_const'"],
"bias": ["BOOL", "VALUE", False],
"ksize_w": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[4]"],
"group_number": ["INT", "CODE", "self.attr_pick(node['Conv'], 'group', 1)"],
"ksize_h": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[3]"],
"ksize_d": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[2]"],
"stride_w": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1, 1, 1])[2]"],
"stride_h": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1, 1, 1])[1]"],
"stride_d": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1, 1, 1])[0]"],
"dilation":
     ['INT',
      'CODE',
      "self.attr_pick(node['Conv'], 'dilations')"\
      " if isinstance(self.attr_pick(node['Conv'], 'dilations'), int)"\
      " else self.attr_pick(node['Conv'], 'dilations')[0]"],
"padding":
   ["STRING",
    "CODE",
    "'SAME' if self.attr_pick(node['Conv'], 'auto_pad', 'NOTSET') in ['SAME_UPPER', 'SAME_LOWER'] else 'VALID' "],
"pad":
   ["INTS",
    "CODE",
    "[p for p in self.array_layout(self.attr_pick(node['Conv'], 'pads', [ 0, 0, 0, 0, 0, 0]), [0, 3, 1, 4, 2, 5])]"]
}
},
"blob_map": {"conv3d":
                 {"weight": ["CODE", "self.tensor_to_numpy(tensor['Constant_0:out0'])"],}},
"acu_inter_flow": [["Constant_0:out0", "conv3d:in1"]],
"priority_tip": 1,
"pre_condition": "len(self.shape_pick(tensor['Constant_0:out0'])) == 5",
"src_ops_main_version": None,
"src_ops_minior_version": None}
ruler_list.append(r_conv3d_no_bias)

r_conv3d_with_bias = {
"ruler_name": "r_conv3d_with_bias",
"src_ops_alias": ["Conv", "Constant_0", "Constant_1"],
"src_inter_flow": [["Constant_0:out0", "Conv:in1"], ["Constant_1:out0", "Conv:in2"]],
"src_in_anchor": [["I:out0", "Conv:in0"]],
"src_out_tensor": ["Conv:out0"],
"acu_lys_alias": ["conv3d"],
"src_acu_in_tensor_map": [["I:out0", "conv3d:in0"]],
"src_acu_out_tensor_map": [["Conv:out0", "conv3d:out0"]],
"param_map":
{"conv3d":
{"weights": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[0]"],
"pad_method":
   ["STRING", "CODE", "'auto' if self.attr_pick(node['Conv'], 'pads', None) == None else 'padding_const'"],
"bias": ["BOOL", "VALUE", True],
"ksize_w": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[4]"],
"group_number": ["INT", "CODE", "self.attr_pick(node['Conv'], 'group', 1)"],
"ksize_h": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[3]"],
"ksize_d": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[2]"],
"stride_w": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1, 1, 1])[2]"],
"stride_h": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1, 1, 1])[1]"],
"stride_d": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1, 1, 1])[0]"],
"dilation":
     ['INT',
      'CODE',
      "self.attr_pick(node['Conv'], 'dilations')"\
      " if isinstance(self.attr_pick(node['Conv'], 'dilations'), int)"\
      " else self.attr_pick(node['Conv'], 'dilations')[0]"],
"padding":
   ["STRING",
    "CODE",
    "'SAME' if self.attr_pick(node['Conv'], 'auto_pad', 'NOTSET') in ['SAME_UPPER', 'SAME_LOWER'] else 'VALID' "],
"pad":
   ["INTS",
    "CODE",
    "[p for p in self.array_layout(self.attr_pick(node['Conv'], 'pads', [ 0, 0, 0, 0, 0, 0]), [0, 3, 1, 4, 2, 5])]"]
}
},
"blob_map": {"conv3d":
                 {"weight": ["CODE", "self.tensor_to_numpy(tensor['Constant_0:out0'])"],
                  "bias": ["CODE", "self.tensor_to_numpy(tensor['Constant_1:out0'])"]}},
"acu_inter_flow": [["Constant_0:out0", "conv3d:in1"], ["Constant_1:out0", "conv3d:in2"]],
"priority_tip": 1,
"pre_condition": "len(self.shape_pick(tensor['Constant_0:out0'])) == 5",
"src_ops_main_version": None,
"src_ops_minior_version": None}
ruler_list.append(r_conv3d_with_bias)

r_conv_add = {
"ruler_name": "r_conv_add",
"src_ops_alias": ["Conv", "Add", "Constant_0", "Constant_1"],
"src_inter_flow": [["Conv:out0", "Add:in0"], ["Constant_0:out0", "Conv:in1"], ["Constant_1:out0", "Add:in1"]],
"src_in_anchor": [["I:out0", "Conv:in0"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["convolution"],
"src_acu_in_tensor_map": [["I:out0", "convolution:in0"]],
"src_acu_out_tensor_map": [["Add:out0", "convolution:out0"]],
"param_map":
    {"convolution":
         {"weights": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[0]"],
          "pad_method":
              ["STRING", "CODE", "'auto' if self.attr_pick(node['Conv'], 'pads', None) == None else 'padding_const'"],
          "bias": ["BOOL", "VALUE", True],
          "ksize_w": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[3]"],
          "group_number": ["INT", "CODE", "self.attr_pick(node['Conv'], 'group', 1)"],
          "ksize_h": ["INT", "CODE", "self.shape_pick(tensor['Constant_0:out0'])[2]"],
          "stride_w": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1, 1])[1]"],
          "stride_h": ["INT", "CODE", "self.attr_pick(node['Conv'], 'strides', [1, 1])[0]"],
          "dilation":
             ['INT',
              'CODE',
              "self.attr_pick(node['Conv'], 'dilations')"\
              " if isinstance(self.attr_pick(node['Conv'], 'dilations'), int)"\
              " else self.attr_pick(node['Conv'], 'dilations')[0]"],
          "padding":
              ["STRING",
               "CODE",
               "'SAME'"\
               " if self.attr_pick(node['Conv'], 'auto_pad', 'NOTSET') in ['SAME_UPPER', 'SAME_LOWER'] else "\
               "'VALID' "],
          "pad":
              ["INTS",
               "CODE",
               "[p for p in self.array_layout(self.attr_pick(node['Conv'], 'pads', [ 0, 0, 0, 0]), [0, 2, 1, 3])]"]
          }
     },
"blob_map": {"convolution":
                 {"weight": ["CODE", "self.tensor_to_numpy(tensor['Constant_0:out0'])"],
                  "bias": ["CODE", "self.conv_bias(tensor['Constant_0:out0'], tensor['Constant_1:out0'])"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "self.is_single_refs(tensor['Conv:out0']) == True",
"src_ops_main_version": None,
"src_ops_minior_version": None}
ruler_list.append(r_conv_add)

r_dconvolution = {
"ruler_name": "r_dconvolution",
"src_ops_alias": ["ConvTranspose", "Constant_0", "Constant_1"],
"src_inter_flow": [ ["Constant_0:out0", "ConvTranspose:in1"], ["Constant_1:out0", "ConvTranspose:in2"]],
"src_in_anchor": [["I:out0", "ConvTranspose:in0"]],
"src_out_tensor": ["ConvTranspose:out0"],
"acu_lys_alias": ["deconvolution"],
"src_acu_in_tensor_map": [["I:out0", "deconvolution:in0"]],
"src_acu_out_tensor_map": [["ConvTranspose:out0", "deconvolution:out0"]],
"param_map":
    {"deconvolution":
         {"weights": ['INT', 'PYFUNC', r_get_deconv_weights(weight='Constant_0:out0')],
          "output_shape": ["INTS", "CODE", "self.attr_pick(node['ConvTranspose'], '_out_shape')[0]"],
          "pad_h": ["INT", "CODE", "self.attr_pick(node['ConvTranspose'], 'pads', [0, 0, 0, 0])[0]"],
          "bias": ["BOOL", "VALUE", True],
          "ksize_w": ["INT", "PYFUNC", r_dconv_get_kernel_shape(weight='Constant_0:out0', dim='width')],
          "group_number": ["INT", "CODE", "self.attr_pick(node['ConvTranspose'], 'group', 1)"],
          "ksize_h": ["INT", "PYFUNC", r_dconv_get_kernel_shape(weight='Constant_0:out0', dim='height')],
          "stride_w": ["INT", "CODE", "self.attr_pick(node['ConvTranspose'], 'strides', [1, 1])[1]"],
          "stride_h": ["INT", "CODE", "self.attr_pick(node['ConvTranspose'], 'strides', [1, 1])[0]"],
          "padding":
          ["STRING",
           "CODE",
           "'SAME'"\
           " if self.attr_pick(node['ConvTranspose'], 'auto_pad', 'NOTSET') in ['SAME_UPPER', 'SAME_LOWER'] else "\
           "'VALID' "],
          'pad_method':
          ["STRING",
           "CODE",
           "'padding_const' if self.attr_pick(node['ConvTranspose'], 'auto_pad', 'NOTSET') "
           "== 'NOTSET' else 'auto' "],
          'pad': ['INTS', 'CODE',
                  "self.array_layout(self.attr_pick(node['ConvTranspose'], 'pads', [0, 0, 0, 0]), [0, 2, 1, 3])"],
          "pad_w": ["INT", "CODE", "self.attr_pick(node['ConvTranspose'], 'pads', [0, 0, 0, 0])[1]"]}},
"blob_map": {"deconvolution": {"weight": ["CODE", "self.tensor_to_numpy(tensor['Constant_0:out0'])"],
                                 "bias": ["CODE", "self.tensor_to_numpy(tensor['Constant_1:out0'])"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "len(self.shape_pick(tensor['Constant_0:out0'])) == 4",
"src_ops_main_version": None,
"src_ops_minior_version": None}
ruler_list.append(r_dconvolution)

r_dconvolution_no_bias = {
"ruler_name": "r_dconvolution_no_bias",
"src_ops_alias": ["ConvTranspose", "Constant_0"],
"src_inter_flow": [ ["Constant_0:out0", "ConvTranspose:in1"]],
"src_in_anchor": [["I:out0", "ConvTranspose:in0"]],
"src_out_tensor": ["ConvTranspose:out0"],
"acu_lys_alias": ["deconvolution"],
"src_acu_in_tensor_map": [["I:out0", "deconvolution:in0"]],
"src_acu_out_tensor_map": [["ConvTranspose:out0", "deconvolution:out0"]],
"param_map":
    {"deconvolution":
         {"weights": ['INT', 'PYFUNC', r_get_deconv_weights(weight='Constant_0:out0')],
          "output_shape": ["INTS", "CODE", "self.attr_pick(node['ConvTranspose'], '_out_shape')[0]"],
          "pad_h": ["INT", "CODE", "self.attr_pick(node['ConvTranspose'], 'pads', [0, 0, 0, 0])[0]"],
          "bias": ["BOOL", "VALUE", False],
          "ksize_w": ["INT", "PYFUNC", r_dconv_get_kernel_shape(weight='Constant_0:out0', dim='width')],
          "group_number": ["INT", "CODE", "self.attr_pick(node['ConvTranspose'], 'group', 1)"],
          "ksize_h": ["INT", "PYFUNC", r_dconv_get_kernel_shape(weight='Constant_0:out0', dim='height')],
          "stride_w": ["INT", "CODE", "self.attr_pick(node['ConvTranspose'], 'strides', [1, 1])[1]"],
          "stride_h": ["INT", "CODE", "self.attr_pick(node['ConvTranspose'], 'strides', [1, 1])[0]"],
          "padding":
          ["STRING",
           "CODE",
           "'SAME'"\
           " if self.attr_pick(node['ConvTranspose'], 'auto_pad', 'NOTSET') in ['SAME_UPPER', 'SAME_LOWER'] else "\
           "'VALID' "],
          'pad_method':
          ["STRING",
           "CODE",
           "'padding_const' if self.attr_pick(node['ConvTranspose'], 'auto_pad', 'NOTSET') "
           "== 'NOTSET' else 'auto' "],
          'pad': ['INTS', 'CODE',
                  "self.array_layout(self.attr_pick(node['ConvTranspose'], 'pads', [0, 0, 0, 0]), [0, 2, 1, 3])"],
          "pad_w": ["INT", "CODE", "self.attr_pick(node['ConvTranspose'], 'pads', [0, 0, 0, 0])[1]"]}},
"blob_map": {"deconvolution": {"weight": ["CODE", "self.tensor_to_numpy(tensor['Constant_0:out0'])"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "len(self.shape_pick(tensor['Constant_0:out0'])) == 4",
"src_ops_main_version": None,
"src_ops_minior_version": None}
ruler_list.append(r_dconvolution_no_bias)

@rule_pyfunc_def
def r_bias_var_check(self, node, tensor, bias_tensor):
    data = self.tensor_to_numpy(tensor[bias_tensor])
    if data.ndim >= 2 and data.shape[1] == data.size:
        return True
    return False

r_deconvolution_with_add = {
"ruler_name": 'r_deconvolution_with_add',
"src_ops_alias": ["Add", "ConvTranspose", "Constant", "Constant_1"],
"src_inter_flow": [["ConvTranspose:out0", "Add:in0"], \
        ["Constant_1:out0", "Add:in1"], ["Constant:out0", "ConvTranspose:in1"]],
"src_in_anchor": [["I_0:out0", "ConvTranspose:in0"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["deconvolution"],
"src_acu_in_tensor_map": [["I_0:out0", "deconvolution:in0"]],
"src_acu_out_tensor_map": [["Add:out0", "deconvolution:out0"]],
"acu_inter_flow": [],
"param_map": {"deconvolution":
         {"weights": ['INT', 'PYFUNC', r_get_deconv_weights(weight='Constant:out0')],
          "output_shape": ["INTS", "CODE", "self.attr_pick(node['ConvTranspose'], '_out_shape')[0]"],
          "pad_h": ["INT", "CODE", "self.attr_pick(node['ConvTranspose'], 'pads', [0, 0, 0, 0])[0]"],
          "bias": ["BOOL", "VALUE", True],
          "ksize_w": ["INT", "PYFUNC", r_dconv_get_kernel_shape(weight='Constant:out0', dim='width')],
          "group_number": ["INT", "CODE", "self.attr_pick(node['ConvTranspose'], 'group', 1)"],
          "ksize_h": ["INT", "PYFUNC", r_dconv_get_kernel_shape(weight='Constant:out0', dim='height')],
          "stride_w": ["INT", "CODE", "self.attr_pick(node['ConvTranspose'], 'strides', [1, 1])[1]"],
          "stride_h": ["INT", "CODE", "self.attr_pick(node['ConvTranspose'], 'strides', [1, 1])[0]"],
          "padding":
          ["STRING",
           "CODE",
           "'SAME'"\
           " if self.attr_pick(node['ConvTranspose'], 'auto_pad', 'NOTSET') in ['SAME_UPPER', 'SAME_LOWER'] else "\
           "'VALID' "],
          'pad_method':
          ["STRING",
           "CODE",
           "'padding_const' if self.attr_pick(node['ConvTranspose'], 'auto_pad', 'NOTSET') "
           "== 'NOTSET' else 'auto' "],
          'pad': ['INTS', 'CODE',
                  "self.array_layout(self.attr_pick(node['ConvTranspose'], 'pads', [0, 0, 0, 0]), [0, 2, 1, 3])"],
          "pad_w": ["INT", "CODE", "self.attr_pick(node['ConvTranspose'], 'pads', [0, 0, 0, 0])[1]"]}},
"blob_map": {"deconvolution": {
    "weight": ["CODE", "self.tensor_to_numpy(tensor['Constant:out0'])"],
    "bias": ["CODE", "self.tensor_to_numpy(tensor['Constant_1:out0']).flatten()"],
    }},
"priority_tip": 0,
"pre_condition": r_bias_var_check(bias_tensor='Constant_1:out0'),
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_deconvolution_with_add)

r_deconv1d_no_bias = {
"ruler_name": "r_dconv1d_no_bias",
"src_ops_alias": ["ConvTranspose", "Constant_0"],
"src_inter_flow": [["Constant_0:out0", "ConvTranspose:in1"]],
"src_in_anchor": [["I:out0", "ConvTranspose:in0"]],
"src_out_tensor": ["ConvTranspose:out0"],
"src_acu_in_tensor_map": [["I:out0", "deconvolution1d:in0"]],
"src_acu_out_tensor_map": [["ConvTranspose:out0", "deconvolution1d:out0"]],
"acu_lys_alias": ["deconvolution1d"],
"acu_inter_flow": [],
"param_map": {"deconvolution1d": {'ksize': ['INT', 'CODE', "self.shape_pick(tensor['Constant_0:out0'])[2]"],
                           'stride': ['INT', 'CODE', "self.attr_pick(node['ConvTranspose'], 'strides')[0]"],
                           'bias': ['BOOL', 'VALUE', False],
                           "padding":
                           ["STRING",
                            "CODE",
                            "'SAME' if self.attr_pick(node['ConvTranspose'], 'auto_pad', 'NOTSET') in "
                            "['SAME_UPPER', 'SAME_LOWER'] else 'VALID' "],
                           'pad_method':
                           ["STRING",
                            "CODE",
                            "'padding_const' if self.attr_pick(node['ConvTranspose'], 'auto_pad', 'NOTSET') "
                            "== 'NOTSET' else 'auto' "],
                           'pad': ['INTS', 'CODE', "self.attr_pick(node['ConvTranspose'], 'pads')"],
                           'group_number': ['INT', 'CODE', "self.attr_pick(node['ConvTranspose'], 'group', 1)"],
                           'weights': ['INT', 'PYFUNC', r_get_deconv_weights(weight='Constant_0:out0')],
                           'dilation': ['INT', 'CODE', "self.attr_pick(node['ConvTranspose'], 'dilations', [1])[0]"],
                           'output_shape': ['INTS', 'CODE', "self.attr_pick(node['ConvTranspose'], '_out_shape')[0]"],
                           'output_padding': ['INT', 'CODE',
                                              "self.attr_pick(node['ConvTranspose'], 'output_padding', 0) "
                                              "if isinstance("
                                              "self.attr_pick(node['ConvTranspose'], 'output_padding', 0), int)"
                                              "else self.attr_pick(node['ConvTranspose'], 'output_padding', 0)[0]"]
                           }},
"blob_map": {"deconvolution1d": {'weight': ['CODE', "self.tensor_to_numpy(tensor['Constant_0:out0'])"]}},
"priority_tip": 0,
"pre_condition": "len(self.shape_pick(tensor['Constant_0:out0'])) == 3"}
ruler_list.append(r_deconv1d_no_bias)

r_deconv1d = {
"ruler_name": 'r_deconv1d',
"src_ops_alias": ["ConvTranspose", "Constant", "Constant_1"],
"src_inter_flow": [["Constant:out0", "ConvTranspose:in1"], ["Constant_1:out0", "ConvTranspose:in2"]],
"src_in_anchor": [["I_0:out0", "ConvTranspose:in0"]],
"src_out_tensor": ["ConvTranspose:out0"],
"acu_lys_alias": ["deconvolution1d"],
"src_acu_in_tensor_map": [["I_0:out0", "deconvolution1d:in0"]],
"src_acu_out_tensor_map": [["ConvTranspose:out0", "deconvolution1d:out0"]],
"acu_inter_flow": [],
"param_map": {"deconvolution1d": {'ksize': ['INT', 'CODE', "self.shape_pick(tensor['Constant:out0'])[2]"],
                           'stride': ['INT', 'CODE', "self.attr_pick(node['ConvTranspose'], 'strides')[0]"],
                           'bias': ['BOOL', 'VALUE', True],
                           "padding":
                           ["STRING",
                            "CODE",
                            "'SAME' if self.attr_pick(node['ConvTranspose'], 'auto_pad', 'NOTSET') in "
                            "['SAME_UPPER', 'SAME_LOWER'] else 'VALID' "],
                           'pad_method':
                           ["STRING",
                            "CODE",
                            "'padding_const' if self.attr_pick(node['ConvTranspose'], 'auto_pad', 'NOTSET') "
                            "== 'NOTSET' else 'auto' "],
                           'pad': ['INTS', 'CODE', "self.attr_pick(node['ConvTranspose'], 'pads')"],
                           'group_number': ['INT', 'CODE', "self.attr_pick(node['ConvTranspose'], 'group', 1)"],
                           'weights': ['INT', 'PYFUNC', r_get_deconv_weights(weight='Constant:out0')],
                           'dilation': ['INT', 'CODE', "self.attr_pick(node['ConvTranspose'], 'dilations', [1])[0]"],
                           'output_shape': ['INTS', 'CODE', "self.attr_pick(node['ConvTranspose'], '_out_shape')[0]"],
                           'output_padding': ['INT', 'CODE',
                                              "self.attr_pick(node['ConvTranspose'], 'output_padding', 0) "
                                              "if isinstance("
                                              "self.attr_pick(node['ConvTranspose'], 'output_padding', 0), int)"
                                              "else self.attr_pick(node['ConvTranspose'], 'output_padding', 0)[0]"]
                           }},
"blob_map": {"deconvolution1d": {'weight': ['CODE', "self.tensor_to_numpy(tensor['Constant:out0'])"],
                                 'bias': ['CODE', "self.tensor_to_numpy(tensor['Constant_1:out0'])"]}},
"priority_tip": 0,
"pre_condition": "len(self.shape_pick(tensor['Constant:out0'])) == 3",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_deconv1d)

r_deconv3d_no_bias = {
"ruler_name": 'deconv3d_no_bias',
"src_ops_alias": ["ConvTranspose", "Constant"],
"src_inter_flow": [["Constant:out0", "ConvTranspose:in1"]],
"src_in_anchor": [["I_0:out0", "ConvTranspose:in0"]],
"src_out_tensor": ["ConvTranspose:out0"],
"acu_lys_alias": ["deconvolution3d"],
"src_acu_in_tensor_map": [["I_0:out0", "deconvolution3d:in0"]],
"src_acu_out_tensor_map": [["ConvTranspose:out0", "deconvolution3d:out0"]],
"acu_inter_flow": [],
"param_map": {"deconvolution3d": {"weights": ['INT', 'PYFUNC', r_get_deconv_weights(weight='Constant:out0')],
              "output_shape": ["INTS", "CODE", "self.attr_pick(node['ConvTranspose'], '_out_shape')[0]"],
              "pad_d": ["INT", "CODE", "self.attr_pick(node['ConvTranspose'], 'pads', [0, 0, 0, 0])[0]"],
              "pad_h": ["INT", "CODE", "self.attr_pick(node['ConvTranspose'], 'pads', [0, 0, 0, 0])[1]"],
              "pad_w": ["INT", "CODE", "self.attr_pick(node['ConvTranspose'], 'pads', [0, 0, 0, 0])[2]"],
              "bias": ["BOOL", "VALUE", False],
              "ksize_d": ["INT", "PYFUNC", r_dconv_get_kernel_shape(weight='Constant:out0', dim='depth')],
              "ksize_h": ["INT", "PYFUNC", r_dconv_get_kernel_shape(weight='Constant:out0', dim='height')],
              "ksize_w": ["INT", "PYFUNC", r_dconv_get_kernel_shape(weight='Constant:out0', dim='width')],
              "group_number": ["INT", "CODE", "self.attr_pick(node['ConvTranspose'], 'group', 1)"],
              "stride_d": ["INT", "CODE", "self.attr_pick(node['ConvTranspose'], 'strides', [1, 1])[0]"],
              "stride_h": ["INT", "CODE", "self.attr_pick(node['ConvTranspose'], 'strides', [1, 1])[1]"],
              "stride_w": ["INT", "CODE", "self.attr_pick(node['ConvTranspose'], 'strides', [1, 1])[2]"],
              "padding":
              ["STRING", "CODE", "'SAME'"
               " if self.attr_pick(node['ConvTranspose'], 'auto_pad', 'NOTSET') in ['SAME_UPPER', 'SAME_LOWER'] else "
               "'VALID' "],
              'pad_method': ["STRING", "CODE", "'padding_const' if self.attr_pick(node['ConvTranspose'], "
                                               "'auto_pad', 'NOTSET') == 'NOTSET' else 'auto' "],
              'pad': ['INTS', 'CODE',
                    "self.array_layout(self.attr_pick(node['ConvTranspose'],"
                    " 'pads', [0, 0, 0, 0, 0, 0]), [0, 3, 1, 4, 2, 5])"],
              }},
"blob_map": {"deconvolution3d": {'weight': ['CODE', "self.tensor_to_numpy(tensor['Constant:out0'])"]}},
"priority_tip": 0,
"pre_condition": "len(self.shape_pick(tensor['Constant:out0'])) == 5",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_deconv3d_no_bias)

r_deconv3d = {
"ruler_name": 'deconv3d',
"src_ops_alias": ["ConvTranspose", "Constant", "Constant_1"],
"src_inter_flow": [["Constant:out0", "ConvTranspose:in1"], ["Constant_1:out0", "ConvTranspose:in2"]],
"src_in_anchor": [["I_0:out0", "ConvTranspose:in0"]],
"src_out_tensor": ["ConvTranspose:out0"],
"acu_lys_alias": ["deconvolution3d"],
"src_acu_in_tensor_map": [["I_0:out0", "deconvolution3d:in0"]],
"src_acu_out_tensor_map": [["ConvTranspose:out0", "deconvolution3d:out0"]],
"acu_inter_flow": [],
"param_map": {"deconvolution3d": {"weights": ['INT', 'PYFUNC', r_get_deconv_weights(weight='Constant:out0')],
              "output_shape": ["INTS", "CODE", "self.attr_pick(node['ConvTranspose'], '_out_shape')[0]"],
              "pad_d": ["INT", "CODE", "self.attr_pick(node['ConvTranspose'], 'pads', [0, 0, 0, 0])[0]"],
              "pad_h": ["INT", "CODE", "self.attr_pick(node['ConvTranspose'], 'pads', [0, 0, 0, 0])[1]"],
              "pad_w": ["INT", "CODE", "self.attr_pick(node['ConvTranspose'], 'pads', [0, 0, 0, 0])[2]"],
              "bias": ["BOOL", "VALUE", True],
              "ksize_d": ["INT", "PYFUNC", r_dconv_get_kernel_shape(weight='Constant:out0', dim='depth')],
              "ksize_h": ["INT", "PYFUNC", r_dconv_get_kernel_shape(weight='Constant:out0', dim='height')],
              "ksize_w": ["INT", "PYFUNC", r_dconv_get_kernel_shape(weight='Constant:out0', dim='width')],
              "group_number": ["INT", "CODE", "self.attr_pick(node['ConvTranspose'], 'group', 1)"],
              "stride_d": ["INT", "CODE", "self.attr_pick(node['ConvTranspose'], 'strides', [1, 1])[0]"],
              "stride_h": ["INT", "CODE", "self.attr_pick(node['ConvTranspose'], 'strides', [1, 1])[1]"],
              "stride_w": ["INT", "CODE", "self.attr_pick(node['ConvTranspose'], 'strides', [1, 1])[2]"],
              "padding":
              ["STRING", "CODE", "'SAME'"
               " if self.attr_pick(node['ConvTranspose'], 'auto_pad', 'NOTSET') in ['SAME_UPPER', 'SAME_LOWER'] else "
               "'VALID' "],
              'pad_method': ["STRING", "CODE", "'padding_const' if self.attr_pick(node['ConvTranspose'], "
                                               "'auto_pad', 'NOTSET') == 'NOTSET' else 'auto' "],
              'pad': ['INTS', 'CODE',
                    "self.array_layout(self.attr_pick(node['ConvTranspose'],"
                    " 'pads', [0, 0, 0, 0, 0, 0]), [0, 3, 1, 4, 2, 5])"],
              }},
"blob_map": {"deconvolution3d": {'weight': ['CODE', "self.tensor_to_numpy(tensor['Constant:out0'])"],
                                 'bias': ['CODE', "self.tensor_to_numpy(tensor['Constant_1:out0'])"]}},
"priority_tip": 0,
"pre_condition": "len(self.shape_pick(tensor['Constant:out0'])) == 5",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_deconv3d)

r_bn_v6 = {
"ruler_name": "r_bn_v6",
"src_ops_alias": ["BatchNormalization", "Constant_0", "Constant_1", "Constant_2", "Constant_3"],
"src_inter_flow": [["Constant_2:out0", "BatchNormalization:in4"], ["Constant_1:out0", "BatchNormalization:in2"],
                   ["Constant_3:out0", "BatchNormalization:in3"], ["Constant_0:out0", "BatchNormalization:in1"]],
"src_in_anchor": [["I:out0", "BatchNormalization:in0"]],
"src_out_tensor": ["BatchNormalization:out0"],
"acu_lys_alias": ["batchnormalize"],
"src_acu_in_tensor_map": [["I:out0", "batchnormalize:in0"]],
"src_acu_out_tensor_map": [["BatchNormalization:out0", "batchnormalize:out0"]],
"param_map": {"batchnormalize":
                  {"eps": ["FLOAT", "CODE", "self.attr_pick(node['BatchNormalization'], 'epsilon', 1e-5)"]}},
"blob_map":
    {"batchnormalize":
         {"gamma": ["CODE", "self.tensor_to_numpy(tensor['Constant_0:out0'])"],
          "beta": ["CODE", "self.tensor_to_numpy(tensor['Constant_1:out0'])"],
          "variance": ["CODE", "self.tensor_to_numpy(tensor['Constant_2:out0'])"],
          "mean": ["CODE", "self.tensor_to_numpy(tensor['Constant_3:out0'])"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "self.attr_pick(node['BatchNormalization'], 'training_mode', 0) == 0",
"src_ops_main_version": None,
"src_ops_minior_version": [6, -1]}
ruler_list.append(r_bn_v6)

r_bn_v7 = {
    "ruler_name": "r_bn_v8_bn_5_outputs",
    "src_ops_alias": ["BatchNormalization", "Constant_0", "Constant_1", "Constant_2", "Constant_3"],
    "src_inter_flow": [["Constant_2:out0", "BatchNormalization:in4"], ["Constant_1:out0", "BatchNormalization:in2"],
                       ["Constant_3:out0", "BatchNormalization:in3"], ["Constant_0:out0", "BatchNormalization:in1"]],
    "src_in_anchor": [["I:out0", "BatchNormalization:in0"]],
    "src_out_tensor": ["BatchNormalization:out0", "BatchNormalization:out1", "BatchNormalization:out2",
                       "BatchNormalization:out3", "BatchNormalization:out4"],
    "acu_lys_alias": ["batchnormalize"],
    "src_acu_in_tensor_map": [["I:out0", "batchnormalize:in0"]],
    "src_acu_out_tensor_map": [["BatchNormalization:out0", "batchnormalize:out0"]],
    "param_map": {"batchnormalize":
                      {"eps": ["FLOAT", "CODE", "self.attr_pick(node['BatchNormalization'], 'epsilon', 1e-5)"]}},
    "blob_map":
        {"batchnormalize":
             {"gamma": ["CODE", "self.tensor_to_numpy(tensor['Constant_0:out0'])"],
              "beta": ["CODE", "self.tensor_to_numpy(tensor['Constant_1:out0'])"],
              "variance": ["CODE", "self.tensor_to_numpy(tensor['Constant_2:out0'])"],
              "mean": ["CODE", "self.tensor_to_numpy(tensor['Constant_3:out0'])"]}},
    "acu_inter_flow": [],
    "priority_tip": 0,
    "pre_condition": None,
    "src_ops_main_version": None,
    "src_ops_minior_version": [1, 13]}
ruler_list.append(r_bn_v7)

r_bn_v8 = {
    "ruler_name": "r_bn_v8_bn_3_outputs",
    "src_ops_alias": ["BatchNormalization", "Constant_0", "Constant_1", "Constant_2", "Constant_3"],
    "src_inter_flow": [["Constant_2:out0", "BatchNormalization:in4"], ["Constant_1:out0", "BatchNormalization:in2"],
                       ["Constant_3:out0", "BatchNormalization:in3"], ["Constant_0:out0", "BatchNormalization:in1"]],
    "src_in_anchor": [["I:out0", "BatchNormalization:in0"]],
    "src_out_tensor": ["BatchNormalization:out0", "BatchNormalization:out1", "BatchNormalization:out2"],
    "acu_lys_alias": ["batchnormalize"],
    "src_acu_in_tensor_map": [["I:out0", "batchnormalize:in0"]],
    "src_acu_out_tensor_map": [["BatchNormalization:out0", "batchnormalize:out0"]],
    "param_map": {"batchnormalize":
                      {"eps": ["FLOAT", "CODE", "self.attr_pick(node['BatchNormalization'], 'epsilon', 1e-5)"]}},
    "blob_map":
        {"batchnormalize":
             {"gamma": ["CODE", "self.tensor_to_numpy(tensor['Constant_0:out0'])"],
              "beta": ["CODE", "self.tensor_to_numpy(tensor['Constant_1:out0'])"],
              "variance": ["CODE", "self.tensor_to_numpy(tensor['Constant_2:out0'])"],
              "mean": ["CODE", "self.tensor_to_numpy(tensor['Constant_3:out0'])"]}},
    "acu_inter_flow": [],
    "priority_tip": 0,
    "pre_condition": None,
    "src_ops_main_version": None,
    "src_ops_minior_version": [1, -1]}
ruler_list.append(r_bn_v8)

r_bn_c4 = {
"ruler_name": "r_bn_c4",
"src_ops_alias": ["BatchNormalization", "Constant_0", "Constant_1", "Constant_2", "Constant_3"],
"src_inter_flow": [["Constant_0:out0", "BatchNormalization:in1"], ["Constant_1:out0", "BatchNormalization:in2"],
                   ["Constant_2:out0", "BatchNormalization:in3"], ["Constant_3:out0", "BatchNormalization:in4"]],
"src_in_anchor": [["I:out0", "BatchNormalization:in0"]],
"src_out_tensor": ["BatchNormalization:out0"],
"acu_lys_alias": ["batchnormalize"],
"src_acu_in_tensor_map": [["I:out0", "batchnormalize:in0"]],
"src_acu_out_tensor_map": [["BatchNormalization:out0", "batchnormalize:out0"]],
"param_map":
{"batchnormalize": {"eps": ["FLOAT", "CODE", "self.attr_pick(node['BatchNormalization'], 'epsilon', 9e-6)"],
}},
"blob_map":{"batchnormalize": {
 "gamma": ['CODE', "self.tensor_to_numpy(tensor['Constant_0:out0'])"],
 "bata": ['CODE', "self.tensor_to_numpy(tensor['Constant_1:out0'])"],
 "mean": ['CODE', "self.tensor_to_numpy(tensor['Constant_2:out0'])"],
 "var": ['CODE', "self.tensor_to_numpy(tensor['Constant_3:out0'])"],}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "self.attr_pick(node['BatchNormalization'], 'spatial') == 1",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_bn_c4)


r_bn_v5 = {
"ruler_name": "r_bn_v5",
"src_ops_alias": ["BatchNormalization", "Constant_0", "Constant_1", "Constant_2", "Constant_3"],
"src_inter_flow": [["Constant_2:out0", "BatchNormalization:in4"], ["Constant_1:out0", "BatchNormalization:in2"],
                   ["Constant_3:out0", "BatchNormalization:in3"], ["Constant_0:out0", "BatchNormalization:in1"]],
"src_in_anchor": [["I:out0", "BatchNormalization:in0"]],
"src_out_tensor": ["BatchNormalization:out0"],
"acu_lys_alias": ["batchnormalize"],
"src_acu_in_tensor_map": [["I:out0", "batchnormalize:in0"]],
"src_acu_out_tensor_map": [["BatchNormalization:out0", "batchnormalize:out0"]],
"param_map":
    {"batchnormalize": {"eps": ["FLOAT", "CODE", "self.attr_pick(node['BatchNormalization'], 'epsilon', 1e-5)"]}},
"blob_map":
{"batchnormalize":
{"gamma":
  ["CODE",
   "None "\
   "if self.attr_pick(node['BatchNormalization'], 'consumed_inputs')[1] == 0 else"\
   " self.tensor_to_numpy(tensor['Constant_0:out0'])"],
"beta":
  ["CODE",
   "None "\
   "if self.attr_pick(node['BatchNormalization'], 'consumed_inputs')[2] == 0 else"\
   " self.tensor_to_numpy(tensor['Constant_1:out0'])"],
"variance":
  ["CODE",
   "None "\
   "if self.attr_pick(node['BatchNormalization'], 'consumed_inputs')[3] == 0 else"\
   " self.tensor_to_numpy(tensor['Constant_2:out0'])"],
"mean":
  ["CODE",
   "None "\
   "if self.attr_pick(node['BatchNormalization'], 'consumed_inputs')[4] == 0 else"\
   " self.tensor_to_numpy(tensor['Constant_3:out0'])"],
}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, 5]}
ruler_list.append(r_bn_v5)

r_bn_v4 = {
"ruler_name": "r_bn_v4",
"src_ops_alias": ["BatchNormalization", "Constant", "Constant_1", "Constant_2"],
"src_inter_flow": [["Constant:out0", "BatchNormalization:in1"], ["Constant_1:out0", "BatchNormalization:in2"],
                   ["Constant_1:out0", "BatchNormalization:in3"], ["Constant_2:out0", "BatchNormalization:in4"]],
"src_in_anchor": [["I:out0", "BatchNormalization:in0"]],
"src_out_tensor": ["BatchNormalization:out0"],
"acu_lys_alias": ["batchnormalize"],
"src_acu_in_tensor_map": [["I:out0", "batchnormalize:in0"]],
"src_acu_out_tensor_map": [["BatchNormalization:out0", "batchnormalize:out0"]],
"param_map":
    {"batchnormalize": {"eps": ["FLOAT", "CODE", "self.attr_pick(node['BatchNormalization'], 'epsilon', 1e-5)"]}},
"blob_map":
    {"batchnormalize":
         {"gamma": ["CODE", "self.tensor_to_numpy(tensor['Constant:out0'])"],
          "beta": ["CODE", "self.tensor_to_numpy(tensor['Constant_1:out0'])"],
          "variance": ["CODE", "self.tensor_to_numpy(tensor['Constant_2:out0'])"],
          "mean": ["CODE", "self.tensor_to_numpy(tensor['Constant_1:out0'])"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_bn_v4)

r_bn_v3 = {
"ruler_name": "r_bn_v5",
"src_ops_alias": ["BatchNormalization", "Constant", "Constant_1"],
"src_inter_flow": [["Constant:out0", "BatchNormalization:in1"], ["Constant_1:out0", "BatchNormalization:in2"],
                   ["Constant_1:out0", "BatchNormalization:in3"], ["Constant:out0", "BatchNormalization:in4"]],
"src_in_anchor": [["I:out0", "BatchNormalization:in0"]],
"src_out_tensor": ["BatchNormalization:out0"],
"acu_lys_alias": ["batchnormalize"],
"src_acu_in_tensor_map": [["I:out0", "batchnormalize:in0"]],
"src_acu_out_tensor_map": [["BatchNormalization:out0", "batchnormalize:out0"]],
"param_map":
    {"batchnormalize": {"eps": ["FLOAT", "CODE", "self.attr_pick(node['BatchNormalization'], 'epsilon', 1e-5)"]}},
"blob_map":
    {"batchnormalize":
         {"gamma": ["CODE", "self.tensor_to_numpy(tensor['Constant:out0'])"],
          "beta": ["CODE", "self.tensor_to_numpy(tensor['Constant_1:out0'])"],
          "variance": ["CODE", "self.tensor_to_numpy(tensor['Constant:out0'])"],
          "mean": ["CODE", "self.tensor_to_numpy(tensor['Constant:out0'])"]}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_bn_v3)

r_bn_mul_add_v5 = {
"ruler_name": "r_bn_mul_add_v5",
"src_ops_alias": ["BatchNormalization", "Constant_0", "Constant_1", "Constant_2", "Constant_3",
                  "Mul", "Constant_4", "Add", "Constant_5"],
"src_inter_flow": [["Constant_2:out0", "BatchNormalization:in4"], ["Constant_1:out0", "BatchNormalization:in2"],
                   ["Constant_3:out0", "BatchNormalization:in3"], ["Constant_0:out0", "BatchNormalization:in1"],
                   ["BatchNormalization:out0", "Mul:in0"], ["Constant_4:out0", "Mul:in1"],
                   ["Mul:out0", "Add:in0"], ["Constant_5:out0", "Add:in1"]],
"src_in_anchor": [["I:out0", "BatchNormalization:in0"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["batchnormalize"],
"src_acu_in_tensor_map": [["I:out0", "batchnormalize:in0"]],
"src_acu_out_tensor_map": [["Add:out0", "batchnormalize:out0"]],
"param_map":
    {"batchnormalize": {"eps": ["FLOAT", "CODE", "self.attr_pick(node['BatchNormalization'], 'epsilon', 1e-5)"]}},
"blob_map":
{"batchnormalize":
{"gamma":
  ["CODE",
   "self.tensor_to_numpy(tensor['Constant_4:out0']) "\
   "if self.attr_pick(node['BatchNormalization'], 'consumed_inputs')[1] == 0 else"\
   " self.tensor_to_numpy(tensor['Constant_0:out0']) * self.tensor_to_numpy(tensor['Constant_4:out0'])"],
"beta":
  ["CODE",
   "self.tensor_to_numpy(tensor['Constant_5:out0']) "\
   "if self.attr_pick(node['BatchNormalization'], 'consumed_inputs')[1] == 0 else"\
   " self.tensor_to_numpy(tensor['Constant_1:out0']) * self.tensor_to_numpy(tensor['Constant_4:out0']) + "\
   "self.tensor_to_numpy(tensor['Constant_5:out0'])"],
"variance":
  ["CODE",
   "None "\
   "if self.attr_pick(node['BatchNormalization'], 'consumed_inputs')[3] == 0 else"\
   " self.tensor_to_numpy(tensor['Constant_2:out0'])"],
"mean":
  ["CODE",
   "None "\
   "if self.attr_pick(node['BatchNormalization'], 'consumed_inputs')[4] == 0 else"\
   " self.tensor_to_numpy(tensor['Constant_3:out0'])"],
}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, 5]}
ruler_list.append(r_bn_mul_add_v5)

r_bn_mul_add_v6 = {
"ruler_name": "r_bn_mul_add_v6",
"src_ops_alias": ["BatchNormalization", "Constant_0", "Constant_1", "Constant_2", "Constant_3",
                  "Mul", "Constant_4", "Add", "Constant_5"],
"src_inter_flow": [["Constant_2:out0", "BatchNormalization:in4"], ["Constant_1:out0", "BatchNormalization:in2"],
                   ["Constant_3:out0", "BatchNormalization:in3"], ["Constant_0:out0", "BatchNormalization:in1"],
                   ["BatchNormalization:out0", "Mul:in0"], ["Constant_4:out0", "Mul:in1"],
                   ["Mul:out0", "Add:in0"], ["Constant_5:out0", "Add:in1"]],
"src_in_anchor": [["I:out0", "BatchNormalization:in0"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["batchnormalize"],
"src_acu_in_tensor_map": [["I:out0", "batchnormalize:in0"]],
"src_acu_out_tensor_map": [["Add:out0", "batchnormalize:out0"]],
"param_map":
    {"batchnormalize": {"eps": ["FLOAT", "CODE", "self.attr_pick(node['BatchNormalization'], 'epsilon', 1e-5)"]}},
"blob_map":
{"batchnormalize":
{"gamma":
  ["CODE",
   " self.judge_shape(self.tensor_to_numpy(tensor['Constant_0:out0']), " \
   "self.tensor_to_numpy(tensor['Constant_4:out0']))"],
"beta":
  ["CODE",
   " self.judge_shape(self.tensor_to_numpy(tensor['Constant_1:out0']), " \
   "self.tensor_to_numpy(tensor['Constant_4:out0']), self.tensor_to_numpy(tensor['Constant_5:out0']))"],
"variance":
  ["CODE",
   " self.tensor_to_numpy(tensor['Constant_2:out0'])"],
"mean":
  ["CODE",
   " self.tensor_to_numpy(tensor['Constant_3:out0'])"],
}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "self.attr_pick(node['BatchNormalization'], 'training_mode', 0) == 0" \
                 and r_batchnormal_v6_cond(name0='Constant_0:out0', name1='Constant_1:out0', \
                                           name2='Constant_4:out0', name3='Constant_5:out0'),
"src_ops_main_version": None,
"src_ops_minior_version": [6, -1]}
ruler_list.append(r_bn_mul_add_v6)

r_mm_bn_extend_1 = {
"ruler_name": "r_mm_bn_extend_1",
"src_ops_alias": ["Add", "Div", "Constant", "Mul", "Pow", "Constant_1", "Sub", "Add_1", "Constant_2",
                  "ReduceMean", "Mul_1", "Constant_3", "Sqrt", "Abs", "Sub_1", "ReduceMean_1",
                  "Mul_2", "Mul_3", "ReduceMean_2"],
"src_inter_flow": [["Div:out0", "Add:in0"], ["Constant:out0", "Add:in1"], ["Mul:out0", "Div:in0"],
                   ["Pow:out0", "Div:in1"], ["Constant_1:out0", "Mul:in0"], ["Sub:out0", "Mul:in1"],
                   ["Add_1:out0", "Pow:in0"], ["Constant_2:out0", "Pow:in1"], ["ReduceMean:out0", "Sub:in1"],
                   ["Mul_1:out0", "Add_1:in0"], ["Constant_3:out0", "Add_1:in1"], ["Sqrt:out0", "Mul_1:in0"],
                   ["Sqrt:out0", "Mul_1:in1"], ["Abs:out0", "Sqrt:in0"], ["Sub_1:out0", "Abs:in0"],
                   ["ReduceMean_1:out0", "Sub_1:in0"], ["Mul_2:out0", "Sub_1:in1"], ["Mul_3:out0", "ReduceMean_1:in0"],
                   ["ReduceMean_2:out0", "Mul_2:in0"], ["ReduceMean_2:out0", "Mul_2:in1"]],
"src_in_anchor": [["I_0:out0", "ReduceMean:in0"], ["I_0:out0", "Sub:in0"], ["I_0:out0", "Mul_3:in0"],
                  ["I_0:out0", "Mul_3:in1"], ["I_0:out0", "ReduceMean_2:in0"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["moments", "batchnorm_single"],
"src_acu_in_tensor_map": [["I_0:out0", "moments:in0"], ["I_0:out0", "batchnorm_single:in0"]],
"src_acu_out_tensor_map": [["Add:out0", "batchnorm_single:out0"]],
"acu_inter_flow": [["moments:out0", "batchnorm_single:in1"], ["moments:out1", "batchnorm_single:in2"]],
"param_map": {"moments": {"axis_list": ["INTS", "CODE", "self.reducex_axis_list("
                                                        "node['ReduceMean'], self.shape_pick(tensor['I_0:out0']))"],
                          "keep_dims": ["BOOL", "CODE", "self.attr_pick(node['ReduceMean'],'keepdims', False)"]},
              "batchnorm_single": {"eps": ["FLOAT", "CODE", "self.tensor_to_numpy(tensor['Constant_3:out0'])"]}},
"blob_map": {"batchnorm_single": {"scale": ["CODE", "self.tensor_to_numpy(tensor['Constant_1:out0'])"],
             "bias": ["CODE", "self.tensor_to_numpy(tensor['Constant:out0'])"]}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_mm_bn_extend_1)

r_mm_bn_extend_2 = {
"ruler_name": "r_mm_bn_extend_2",
"src_ops_alias": ["Add", "Div", "Constant", "Mul", "Pow", "Constant_1", "Sub", "Add_1",
                  "Constant_2", "ReduceMean", "ReduceMean_1", "Constant_3", "ReduceMean_2",
                  "ReduceMean_3", "Pow_1", "Sub_1", "Constant_4"],
"src_inter_flow": [["Div:out0", "Add:in0"], ["Constant:out0", "Add:in1"], ["Mul:out0", "Div:in0"],
                   ["Pow:out0", "Div:in1"], ["Constant_1:out0", "Mul:in0"], ["Sub:out0", "Mul:in1"],
                   ["Add_1:out0", "Pow:in0"], ["Constant_2:out0", "Pow:in1"], ["ReduceMean:out0", "Sub:in1"],
                   ["ReduceMean_1:out0", "Add_1:in0"], ["Constant_3:out0", "Add_1:in1"],
                   ["ReduceMean_2:out0", "ReduceMean:in0"], ["ReduceMean_3:out0", "ReduceMean_1:in0"],
                   ["Pow_1:out0", "ReduceMean_3:in0"], ["Sub_1:out0", "Pow_1:in0"], ["Constant_4:out0", "Pow_1:in1"],
                   ["ReduceMean:out0", "Sub_1:in1"]],
"src_in_anchor": [["I_0:out0", "ReduceMean_2:in0"], ["I_0:out0", "Sub_1:in0"], ["I_0:out0", "Sub:in0"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["moments", "batchnorm_single"],
"src_acu_in_tensor_map": [["I_0:out0", "moments:in0"], ["I_0:out0", "batchnorm_single:in0"]],
"src_acu_out_tensor_map": [["Add:out0", "batchnorm_single:out0"]],
"acu_inter_flow": [["moments:out0", "batchnorm_single:in1"], ["moments:out1", "batchnorm_single:in2"]],
"param_map": {"moments": {"axis_list": ["INTS", "CODE", "self.reducex_axis_list("
                                                        "node['ReduceMean_2'],"
                                                        "self.shape_pick(tensor['I_0:out0'])) + "
                                                        "self.reducex_axis_list("
                                                        "node['ReduceMean'],"
                                                        "self.shape_pick(tensor['ReduceMean_2:out0']))"],
                          "keep_dims": ["BOOL", "CODE", "self.attr_pick(node['ReduceMean'], 'keepdims', False)"]},
              "batchnorm_single": {"eps": ["FLOAT", "CODE", "self.tensor_to_numpy(tensor['Constant_3:out0'])"]}},
"blob_map": {"batchnorm_single": {"scale": ["CODE", "self.tensor_to_numpy(tensor['Constant_1:out0'])"],
             "bias": ["CODE", "self.tensor_to_numpy(tensor['Constant:out0'])"]}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_mm_bn_extend_2)

r_mm_bn_extend_3 = {
"ruler_name": "r_mm_bn_extend_3",
"src_ops_alias": ["Add", "Mul", "Constant", "Constant_1", "Div", "Sub", "Sqrt", "ReduceMean", "Add_1",
                  "ReduceMean_1", "Constant_2", "Pow", "Sub_1", "Constant_3"],
"src_inter_flow": [["Mul:out0", "Add:in0"], ["Constant:out0", "Add:in1"], ["Constant_1:out0", "Mul:in0"],
                   ["Div:out0", "Mul:in1"], ["Sub:out0", "Div:in0"], ["Sqrt:out0", "Div:in1"],
                   ["ReduceMean:out0", "Sub:in1"], ["Add_1:out0", "Sqrt:in0"], ["ReduceMean_1:out0", "Add_1:in0"],
                   ["Constant_2:out0", "Add_1:in1"], ["Pow:out0", "ReduceMean_1:in0"], ["Sub_1:out0", "Pow:in0"],
                   ["Constant_3:out0", "Pow:in1"], ["ReduceMean:out0", "Sub_1:in1"]],
"src_in_anchor": [["I_0:out0", "ReduceMean:in0"], ["I_0:out0", "Sub_1:in0"], ["I_0:out0", "Sub:in0"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["moments", "batchnorm_single"],
"src_acu_in_tensor_map": [["I_0:out0", "moments:in0"], ["I_0:out0", "batchnorm_single:in0"]],
"src_acu_out_tensor_map": [["Add:out0", "batchnorm_single:out0"]],
"acu_inter_flow": [["moments:out0", "batchnorm_single:in1"], ["moments:out1", "batchnorm_single:in2"]],
"param_map": {"moments": {"axis_list": ["INTS", "CODE", "self.reducex_axis_list("
                                                        "node['ReduceMean'],"
                                                        "self.shape_pick(tensor['I_0:out0']))"],
                          "keep_dims": ["BOOL", "CODE", "self.attr_pick(node['ReduceMean'], 'keepdims', False)"]},
              "batchnorm_single": {"eps": ["FLOAT", "CODE", "self.tensor_to_numpy(tensor['Constant_2:out0'])"]}},
"blob_map": {"batchnorm_single": {"scale": ["CODE", "self.tensor_to_numpy(tensor['Constant_1:out0'])"],
             "bias": ["CODE", "self.tensor_to_numpy(tensor['Constant:out0'])"]}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_mm_bn_extend_3)

r_mm_nb_mean_variance_normalization = {
"ruler_name": "r_mm_nb_mean_variance_normalization",
"src_ops_alias": ["MeanVarianceNormalization"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "MeanVarianceNormalization:in0"]],
"src_out_tensor": ["MeanVarianceNormalization:out0"],
"acu_lys_alias": ["moments", "batchnorm_single"],
"src_acu_in_tensor_map": [["I_0:out0", "moments:in0"], ["I_0:out0", "batchnorm_single:in0"]],
"src_acu_out_tensor_map": [["MeanVarianceNormalization:out0", "batchnorm_single:out0"]],
"acu_inter_flow": [["moments:out0", "batchnorm_single:in1"], ["moments:out1", "batchnorm_single:in2"]],
"param_map":{
    "moments": {"axis_list": ["INTS", "CODE", "self.reducex_axis_list("
                                "node['MeanVarianceNormalization'], self.shape_pick(tensor['I_0:out0']))"],
                "keep_dims": ["BOOL", "CODE", "self.attr_pick(node['MeanVarianceNormalization'],'keepdims', True)"]},
    "batchnorm_single": {"eps": ["FLOAT", "CODE", "self.attr_pick(node['MeanVarianceNormalization'], 'eps', 10e-9)"]}
},
"blob_map": {"batchnorm_single": {"scale": ['CODE', "np.array([1], dtype=np.float32)"],
             "bias": ['CODE', "np.array([0], dtype=np.float32)"]}},
"priority_tip": 0,
"pre_condition": "self.attr_pick(node['MeanVarianceNormalization'], 'axes', [0, 2, 3]) != [0, 2, 3]",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_mm_nb_mean_variance_normalization)

r_maxpool = {
"ruler_name": "r_maxpool",
"src_ops_alias": ["MaxPool"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "MaxPool:in0"]],
"src_out_tensor": ["MaxPool:out0"],
"acu_lys_alias": ["pooling"],
"src_acu_in_tensor_map": [["I:out0", "pooling:in0"]],
"src_acu_out_tensor_map": [["MaxPool:out0", "pooling:out0"]],
"param_map": {"pooling":
  {"type": ["STRING", "VALUE", "MAX"],
   "pad_method": ["STRING", "CODE",
                  "'padding_const' if self.attr_pick(node['MaxPool'], 'auto_pad', None) == None or "
                  "self.attr_pick(node['MaxPool'], 'auto_pad') == 'NOTSET' "
                  "else 'auto'"],
   "pad_h": ["INT", "CODE", "self.attr_pick(node['MaxPool'], 'pads', [0, 0, 0, 0])[0]"],
   "ksize_w": ["INT", "CODE", "self.attr_pick(node['MaxPool'], 'kernel_shape')[1]"],
   "stride_w": ["INT", "CODE", "self.attr_pick(node['MaxPool'], 'strides', [1, 1])[1]"],
   "ksize_h": ["INT", "CODE", "self.attr_pick(node['MaxPool'], 'kernel_shape')[0]"],
   "round_type": ["STRING", "CODE", "'ceil' if self.attr_pick(node['MaxPool'], 'ceil_mode') == 1 else 'floor'"],
   "stride_h": ["INT", "CODE", "self.attr_pick(node['MaxPool'], 'strides', [1, 1])[0]"],
   "padding": ['STRING', 'PYFUNC', r_pool_padding(pool_type='MaxPool')],
   "pad":
       ["INTS",
        "CODE",
        "[p for p in self.array_layout(self.attr_pick(node['MaxPool'], 'pads', [ 0, 0, 0, 0]), [0, 2, 1, 3])]"],
   "pad_w": ["INT", "CODE", "self.attr_pick(node['MaxPool'], 'pads', [0, 0, 0, 0])[1]"]}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "len(self.attr_pick(node['MaxPool'], 'kernel_shape')) == 2",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_maxpool)

r_maxpool_3d = {
"ruler_name": "r_maxpool3d",
"src_ops_alias": ["MaxPool"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "MaxPool:in0"]],
"src_out_tensor": ["MaxPool:out0"],
"acu_lys_alias": ["pool3d"],
"src_acu_in_tensor_map": [["I_0:out0", "pool3d:in0"]],
"src_acu_out_tensor_map": [["MaxPool:out0", "pool3d:out0"]],
"acu_inter_flow": [],
"param_map": {"pool3d": {
   "type": ["STRING", "VALUE", "MAX"],
   "pad_method": ["STRING", "CODE",
                   "'padding_const' if self.attr_pick(node['MaxPool'], 'auto_pad', None) == None or "
                   "self.attr_pick(node['MaxPool'], 'auto_pad') == 'NOTSET' "
                   "else 'auto'"],
   "ksize_d": ["INT", "CODE", "self.attr_pick(node['MaxPool'], 'kernel_shape')[0]"],
   "ksize_h": ["INT", "CODE", "self.attr_pick(node['MaxPool'], 'kernel_shape')[1]"],
   "ksize_w": ["INT", "CODE", "self.attr_pick(node['MaxPool'], 'kernel_shape')[2]"],
   "stride_d": ["INT", "CODE", "self.attr_pick(node['MaxPool'], 'strides', [1, 1, 1])[0]"],
   "stride_h": ["INT", "CODE", "self.attr_pick(node['MaxPool'], 'strides', [1, 1, 1])[1]"],
   "stride_w": ["INT", "CODE", "self.attr_pick(node['MaxPool'], 'strides', [1, 1, 1])[2]"],
   "round_type": ["STRING", "CODE", "'ceil' if self.attr_pick(node['MaxPool'], 'ceil_mode') == 1 else 'floor'"],
   "padding": ['STRING', 'PYFUNC', r_pool_padding(pool_type='MaxPool')],
   "pad_d": ["INT", "CODE", "self.attr_pick(node['MaxPool'], 'pads', [0, 0, 0, 0, 0, 0])[0]"],
   "pad_h": ["INT", "CODE", "self.attr_pick(node['MaxPool'], 'pads', [0, 0, 0, 0, 0, 0])[1]"],
   "pad_w": ["INT", "CODE", "self.attr_pick(node['MaxPool'], 'pads', [0, 0, 0, 0, 0, 0])[2]"],
   "pad":
       ["INTS",
        "CODE",
        "[p for p in self.array_layout(self.attr_pick(node['MaxPool'], 'pads',"
        " [ 0, 0, 0, 0, 0, 0]), [0, 3, 1, 4, 2, 5])]"],
   }},
"blob_map": {"pool3d": {}},
"priority_tip": 0,
"pre_condition": "len(self.attr_pick(node['MaxPool'], 'kernel_shape')) == 3",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_maxpool_3d)

r_avgpool = {
"ruler_name": "r_avgpool",
"src_ops_alias": ["AveragePool"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "AveragePool:in0"]],
"src_out_tensor": ["AveragePool:out0"],
"acu_lys_alias": ["pooling"],
"src_acu_in_tensor_map": [["I:out0", "pooling:in0"]],
"src_acu_out_tensor_map": [["AveragePool:out0", "pooling:out0"]],
"param_map": {"pooling":
{"type": ["STRING", "VALUE", "AVG"],
 "pad_method": ["STRING", "CODE",
                "'padding_const' if self.attr_pick(node['AveragePool'], 'auto_pad', None) == None or "
                "self.attr_pick(node['AveragePool'], 'auto_pad') == 'NOTSET' "
                "else 'auto'"],
"pad_h": ["INT", "CODE", "self.attr_pick(node['AveragePool'], 'pads', [0, 0, 0, 0])[0]"],
"ksize_w": ["INT", "CODE", "self.attr_pick(node['AveragePool'], 'kernel_shape')[1]"],
"stride_w": ["INT", "CODE", "self.attr_pick(node['AveragePool'], 'strides', [1, 1])[1]"],
"ksize_h": ["INT", "CODE", "self.attr_pick(node['AveragePool'], 'kernel_shape')[0]"],
"round_type": ["STRING", "CODE", "'ceil' if self.attr_pick(node['AveragePool'], 'ceil_mode') == 1 else 'floor'"],
"stride_h": ["INT", "CODE", "self.attr_pick(node['AveragePool'], 'strides', [1, 1])[0]"],
"padding":['STRING', 'PYFUNC', r_pool_padding(pool_type='AveragePool')],
"pad":
   ["INTS",
    "CODE",
    "[str(p) for p in self.array_layout(self.attr_pick(node['AveragePool'], 'pads', [ 0, 0, 0, 0]), [0, 2, 1, 3])]"
    ],
"pad_w": ["INT", "CODE", "self.attr_pick(node['AveragePool'], 'pads', [0, 0, 0, 0])[1]"],
"count_include_pad": ["INT", "CODE", "self.attr_pick(node['AveragePool'], 'count_include_pad', 0)"]}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "len(self.attr_pick(node['AveragePool'], 'kernel_shape')) == 2",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_avgpool)

r_avgpool1d = {
"ruler_name": "r_avgpool1d",
"src_ops_alias": ["AveragePool"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "AveragePool:in0"]],
"src_out_tensor": ["AveragePool:out0"],
"acu_lys_alias": ["pool1d"],
"src_acu_in_tensor_map": [["I:out0", "pool1d:in0"]],
"src_acu_out_tensor_map": [["AveragePool:out0", "pool1d:out0"]],
"param_map": {"pool1d":
{"type": ["STRING", "VALUE", "AVG"],
 "pad_method": ["STRING", "CODE",
                "'padding_const' if self.attr_pick(node['AveragePool'], 'auto_pad', None) == None or "
                "self.attr_pick(node['AveragePool'], 'auto_pad') == 'NOTSET' "
                "else 'auto'"],
"stride": ["INT", "CODE", "self.attr_pick(node['AveragePool'], 'strides', [1])[0]"],
"ksize": ["INT", "CODE", "self.attr_pick(node['AveragePool'], 'kernel_shape')[0]"],
"round_type": ["STRING", "CODE", "'ceil' if self.attr_pick(node['AveragePool'], 'ceil_mode') == 1 else 'floor'"],
"padding":['STRING', 'PYFUNC', r_pool_padding(pool_type='AveragePool')],
"pad":
   ["INTS",
    "CODE",
    "[str(p) for p in self.array_layout(self.attr_pick(node['AveragePool'], 'pads', [ 0, 0]), [0, 1])]"
    ],
}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "len(self.attr_pick(node['AveragePool'], 'kernel_shape')) == 1",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_avgpool1d)

r_avgpool3d = {
"ruler_name": "r_avgpool3d",
"src_ops_alias": ["AveragePool"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "AveragePool:in0"]],
"src_out_tensor": ["AveragePool:out0"],
"acu_lys_alias": ["pool3d"],
"src_acu_in_tensor_map": [["I_0:out0", "pool3d:in0"]],
"src_acu_out_tensor_map": [["AveragePool:out0", "pool3d:out0"]],
"acu_inter_flow": [],
"param_map": {"pool3d": {
    "type": ["STRING", "VALUE", "AVG"],
    "pad_method": ["STRING", "CODE",
                   "'padding_const' if self.attr_pick(node['AveragePool'], 'auto_pad', None) == None or "
                   "self.attr_pick(node['AveragePool'], 'auto_pad') == 'NOTSET' "
                   "else 'auto'"],
    "pad_h": ["INT", "CODE", "self.attr_pick(node['AveragePool'], 'pads', [0, 0, 0, 0, 0, 0])[0]"],
    "pad_w": ["INT", "CODE", "self.attr_pick(node['AveragePool'], 'pads', [0, 0, 0, 0, 0, 0])[1]"],
    "pad_d": ["INT", "CODE", "self.attr_pick(node['AveragePool'], 'pads', [0, 0, 0, 0, 0, 0])[2]"],
    "ksize_h": ["INT", "CODE", "self.attr_pick(node['AveragePool'], 'kernel_shape')[0]"],
    "ksize_w": ["INT", "CODE", "self.attr_pick(node['AveragePool'], 'kernel_shape')[1]"],
    "ksize_d": ["INT", "CODE", "self.attr_pick(node['AveragePool'], 'kernel_shape')[2]"],
    "stride_h": ["INT", "CODE", "self.attr_pick(node['AveragePool'], 'strides', [1, 1, 1])[0]"],
    "stride_w": ["INT", "CODE", "self.attr_pick(node['AveragePool'], 'strides', [1, 1, 1])[1]"],
    "stride_d": ["INT", "CODE", "self.attr_pick(node['AveragePool'], 'strides', [1, 1, 1])[2]"],
    "round_type": ["STRING", "CODE", "'ceil' if self.attr_pick(node['AveragePool'], 'ceil_mode') == 1 else 'floor'"],
    "padding": ['STRING', 'PYFUNC', r_pool_padding(pool_type='AveragePool')],
    "pad": ["INTS", "CODE", "[str(p) for p in self.array_layout(self.attr_pick(node['AveragePool'], "
                            "'pads', [ 0, 0, 0, 0, 0, 0]), [0, 2, 4, 1, 3, 5])]"],
    }},
"blob_map": {"pool3d": {}},
"priority_tip": 0,
"pre_condition": "len(self.attr_pick(node['AveragePool'], 'kernel_shape')) == 3",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_avgpool3d)

r_global_avgpool_3d = {
    "ruler_name": "r_global_avgpool",
    "src_ops_alias": ["GlobalAveragePool"],
    "src_inter_flow": [],
    "src_in_anchor": [["I:out0", "GlobalAveragePool:in0"]],
    "src_out_tensor": ["GlobalAveragePool:out0"],
    "acu_lys_alias": ["pool3d"],
    "src_acu_in_tensor_map": [["I:out0", "pool3d:in0"]],
    "src_acu_out_tensor_map": [["GlobalAveragePool:out0", "pool3d:out0"]],
    "param_map": {"pool3d":
                      {"type": ["STRING", "VALUE", "AVG"],
                       "global_pooling": ["BOOL", "VALUE", True],
                       }},
    "blob_map": {},
    "acu_inter_flow": [],
    "priority_tip": 0,
    "pre_condition": "len(self.shape_pick(tensor['I:out0'])) == 5",
    "src_ops_main_version": None,
    "src_ops_minior_version": [1, -1]}
ruler_list.append(r_global_avgpool_3d)

r_global_avgpool = {
"ruler_name": "r_global_avgpool",
"src_ops_alias": ["GlobalAveragePool"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "GlobalAveragePool:in0"]],
"src_out_tensor": ["GlobalAveragePool:out0"],
"acu_lys_alias": ["pooling"],
"src_acu_in_tensor_map": [["I:out0", "pooling:in0"]],
"src_acu_out_tensor_map": [["GlobalAveragePool:out0", "pooling:out0"]],
"param_map": {"pooling":
{"type": ["STRING", "VALUE", "AVG"],
"pad_method": ["STRING", "VALUE", "padding_const"],
"pad_h": ["INT", "CODE", "self.attr_pick(node['GlobalAveragePool'], 'pads', [0, 0, 0, 0])[0]"],
"round_type": ["STRING", "VALUE", "floor"],
"pad":
   ["INTS",
    "CODE",
"[str(p) for p in self.array_layout(self.attr_pick(node['GlobalAveragePool'], 'pads', [ 0, 0, 0, 0]), [0, 2, 1, 3])]"
    ],
"padding": ["STRING", "VALUE", "VALID"],
"global_pooling": ["BOOL", "VALUE", True],
"pad_w": ["INT", "CODE", "self.attr_pick(node['GlobalAveragePool'], 'pads', [0, 0, 0, 0])[1]"]}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "len(self.shape_pick(tensor['I:out0'])) == 4",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_global_avgpool)

r_global_avgpool1d = {
"ruler_name": "r_global_avgpool1d",
"src_ops_alias": ["GlobalAveragePool"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "GlobalAveragePool:in0"]],
"src_out_tensor": ["GlobalAveragePool:out0"],
"acu_lys_alias": ["pool1d"],
"src_acu_in_tensor_map": [["I:out0", "pool1d:in0"]],
"src_acu_out_tensor_map": [["GlobalAveragePool:out0", "pool1d:out0"]],
"param_map": {"pool1d":
{"type": ["STRING", "VALUE", "AVG"],
"pad_method": ["STRING", "VALUE", "padding_const"],
"round_type": ["STRING", "VALUE", "floor"],
"pad":
   ["INTS",
    "CODE",
"[str(p) for p in self.array_layout(self.attr_pick(node['GlobalAveragePool'], 'pads', [ 0, 0]), [0, 1])]"
    ],
"padding": ["STRING", "VALUE", "VALID"],
"global_pooling": ["BOOL", "VALUE", True]}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "len(self.shape_pick(tensor['I:out0'])) == 3",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_global_avgpool1d)

r_global_maxpool = {
"ruler_name": "r_global_maxpool",
"src_ops_alias": ["GlobalMaxPool"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "GlobalMaxPool:in0"]],
"src_out_tensor": ["GlobalMaxPool:out0"],
"acu_lys_alias": ["pooling"],
"src_acu_in_tensor_map": [["I:out0", "pooling:in0"]],
"src_acu_out_tensor_map": [["GlobalMaxPool:out0", "pooling:out0"]],
"param_map": {
    "pooling":{
        "type": ["STRING", "VALUE", "MAX"],
        "pad_method": ["STRING", "VALUE", "padding_const"],
        "pad_h": ["INT", "CODE", "self.attr_pick(node['GlobalMaxPool'], 'pads', [0, 0, 0, 0])[0]"],
        "round_type": ["STRING", "VALUE", "floor"],
        "pad": ["INTS", "CODE",
"[str(p) for p in self.array_layout(self.attr_pick(node['GlobalMaxPool'], 'pads', [ 0, 0, 0, 0]), [0, 2, 1, 3])]"],
        "padding": ["STRING", "VALUE", "VALID"],
        "global_pooling": ["BOOL", "VALUE", True],
        "pad_w": ["INT", "CODE", "self.attr_pick(node['GlobalMaxPool'], 'pads', [0, 0, 0, 0])[1]"]
}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "len(self.shape_pick(tensor['I:out0'])) == 4",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_global_maxpool)

r_global_maxpool1d = {
"ruler_name": "r_global_maxpool1d",
"src_ops_alias": ["GlobalMaxPool"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "GlobalMaxPool:in0"]],
"src_out_tensor": ["GlobalMaxPool:out0"],
"acu_lys_alias": ["pool1d"],
"src_acu_in_tensor_map": [["I:out0", "pool1d:in0"]],
"src_acu_out_tensor_map": [["GlobalMaxPool:out0", "pool1d:out0"]],
"param_map": {
    "pool1d":{
        "type": ["STRING", "VALUE", "MAX"],
        "pad_method": ["STRING", "VALUE", "padding_const"],
        "round_type": ["STRING", "VALUE", "floor"],
        "pad": ["INTS", "CODE",
"[str(p) for p in self.array_layout(self.attr_pick(node['GlobalMaxPool'], 'pads', [ 0, 0]), [0, 1])]"],
        "padding": ["STRING", "VALUE", "VALID"],
        "global_pooling": ["BOOL", "VALUE", True],
}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "len(self.shape_pick(tensor['I:out0'])) == 3",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_global_maxpool1d)

r_maxpool1d = {
"ruler_name": "r_maxpool1d",
"src_ops_alias": ["MaxPool"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "MaxPool:in0"]],
"src_out_tensor": ["MaxPool:out0"],
"acu_lys_alias": ["pool1d"],
"src_acu_in_tensor_map": [["I:out0", "pool1d:in0"]],
"src_acu_out_tensor_map": [["MaxPool:out0", "pool1d:out0"]],
"param_map": {
    "pool1d":{
        "type": ["STRING", "VALUE", "MAX"],
        "pad_method": ["STRING", "VALUE", "padding_const"],
        "round_type": ["STRING", "CODE", "'ceil' if self.attr_pick(node['MaxPool'], 'ceil_mode') == 1 else 'floor'"],
        "pad": ["INTS", "CODE",
"[str(p) for p in self.array_layout(self.attr_pick(node['MaxPool'], 'pads', [ 0, 0]), [0, 1])]"],
        "global_pooling": ["BOOL", "VALUE", False],
        "ksize": ["INT", "CODE", "self.attr_pick(node['MaxPool'], 'kernel_shape', [1])[0]"],
        "stride": ["INT", "CODE", "self.attr_pick(node['MaxPool'], 'strides', [1])[0]"],
        "padding": [
            "STRING",
            "CODE",
         "'SAME' if self.attr_pick(node['MaxPool'], 'auto_pad', 'NOTSET') in ['SAME_UPPER', 'SAME_LOWER'] else 'VALID'"
        ],
}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "len(self.shape_pick(tensor['I:out0'])) == 3",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_maxpool1d)

r_rsp_v1 = {
"ruler_name": "r_rsp_v1",
"src_ops_alias": ["Reshape"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Reshape:in0"]],
"src_out_tensor": ["Reshape:out0"],
"acu_lys_alias": ["reshape"],
"src_acu_in_tensor_map": [["I:out0", "reshape:in0"]],
"src_acu_out_tensor_map": [["Reshape:out0", "reshape:out0"]],
"param_map": {"reshape":
                  {"shape":
                       ["STRING", "CODE", "self.shape_pick(tensor['Reshape:out0'])"]
                   }
              },
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, 4]}
ruler_list.append(r_rsp_v1)

r_rsp_v5 = {
"ruler_name": "r_rsp_v5",
"src_ops_alias": ["Reshape", "Constant_0"],
"src_inter_flow": [["Constant_0:out0", "Reshape:in1"]],
"src_in_anchor": [["I:out0", "Reshape:in0"]],
"src_out_tensor": ["Reshape:out0"],
"acu_lys_alias": ["reshape"],
"src_acu_in_tensor_map": [["I:out0", "reshape:in0"]],
"src_acu_out_tensor_map": [["Reshape:out0", "reshape:out0"]],
"param_map": {"reshape": {"shape": ["INTS", "CODE", "self.tensor_to_numpy(tensor['Constant_0:out0'])"]}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [5, -1]}
ruler_list.append(r_rsp_v5)

r_rsp_v5x = {
"ruler_name": "r_rsp_v5x",
"src_ops_alias": ["Reshape", "Constant_0", "Constant_1"],
"src_inter_flow": [["Constant_0:out0", "Reshape:in0"], ["Constant_1:out0", "Reshape:in1"]],
"src_in_anchor": [],
"src_out_tensor": ["Reshape:out0"],
"acu_lys_alias": ["variable"],
"src_acu_in_tensor_map": [],
"src_acu_out_tensor_map": [["Reshape:out0", "variable:out0"]],
"param_map": {"variable": {"shape": ['ORIGIN', 'CODE', "self.shape_pick(tensor['Reshape:out0'])"]}},
"blob_map": {"variable": {'data':
                              ['CODE',
                               "np.reshape("\
                               "self.tensor_to_numpy(tensor['Constant_0:out0']), "\
                               "self.tensor_to_numpy(tensor['Constant_1:out0']).astype(np.int32).tolist())"],}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [5, -1]}
ruler_list.append(r_rsp_v5x)

r_dynamic_rsp_5x = {
"ruler_name": "r_dynamic_rsp_5x",
"src_ops_alias": ["Reshape"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Reshape:in0"], ["I_1:out0", "Reshape:in1"]],
"src_out_tensor": ["Reshape:out0"],
"acu_lys_alias": ["reshape"],
"src_acu_in_tensor_map": [["I:out0", "reshape:in0"]],
"src_acu_out_tensor_map": [["Reshape:out0", "reshape:out0"]],
"param_map": {"reshape": {"shape": ["INTS", "CODE", "self.tensor_to_numpy(tensor['I_1:out0'])"]}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [5, -1]}
ruler_list.append(r_dynamic_rsp_5x)

r_squeeze_with_constant = {
"ruler_name": "r_squeeze_with_constant",
"src_ops_alias": ["Squeeze", "Constant_0"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Squeeze:in0"]],
"src_out_tensor": ["Squeeze:out0"],
"acu_lys_alias": ["reshape"],
"src_acu_in_tensor_map": [["I:out0", "reshape:in0"]],
"src_acu_out_tensor_map": [["Squeeze:out0", "reshape:out0"]],
"param_map":
{"reshape":
 {"shape":
  ["INTS",
   "CODE",
   "self.squeeze_shapes(self.attr_pick(node['Squeeze'], 'axes', None), self.shape_pick(tensor['Constant_0:out0']))"
   ]
  }
 },
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_squeeze_with_constant)

r_squeeze = {
"ruler_name": "r_squeeze",
"src_ops_alias": ["Squeeze"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "Squeeze:in0"]],
"src_out_tensor": ["Squeeze:out0"],
"acu_lys_alias": ["squeeze"],
"src_acu_in_tensor_map": [["I_0:out0", "squeeze:in0"]],
"src_acu_out_tensor_map": [["Squeeze:out0", "squeeze:out0"]],
"acu_inter_flow": [],
"param_map":
{"squeeze":
 {"axis_list": ["ORIGIN", "CODE", "self.attr_pick(node['Squeeze'], 'axes', None)"],
  }
 },
"blob_map": {},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_squeeze)

r_squeeze_with_constant_axes = {
"ruler_name": "r_squeeze_with_constant_axes",
"src_ops_alias": ["Squeeze", "Constant_0"],
"src_inter_flow": [["Constant_0:out0", "Squeeze:in1"]],
"src_in_anchor": [["I_0:out0", "Squeeze:in0"]],
"src_out_tensor": ["Squeeze:out0"],
"acu_lys_alias": ["squeeze"],
"src_acu_in_tensor_map": [["I_0:out0", "squeeze:in0"]],
"src_acu_out_tensor_map": [["Squeeze:out0", "squeeze:out0"]],
"param_map":
{"squeeze":
 {"axis_list": ["ORIGIN", "CODE", "self.tensor_to_numpy(tensor['Constant_0:out0']).tolist()"],
  }
 },
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [13, -1]}
ruler_list.append(r_squeeze_with_constant_axes)

r_unsqueeze = {
"ruler_name": "r_unsqueeze",
"src_ops_alias": ["Unsqueeze"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "Unsqueeze:in0"]],
"src_out_tensor": ["Unsqueeze:out0"],
"acu_lys_alias": ["reshape"],
"src_acu_in_tensor_map": [["I_0:out0", "reshape:in0"]],
"src_acu_out_tensor_map": [["Unsqueeze:out0", "reshape:out0"]],
"acu_inter_flow": [],
"param_map":
{"reshape":
 {"shape":
  ["INTS",
   "CODE",
   "self.unsqueeze_shape(self.attr_pick(node['Unsqueeze'], 'axes'), tensor['I_0:out0'])"
   ]
  }
 },
"blob_map": {},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_unsqueeze)

r_unsqueeze_with_constant_axes = {
"ruler_name": "r_unsqueeze_with_constant_axes",
"src_ops_alias": ["Unsqueeze", "Constant_0"],
"src_inter_flow": [["Constant_0:out0", "Unsqueeze:in1"]],
"src_in_anchor": [["I_0:out0", "Unsqueeze:in0"]],
"src_out_tensor": ["Unsqueeze:out0"],
"acu_lys_alias": ["reshape"],
"src_acu_in_tensor_map": [["I_0:out0", "reshape:in0"]],
"src_acu_out_tensor_map": [["Unsqueeze:out0", "reshape:out0"]],
"acu_inter_flow": [],
"param_map":
{"reshape":
 {"shape":
  ["INTS",
   "CODE",
   "self.unsqueeze_shape(self.tensor_to_numpy(tensor['Constant_0:out0']).tolist(), "
   "tensor['I_0:out0'])"
   ]
  }
 },
"blob_map": {},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [13, -1]}
ruler_list.append(r_unsqueeze_with_constant_axes)

r_flatten = {
"ruler_name": "r_flatten",
"src_ops_alias": ["Flatten"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Flatten:in0"]],
"src_out_tensor": ["Flatten:out0"],
"acu_lys_alias": ["reshape"],
"src_acu_in_tensor_map": [["I:out0", "reshape:in0"]],
"src_acu_out_tensor_map": [["Flatten:out0", "reshape:out0"]],
"param_map": {
    "reshape": {
        "shape": ["INTS", "CODE", "self.flatten_calc_shape(node['Flatten'], tensor['I:out0'])"]
    }
},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_flatten)

r_transpose = {
"ruler_name": "r_transpose",
"src_ops_alias": ["Transpose"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Transpose:in0"]],
"src_out_tensor": ["Transpose:out0"],
"acu_lys_alias": ["permute"],
"src_acu_in_tensor_map": [["I:out0", "permute:in0"]],
"src_acu_out_tensor_map": [["Transpose:out0", "permute:out0"]],
"param_map":
    {"permute":
         {"perm": ["STRING", "PYFUNC", r_permute_value()]}
     },
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_transpose)

r_softmax = {
"ruler_name": "r_softmax",
"src_ops_alias": ["Softmax"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Softmax:in0"]],
"src_out_tensor": ["Softmax:out0"],
"acu_lys_alias": ["softmax"],
"src_acu_in_tensor_map": [["I:out0", "softmax:in0"]],
"src_acu_out_tensor_map": [["Softmax:out0", "softmax:out0"]],
"param_map": {
"softmax": {"sf_axis": ['INT', 'PYFUNC', r_softmax_get_sf_axis()]}
},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": r_softmax_pre_cond(),
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_softmax)

r_log_softmax = {
"ruler_name": "r_log_softmax",
"src_ops_alias": ["LogSoftmax"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "LogSoftmax:in0"]],
"src_out_tensor": ["LogSoftmax:out0"],
"acu_lys_alias": ["log_softmax"],
"src_acu_in_tensor_map": [["I:out0", "log_softmax:in0"]],
"src_acu_out_tensor_map": [["LogSoftmax:out0", "log_softmax:out0"]],
"acu_inter_flow": [],
"param_map": {
"log_softmax": {"sf_axis": ['INT', 'PYFUNC', r_softmax_get_log_sf_axis()]}
},
"blob_map": {"log_softmax": {}},
"priority_tip": 0,
"pre_condition": r_logsoftmax_pre_cond(),
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_log_softmax)

r_softsign = {
"ruler_name": "r_softsign",
"src_ops_alias": ["Softsign"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "Softsign:in0"]],
"src_out_tensor": ["Softsign:out0"],
"acu_lys_alias": ["abs", "add", "divide", "variable"],
"src_acu_in_tensor_map": [["I_0:out0", "abs:in0"], ["I_0:out0", "divide:in0"]],
"src_acu_out_tensor_map": [["Softsign:out0", "divide:out0"]],
"acu_inter_flow": [["variable:out0", "add:in0"], ["abs:out0", "add:in1"], ["add:out0", "divide:in1"]],
"param_map": {},
"blob_map": {"variable": {'data': ['CODE', "np.ones([1], dtype=np.float32)"]}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_softsign)

r_dropout_out2_as_dropout_1_6 = {
"ruler_name": 'dropout_out2_as_dropout_1_6',
"src_ops_alias": ["Dropout"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "Dropout:in0"]],
"src_out_tensor": ["Dropout:out0", "Dropout:out1"],
"acu_lys_alias": ["dropout"],
"src_acu_in_tensor_map": [["I_0:out0", "dropout:in0"]],
"src_acu_out_tensor_map": [["Dropout:out0", "dropout:out0"]],
"acu_inter_flow": [],
"param_map": {'dropout':{'ratio':['FLOAT', 'CODE', "self.attr_pick(node['Dropout'], 'ratio', 0.5)"]}},
"blob_map": {"dropout": {}},
"priority_tip": 0,
"pre_condition": "self.attr_pick(node['Dropout'], 'is_test', 0) == 0 and "\
                 "math.isclose(0.0, self.attr_pick(node['Dropout'], 'ratio', 0.5)) == False",
"src_ops_main_version": None,
"src_ops_minior_version": [1, 6]}
ruler_list.append(r_dropout_out2_as_dropout_1_6)

r_dropout_out2_as_noop_1_6 = {
"ruler_name": 'dropout_out2_as_noop_1_6',
"src_ops_alias": ["Dropout"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "Dropout:in0"]],
"src_out_tensor": ["Dropout:out0", "Dropout:out1"],
"acu_lys_alias": ["noop"],
"src_acu_in_tensor_map": [["I_0:out0", "noop:in0"]],
"src_acu_out_tensor_map": [["Dropout:out0", "noop:out0"]],
"acu_inter_flow": [],
"param_map": {"noop": {}},
"blob_map": {"noop": {}},
"priority_tip": 0,
"pre_condition": "self.attr_pick(node['Dropout'], 'is_test', 0) != 0 or "\
                 "math.isclose(0.0, self.attr_pick(node['Dropout'], 'ratio', 0.5)) == True",
"src_ops_main_version": None,
"src_ops_minior_version": [1, 6]}
ruler_list.append(r_dropout_out2_as_noop_1_6)

r_dropout_out1_as_dropout_1_6 = {
"ruler_name": "r_dropout_out1_as_dropout_1_6",
"src_ops_alias": ["Dropout"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Dropout:in0"]],
"src_out_tensor": ["Dropout:out0"],
"acu_lys_alias": ["dropout"],
"src_acu_in_tensor_map": [["I:out0", "dropout:in0"]],
"src_acu_out_tensor_map": [["Dropout:out0", "dropout:out0"]],
"param_map": {'dropout':{'ratio':['FLOAT', 'CODE', "self.attr_pick(node['Dropout'], 'ratio', 0.5)"]}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "self.attr_pick(node['Dropout'], 'is_test', 0) == 0 and "\
                 "math.isclose(0.0, self.attr_pick(node['Dropout'], 'ratio', 0.5)) == False",
"src_ops_main_version": None,
"src_ops_minior_version": [1, 6]}
ruler_list.append(r_dropout_out1_as_dropout_1_6)

r_dropout_out1_as_noop_1_6 = {
"ruler_name": "r_dropout_out1_as_noop_1_6",
"src_ops_alias": ["Dropout"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Dropout:in0"]],
"src_out_tensor": ["Dropout:out0"],
"acu_lys_alias": ["noop"],
"src_acu_in_tensor_map": [["I:out0", "noop:in0"]],
"src_acu_out_tensor_map": [["Dropout:out0", "noop:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "self.attr_pick(node['Dropout'], 'is_test', 0) != 0 or "\
                 "math.isclose(0.0, self.attr_pick(node['Dropout'], 'ratio', 0.5)) == True",
"src_ops_main_version": None,
"src_ops_minior_version": [1, 6]}
ruler_list.append(r_dropout_out1_as_noop_1_6)

r_dropout_out2_as_dropout_7_10 = {
"ruler_name": 'dropout_out2_as_dropout_7_10',
"src_ops_alias": ["Dropout"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "Dropout:in0"]],
"src_out_tensor": ["Dropout:out0", "Dropout:out1"],
"acu_lys_alias": ["dropout"],
"src_acu_in_tensor_map": [["I_0:out0", "dropout:in0"]],
"src_acu_out_tensor_map": [["Dropout:out0", "dropout:out0"]],
"acu_inter_flow": [],
"param_map": {
    'dropout':{
        'ratio':['FLOAT', 'CODE', "self.attr_pick(node['Dropout'], 'ratio', 0.5)"],
        'scale_train':['BOOL', 'VALUE', "True"],
    }
},
"blob_map": {"dropout": {}},
"priority_tip": 0,
"pre_condition": "math.isclose(0.0, self.attr_pick(node['Dropout'], 'ratio', 0.5)) == False",
"src_ops_main_version": None,
"src_ops_minior_version": [7, 10]}
ruler_list.append(r_dropout_out2_as_dropout_7_10)

r_dropout_out2_as_noop_7_10 = {
"ruler_name": 'dropout_out2_as_noop_7_10',
"src_ops_alias": ["Dropout"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "Dropout:in0"]],
"src_out_tensor": ["Dropout:out0", "Dropout:out1"],
"acu_lys_alias": ["noop"],
"src_acu_in_tensor_map": [["I_0:out0", "noop:in0"]],
"src_acu_out_tensor_map": [["Dropout:out0", "noop:out0"]],
"acu_inter_flow": [],
"param_map": {
    'dropout':{
        'ratio':['FLOAT', 'CODE', "self.attr_pick(node['Dropout'], 'ratio', 0.5)"],
        'scale_train':['BOOL', 'VALUE', "True"],
    }
},
"blob_map": {"dropout": {}},
"priority_tip": 0,
"pre_condition": "math.isclose(0.0, self.attr_pick(node['Dropout'], 'ratio', 0.5)) == True",
"src_ops_main_version": None,
"src_ops_minior_version": [7, 10]}
ruler_list.append(r_dropout_out2_as_noop_7_10)

r_dropout_out1_as_dropout_7_10 = {
"ruler_name": "r_dropout_out1_as_dropout_7_10",
"src_ops_alias": ["Dropout"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Dropout:in0"]],
"src_out_tensor": ["Dropout:out0"],
"acu_lys_alias": ["dropout"],
"src_acu_in_tensor_map": [["I:out0", "dropout:in0"]],
"src_acu_out_tensor_map": [["Dropout:out0", "dropout:out0"]],
"param_map": {
    'dropout':{
        'ratio':['FLOAT', 'CODE', "self.attr_pick(node['Dropout'], 'ratio', 0.5)"],
        'scale_train':['BOOL', 'VALUE', "True"],
    }
},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "math.isclose(0.0, self.attr_pick(node['Dropout'], 'ratio', 0.5)) == False",
"src_ops_main_version": None,
"src_ops_minior_version": [7, 10]}
ruler_list.append(r_dropout_out1_as_dropout_7_10)

r_dropout_out1_as_noop_7_10 = {
"ruler_name": "r_dropout_out1_as_noop_7_10",
"src_ops_alias": ["Dropout"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Dropout:in0"]],
"src_out_tensor": ["Dropout:out0"],
"acu_lys_alias": ["noop"],
"src_acu_in_tensor_map": [["I:out0", "noop:in0"]],
"src_acu_out_tensor_map": [["Dropout:out0", "noop:out0"]],
"param_map": {
    'dropout':{
        'ratio':['FLOAT', 'CODE', "self.attr_pick(node['Dropout'], 'ratio', 0.5)"],
        'scale_train':['BOOL', 'VALUE', "True"],
    }
},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "math.isclose(0.0, self.attr_pick(node['Dropout'], 'ratio', 0.5)) == True",
"src_ops_main_version": None,
"src_ops_minior_version": [7, 10]}
ruler_list.append(r_dropout_out1_as_noop_7_10)

r_lrn = {
"ruler_name": "r_lrn",
"src_ops_alias": ["LRN"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "LRN:in0"]],
"src_out_tensor": ["LRN:out0"],
"acu_lys_alias": ["localresponsenormalization"],
"src_acu_in_tensor_map": [["I:out0", "localresponsenormalization:in0"]],
"src_acu_out_tensor_map": [["LRN:out0", "localresponsenormalization:out0"]],
"param_map":
    {"localresponsenormalization":
         {"beta": ["FLOAT", "CODE", "self.attr_pick(node['LRN'], 'beta', 0.75)"],
          "type": ["STRING", "VALUE", "NORM_ACROSS_CHANNELS"],
          "local_size": ["INT", "CODE", "self.attr_pick(node['LRN'], 'size')"],
          "alpha": ["FLOAT", "CODE", "self.attr_pick(node['LRN'], 'alpha', 1e-4)"],
          "bias": ["FLOAT", "CODE", "self.attr_pick(node['LRN'], 'bias', 1.0)"]}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_lrn)

r_reducel2_div_2_l2normalize = {
"ruler_name": "r_reducel2_div_2_l2normalize",
"src_ops_alias": ["Div", "ReduceL2"],
"src_inter_flow": [["ReduceL2:out0", "Div:in1"]],
"src_in_anchor": [["I_0:out0", "Div:in0"], ["I_0:out0", "ReduceL2:in0"]],
"src_out_tensor": ["Div:out0"],
"acu_lys_alias": ["l2normalize"],
"src_acu_in_tensor_map": [["I_0:out0", "l2normalize:in0"]],
"src_acu_out_tensor_map": [["Div:out0", "l2normalize:out0"]],
"acu_inter_flow": [],
"param_map": {
    "l2normalize":
        {'l2n_dim': ['INTS','CODE', "self.attr_pick(node['ReduceL2'], 'axes')"]}},
"blob_map": {"l2normalize": {}},
"priority_tip": 0,
"pre_condition": "self.attr_pick(node['ReduceL2'], 'keepdims') == 1",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_reducel2_div_2_l2normalize)

r_l2normalaize_scale = {
"ruler_name": "l2normalaize_scale",
"src_ops_alias": ["Mul", "Div", "Reshape", "Add", "Constant", "Constant_1",
                  "Sqrt", "Constant_2", "ReduceSum", "Pow", "Constant_3"],
"src_inter_flow": [["Div:out0", "Mul:in0"], ["Reshape:out0", "Mul:in1"],
                   ["Add:out0", "Div:in1"], ["Constant:out0", "Reshape:in0"], ["Constant_1:out0", "Reshape:in1"],
                   ["Sqrt:out0", "Add:in0"], ["Constant_2:out0", "Add:in1"], ["ReduceSum:out0", "Sqrt:in0"],
                   ["Pow:out0", "ReduceSum:in0"], ["Constant_3:out0", "Pow:in1"]],
"src_in_anchor": [["I_0:out0", "Div:in0"], ["I_0:out0", "Pow:in0"]],
"src_out_tensor": ["Mul:out0"],
"acu_lys_alias": ["l2normalizescale"],
"src_acu_in_tensor_map": [["I_0:out0", "l2normalizescale:in0"]],
"src_acu_out_tensor_map": [["Mul:out0", "l2normalizescale:out0"]],
"acu_inter_flow": [],
"param_map": {"l2normalizescale": {'l2n_dim': ['ORIGIN','VALUE', -1]}},
"blob_map":
    {"l2normalizescale":
         {'scale':
              ['CODE',
               "np.reshape("\
               "self.tensor_to_numpy(tensor['Constant:out0']), "\
               "self.array_layout("\
               "self.tensor_to_numpy(tensor['Constant_1:out0']).astype(np.int32).tolist(),"\
               " [0, 2, 3, 1]))"],}},
"priority_tip": 0,
"pre_condition": "(self.tensor_to_numpy(tensor['Constant_3:out0']) == 2.0).all() and "\
                 " self.attr_pick(node['ReduceSum'], 'axes') == [1]",
"src_ops_main_version": None,
"src_ops_minior_version": [5, -1]}
ruler_list.append(r_l2normalaize_scale)

r_l2normalaize = {
"ruler_name": "r_l2normalaize",
"src_ops_alias": ["Div", "Add", "Reshape", "Constant", "Sqrt", "Constant_1", "ReduceSum", "Pow", "Constant_2"],
"src_inter_flow": [["Add:out0", "Div:in1"], ["Reshape:out0", "Add:in0"], ["Constant:out0", "Add:in1"],
                   ["Sqrt:out0", "Reshape:in0"], ["Constant_1:out0", "Reshape:in1"], ["ReduceSum:out0", "Sqrt:in0"],
                   ["Pow:out0", "ReduceSum:in0"], ["Constant_2:out0", "Pow:in1"]],
"src_in_anchor": [["I_0:out0", "Div:in0"], ["I_0:out0", "Pow:in0"]],
"src_out_tensor": ["Div:out0"],
"acu_lys_alias": ["l2normalize"],
"src_acu_in_tensor_map": [["I_0:out0", "l2normalize:in0"]],
"src_acu_out_tensor_map": [["Div:out0", "l2normalize:out0"]],
"acu_inter_flow": [],
"param_map": {
    "l2normalize": {
        'l2n_dim': ['INTS', 'CODE', "self.attr_pick(node['ReduceSum'], 'axes')"],
        'eps': ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['Constant:out0'])"]
    }
},
"blob_map": {"l2normalize": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_l2normalaize)

r_l2normalaize_1 = {
"ruler_name": "r_l2normalaize_1",
"src_ops_alias": ["LpNormalization"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "LpNormalization:in0"]],
"src_out_tensor": ["LpNormalization:out0"],
"acu_lys_alias": ["l2normalize"],
"src_acu_in_tensor_map": [["I_0:out0", "l2normalize:in0"]],
"src_acu_out_tensor_map": [["LpNormalization:out0", "l2normalize:out0"]],
"acu_inter_flow": [],
"param_map": {"l2normalize": {
        'l2n_dim': ['INT', 'CODE', "self.attr_pick(node['LpNormalization'], 'axis')"],}},
"blob_map": {"l2normalize": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_l2normalaize_1)

r_instancenorm = {
"ruler_name": "r_instancenorm",
"src_ops_alias": ["InstanceNormalization", "Constant_0", "Constant_1"],
"src_inter_flow": [["Constant_0:out0", "InstanceNormalization:in1"],
                   ["Constant_1:out0", "InstanceNormalization:in2"]],
"src_in_anchor": [["I:out0", "InstanceNormalization:in0"]],
"src_out_tensor": ["InstanceNormalization:out0"],
"acu_lys_alias": ["instancenormalize"],
"src_acu_in_tensor_map": [["I:out0", "instancenormalize:in0"]],
"src_acu_out_tensor_map": [["InstanceNormalization:out0", "instancenormalize:out0"]],
"param_map":{
    "instancenormalize":{
        'eps': ['FLOAT', 'CODE', "self.attr_pick(node['InstanceNormalization'], 'epsilon', 1e-5)"],
        'axis':['INTS', 'CODE', "list(range(2, len(self.attr_pick(node['InstanceNormalization'], '_out_shape')[0])))"],
    }
},
"blob_map": {"instancenormalize": {'bias': ['CODE', "self.tensor_to_numpy(tensor['Constant_1:out0'])"],
                                'scale': ['CODE', "self.tensor_to_numpy(tensor['Constant_0:out0'])"],}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [6, -1]}
ruler_list.append(r_instancenorm)

r_add = {
"ruler_name": "r_add",
"src_ops_alias": ["Add"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Add:in0"], ['I_1:out0', "Add:in1"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["add"],
"src_acu_in_tensor_map":[["I:out0", "add:in0"], ['I_1:out0', "add:in1"]],
"src_acu_out_tensor_map": [["Add:out0", "add:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_add)

r_add_1 = {
"ruler_name": "r_add_1",
"src_ops_alias": ["Add"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Add:in0"], ['I:out0', "Add:in1"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["add"],
"src_acu_in_tensor_map":[["I:out0", "add:in0"], ['I:out0', "add:in1"]],
"src_acu_out_tensor_map": [["Add:out0", "add:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_add_1)

r_sub = {
"ruler_name": "r_sub",
"src_ops_alias": ["Sub"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Sub:in0"], ['I_1:out0', "Sub:in1"]],
"src_out_tensor": ["Sub:out0"],
"acu_lys_alias": ["subtract"],
"src_acu_in_tensor_map":[["I:out0", "subtract:in0"], ['I_1:out0', "subtract:in1"]],
"src_acu_out_tensor_map": [["Sub:out0", "subtract:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_sub)

r_mul = {
"ruler_name": "r_mul",
"src_ops_alias": ["Mul"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Mul:in0"], ['I_1:out0', "Mul:in1"]],
"src_out_tensor": ["Mul:out0"],
"acu_lys_alias": ["multiply"],
"src_acu_in_tensor_map":[["I:out0", "multiply:in0"], ['I_1:out0', "multiply:in1"]],
"src_acu_out_tensor_map": [["Mul:out0", "multiply:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_mul)

r_mul_with_same_input = {
"ruler_name": "r_mul_with_same_input",
"src_ops_alias": ["Mul"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Mul:in0"], ['I:out0', "Mul:in1"]],
"src_out_tensor": ["Mul:out0"],
"acu_lys_alias": ["multiply"],
"src_acu_in_tensor_map":[["I:out0", "multiply:in0"], ['I:out0', "multiply:in1"]],
"src_acu_out_tensor_map": [["Mul:out0", "multiply:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_mul_with_same_input)

r_selu = {
"ruler_name": "r_selu",
"src_ops_alias": ["Selu"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "Selu:in0"]],
"src_out_tensor": ["Selu:out0"],
"acu_lys_alias": ["selu"],
"src_acu_in_tensor_map": [["I_0:out0", "selu:in0"]],
"src_acu_out_tensor_map": [["Selu:out0", "selu:out0"]],
"acu_inter_flow": [],
"param_map": {"selu": {}},
"blob_map": {"selu": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_selu)

r_div = {
"ruler_name": "r_div",
"src_ops_alias": ["Div"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Div:in0"], ['I_1:out0', "Div:in1"]],
"src_out_tensor": ["Div:out0"],
"acu_lys_alias": ["real_div"],
"src_acu_in_tensor_map":[["I:out0", "real_div:in0"], ['I_1:out0', "real_div:in1"]],
"src_acu_out_tensor_map": [["Div:out0", "real_div:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_div)

r_mod = {
"ruler_name": "r_mod",
"src_ops_alias": ["Mod"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Mod:in0"], ['I_1:out0', "Mod:in1"]],
"src_out_tensor": ["Mod:out0"],
"acu_lys_alias": ["mod"],
"src_acu_in_tensor_map":[["I:out0", "mod:in0"], ['I_1:out0', "mod:in1"]],
"src_acu_out_tensor_map": [["Mod:out0", "mod:out0"]],
"param_map": {"mod":{"fmod":['INT', 'CODE', "self.attr_pick(node['Mod'], 'fmod', 0)"]}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_mod)

r_logical_or = {
"ruler_name": "r_logical_or",
"src_ops_alias": ["Or"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Or:in0"], ['I_1:out0', "Or:in1"]],
"src_out_tensor": ["Or:out0"],
"acu_lys_alias": ["logical_or"],
"src_acu_in_tensor_map":[["I:out0", "logical_or:in0"], ['I_1:out0', "logical_or:in1"]],
"src_acu_out_tensor_map": [["Or:out0", "logical_or:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_logical_or)

r_logical_and = {
"ruler_name": "r_logical_and",
"src_ops_alias": ["And"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "And:in0"], ['I_1:out0', "And:in1"]],
"src_out_tensor": ["And:out0"],
"acu_lys_alias": ["logical_and"],
"src_acu_in_tensor_map":[["I:out0", "logical_and:in0"], ['I_1:out0', "logical_and:in1"]],
"src_acu_out_tensor_map": [["And:out0", "logical_and:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_logical_and)

r_greater = {
"ruler_name": "r_greater",
"src_ops_alias": ["Greater"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Greater:in0"], ['I_1:out0', "Greater:in1"]],
"src_out_tensor": ["Greater:out0"],
"acu_lys_alias": ["greater"],
"src_acu_in_tensor_map":[["I:out0", "greater:in0"], ['I_1:out0', "greater:in1"]],
"src_acu_out_tensor_map": [["Greater:out0", "greater:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_greater)

r_greater_equal = {
"ruler_name": "r_greater_equal",
"src_ops_alias": ["GreaterOrEqual"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "GreaterOrEqual:in0"], ['I_1:out0', "GreaterOrEqual:in1"]],
"src_out_tensor": ["GreaterOrEqual:out0"],
"acu_lys_alias": ["greater_equal"],
"src_acu_in_tensor_map":[["I:out0", "greater_equal:in0"], ['I_1:out0', "greater_equal:in1"]],
"src_acu_out_tensor_map": [["GreaterOrEqual:out0", "greater_equal:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [12, -1]}
ruler_list.append(r_greater_equal)

r_abs = {
"ruler_name": "r_abs",
"src_ops_alias": ["Abs"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Abs:in0"]],
"src_out_tensor": ["Abs:out0"],
"acu_lys_alias": ["abs"],
"src_acu_in_tensor_map": [["I:out0", "abs:in0"]],
"src_acu_out_tensor_map": [["Abs:out0", "abs:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_abs)

r_ceil = {
"ruler_name": "r_ceil",
"src_ops_alias": ["Ceil"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Ceil:in0"]],
"src_out_tensor": ["Ceil:out0"],
"acu_lys_alias": ["ceil"],
"src_acu_in_tensor_map": [["I:out0", "ceil:in0"]],
"src_acu_out_tensor_map": [["Ceil:out0", "ceil:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_ceil)

r_erf = {
"ruler_name": "r_erf",
"src_ops_alias": ["Erf"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Erf:in0"]],
"src_out_tensor": ["Erf:out0"],
"acu_lys_alias": ["erf"],
"src_acu_in_tensor_map": [["I:out0", "erf:in0"]],
"src_acu_out_tensor_map": [["Erf:out0", "erf:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_erf)

r_floor = {
"ruler_name": "r_floor",
"src_ops_alias": ["Floor"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Floor:in0"]],
"src_out_tensor": ["Floor:out0"],
"acu_lys_alias": ["floor"],
"src_acu_in_tensor_map": [["I:out0", "floor:in0"]],
"src_acu_out_tensor_map": [["Floor:out0", "floor:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_floor)

r_sqrt = {
"ruler_name": "r_sqrt",
"src_ops_alias": ["Sqrt"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Sqrt:in0"]],
"src_out_tensor": ["Sqrt:out0"],
"acu_lys_alias": ["sqrt"],
"src_acu_in_tensor_map": [["I:out0", "sqrt:in0"]],
"src_acu_out_tensor_map": [["Sqrt:out0", "sqrt:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_sqrt)

r_log = {
"ruler_name": "r_log",
"src_ops_alias": ["Log"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Log:in0"]],
"src_out_tensor": ["Log:out0"],
"acu_lys_alias": ["log"],
"src_acu_in_tensor_map": [["I:out0", "log:in0"]],
"src_acu_out_tensor_map": [["Log:out0", "log:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_log)

r_cast = {
"ruler_name": "r_cast",
"src_ops_alias": ["Cast"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Cast:in0"]],
"src_out_tensor": ["Cast:out0"],
"acu_lys_alias": ["cast"],
"src_acu_in_tensor_map": [["I:out0", "cast:in0"]],
"src_acu_out_tensor_map": [["Cast:out0", "cast:out0"]],
"param_map": {
    "cast": {
        'in_data_type':["STRING", "CODE", "self.cast_map_type(node['Cast'])"],
        'out_data_type':["STRING", "CODE", "self.cast_map_type(node['Cast'])"],
}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_cast)

r_cast_like = {
"ruler_name": "r_cast_like",
"src_ops_alias": ["CastLike"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "CastLike:in0"], ["I_1:out0", "CastLike:in1"]],
"src_out_tensor": ["CastLike:out0"],
"acu_lys_alias": ["cast"],
"src_acu_in_tensor_map": [["I:out0", "cast:in0"]],
"src_acu_out_tensor_map": [["CastLike:out0", "cast:out0"]],
"param_map": {
    "cast": {
        'in_data_type':["STRING", "CODE", "self.dtype_pick(tensor['I:out0'])"],
        'out_data_type':["STRING", "CODE", "self.dtype_pick(tensor['I_1:out0'])"],
}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [15, -1]}
ruler_list.append(r_cast_like)

r_exp = {
"ruler_name": "r_exp",
"src_ops_alias": ["Exp"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Exp:in0"]],
"src_out_tensor": ["Exp:out0"],
"acu_lys_alias": ["exp"],
"src_acu_in_tensor_map": [["I:out0", "exp:in0"]],
"src_acu_out_tensor_map": [["Exp:out0", "exp:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_exp)

r_rsp_tsp_rsp_2_shuffle_v1 = {
"ruler_name": "r_rsp_tsp_rsp_2_shuffle_v1",
"src_ops_alias": ["Reshape", "Transpose", "Reshape_1"],
"src_inter_flow": [["Reshape:out0", "Transpose:in0"], ["Transpose:out0", "Reshape_1:in0"]],
"src_in_anchor": [["I:out0", "Reshape:in0"]],
"src_out_tensor": ["Reshape_1:out0"],
"acu_lys_alias": ["shuffle"],
"src_acu_in_tensor_map": [["I:out0", "shuffle:in0"]],
"src_acu_out_tensor_map": [["Reshape_1:out0", "shuffle:out0"]],
"param_map": {"shuffle": {"group_number": ["INT", "CODE", "self.attr_pick(node['Reshape'], 'shape')[1]"]}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "len(self.attr_pick(node['Reshape'], 'shape')) == 5 and "\
    "len(self.attr_pick(node['Transpose'], 'perm')) == 5 and "\
    "len(self.attr_pick(node['Reshape_1'], 'shape')) == 4 and "\
    "self.attr_pick(node['Transpose'], 'perm') == [0, 2, 1, 3, 4]",
"src_ops_main_version": None,
"src_ops_minior_version": [1, 4]}
ruler_list.append(r_rsp_tsp_rsp_2_shuffle_v1)

r_rsp_tsp_rsp_2_shuffle_v5 = {
"ruler_name": "r_rsp_tsp_rsp_2_shuffle_v5",
"src_ops_alias": ["Reshape", "Transpose", "Reshape_1", "Constant_0"],
"src_inter_flow": [["Reshape:out0", "Transpose:in0"], ["Transpose:out0", "Reshape_1:in0"]],
"src_in_anchor": [["I:out0", "Reshape:in0"]],
"src_out_tensor": ["Reshape_1:out0"],
"acu_lys_alias": ["shuffle"],
"src_acu_in_tensor_map": [["I:out0", "shuffle:in0"]],
"src_acu_out_tensor_map": [["Reshape_1:out0", "shuffle:out0"]],
"param_map": {"shuffle": {"group_number": ["INT", "CODE", "self.tensor_to_numpy(tensor['Constant_0:out0'])[1]"]}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "len(self.tensor_to_numpy(tensor['Reshape'].inputs[1])) == 5 and "\
    "len(self.attr_pick(node['Transpose'], 'perm')) == 5 and "\
    "len(self.tensor_to_numpy(tensor['Reshape_1'].inputs[1])) == 4 and "\
    "self.attr_pick(node['Transpose'], 'perm') == [0, 2, 1, 3, 4]",
"src_ops_main_version": None,
"src_ops_minior_version": [5, -1]}
ruler_list.append(r_rsp_tsp_rsp_2_shuffle_v5)

r_identity = {
"ruler_name": "r_identity",
"src_ops_alias": ["Identity"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Identity:in0"]],
"src_out_tensor": ["Identity:out0"],
"acu_lys_alias": ["noop"],
"src_acu_in_tensor_map": [["I:out0", "noop:in0"]],
"src_acu_out_tensor_map": [["Identity:out0", "noop:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_identity)

r_constantfill = {
"ruler_name": "r_constantfill",
"src_ops_alias": ["ConstantFill"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "ConstantFill:in0"]],
"src_out_tensor": ["ConstantFill:out0"],
"acu_lys_alias": ["noop"],
"src_acu_in_tensor_map": [["I:out0", "noop:in0"]],
"src_acu_out_tensor_map": [["ConstantFill:out0", "noop:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_constantfill)

r_slice = {
"ruler_name": "r_slice",
"src_ops_alias": ["Slice"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Slice:in0"]],
"src_out_tensor": ["Slice:out0"],
"acu_lys_alias": ["slice"],
"src_acu_in_tensor_map": [["I:out0", "slice:in0"]],
"src_acu_out_tensor_map": [["Slice:out0", "slice:out0"]],
"param_map":
    {"slice":
         {"begin": ["INTS", "PYFUNC", r_slice_get_begin()],
          "size": ["INTS", "PYFUNC", r_slice_get_size()],
          }
     },
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "self.attr_pick(node['Slice'], 'axes', None) == None",
"src_ops_main_version": None,
"src_ops_minior_version": [1, 9]}
ruler_list.append(r_slice)

r_slice_axes = {
"ruler_name": "r_slice_axes",
"src_ops_alias": ["Slice"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Slice:in0"]],
"src_out_tensor": ["Slice:out0"],
"acu_lys_alias": ["slice"],
"src_acu_in_tensor_map": [["I:out0", "slice:in0"]],
"src_acu_out_tensor_map": [["Slice:out0", "slice:out0"]],
"param_map":
    {"slice":
         {"begin": ["INTS", "PYFUNC", r_slice_get_begin()],
          "size": ["INTS", "PYFUNC", r_slice_get_size()],
          }
     },
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "self.attr_pick(node['Slice'], 'axes', None) != None",
"src_ops_main_version": None,
"src_ops_minior_version": [1, 9]}
ruler_list.append(r_slice_axes)

r_slice_ex = {
"ruler_name": "r_slice_ex",
"src_ops_alias": ["Slice", "Constant", "Constant_1"],
"src_inter_flow": [["Constant:out0", "Slice:in1"], ["Constant_1:out0", "Slice:in2"]],
"src_in_anchor": [["I_0:out0", "Slice:in0"]],
"src_out_tensor": ["Slice:out0"],
"acu_lys_alias": ["slice"],
"src_acu_in_tensor_map": [["I_0:out0", "slice:in0"]],
"src_acu_out_tensor_map": [["Slice:out0", "slice:out0"]],
"acu_inter_flow": [],
"param_map": {"slice": {
    'begin':["INTS", "CODE",
             "self.slice_ex_begin(node['Slice'], self.shape_pick(tensor['I_0:out0']), tensor['Constant:out0'])"],
    'size':["INTS", "CODE",
            "self.slice_ex_size(node['Slice'], self.shape_pick(tensor['I_0:out0']), "
                                "tensor['Constant:out0'], tensor['Constant_1:out0'])"],
}},
"blob_map": {"slice": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [10, -1]}
ruler_list.append(r_slice_ex)

r_slice_2 = {
"ruler_name": 'r_slice_2',
"src_ops_alias": ["Slice", "Constant", "Constant_1", "Constant_2", "Constant_3"],
"src_inter_flow": [["Constant:out0", "Slice:in1"], ["Constant_1:out0", "Slice:in2"],
                   ["Constant_2:out0", "Slice:in3"],
                   ["Constant_3:out0", "Slice:in4"]],
"src_in_anchor": [["I_0:out0", "Slice:in0"]],
"src_out_tensor": ["Slice:out0"],
"acu_lys_alias": ["slice"],
"src_acu_in_tensor_map": [["I_0:out0", "slice:in0"]],
"src_acu_out_tensor_map": [["Slice:out0", "slice:out0"]],
"acu_inter_flow": [],
"param_map": {"slice": {'begin': ["INTS", "CODE",
                                  "self.slice_ex_begin(node['Slice'], self.shape_pick(tensor['I_0:out0']), "
                                  "tensor['Constant:out0'], tensor['Constant_2:out0'])"],
                        'size': ["INTS", "CODE",
                                 "self.slice_ex_size(node['Slice'], self.shape_pick(tensor['I_0:out0']), "
                                 "tensor['Constant:out0'], tensor['Constant_1:out0'], tensor['Constant_2:out0'])"]
                        }},
"blob_map": {"slice": {}},
"priority_tip": 0,
"pre_condition": r_slice_pre_cond(steps_tensor='Constant_3:out0'),
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_slice_2)

r_slice_3 = {
"ruler_name": 'r_slice_3',
"src_ops_alias": ["Slice", "Constant", "Constant_1", "Constant_2"],
"src_inter_flow": [["Constant:out0", "Slice:in1"], ["Constant_1:out0", "Slice:in2"],
                   ["Constant_2:out0", "Slice:in3"]],
"src_in_anchor": [["I_0:out0", "Slice:in0"]],
"src_out_tensor": ["Slice:out0"],
"acu_lys_alias": ["slice"],
"src_acu_in_tensor_map": [["I_0:out0", "slice:in0"]],
"src_acu_out_tensor_map": [["Slice:out0", "slice:out0"]],
"acu_inter_flow": [],
"param_map": {"slice": {'begin': ["INTS", "CODE",
                                  "self.slice_ex_begin(node['Slice'], self.shape_pick(tensor['I_0:out0']), "
                                  "tensor['Constant:out0'], tensor['Constant_2:out0'])"],
                        'size': ["INTS", "CODE",
                                 "self.slice_ex_size(node['Slice'], self.shape_pick(tensor['I_0:out0']), "
                                 "tensor['Constant:out0'], tensor['Constant_1:out0'], tensor['Constant_2:out0'])"]}},
"blob_map": {"slice": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_slice_3)

r_slice_to_stride_slice = {
"ruler_name": "r_slice_to_stride_slice",
"src_ops_alias": ["Slice", "Constant", "Constant_1", "Constant_2", "Constant_3"],
"src_inter_flow": [["Constant:out0", "Slice:in1"], ["Constant_1:out0", "Slice:in2"], ["Constant_2:out0", "Slice:in3"],
                   ["Constant_3:out0", "Slice:in4"]],
"src_in_anchor": [["I_0:out0", "Slice:in0"]],
"src_out_tensor": ["Slice:out0"],
"acu_lys_alias": ["stridedslice"],
"src_acu_in_tensor_map": [["I_0:out0", "stridedslice:in0"]],
"src_acu_out_tensor_map": [["Slice:out0", "stridedslice:out0"]],
"acu_inter_flow": [],
"param_map": {"stridedslice": {'slice_begin':
                                   ["INTS", "CODE",
                                    "self.slice_ex_begin(node['Slice'], self.shape_pick(tensor['I_0:out0']), "
                                    "tensor['Constant:out0'], tensor['Constant_2:out0'])"],
                               'slice_end':
                                   ["INTS", "CODE", "self.stride_slice_end(node['Slice'], self.shape_pick("
                                                    "tensor['I_0:out0']), tensor['Constant_1:out0'], "
                                                    "tensor['Constant_2:out0'])"],
                               'slice_strides': ["INTS", "CODE", "self.stride_slice_strides(node['Slice'], "
                                                 "self.shape_pick(tensor['I_0:out0']), "
                                                 "tensor['Constant_3:out0'], tensor['Constant_2:out0'])"
                                                 ],
                               'slice_end_mask': ["INT", "CODE", "self.slice_end_mask(node['Slice'], "
                                                                 "self.shape_pick(tensor['I_0:out0']), "
                                                                 "tensor['Constant_1:out0'], "
                                                                 "tensor['Constant_3:out0'], "
                                                                 "tensor['Constant_2:out0'])"
                                                 ],
                               }
              },
"blob_map": {"stridedslice": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_slice_to_stride_slice)

r_upsample_l7_to_resize = {
"ruler_name": "upsample_l7_to_resize",
"src_ops_alias": ["Upsample"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Upsample:in0"]],
"src_out_tensor": ["Upsample:out0"],
"acu_lys_alias": ["image_resize"],
"src_acu_in_tensor_map": [["I:out0", "image_resize:in0"]],
"src_acu_out_tensor_map": [["Upsample:out0", "image_resize:out0"]],
"param_map": {"image_resize":
                  {"new_size": ["ORIGIN", "CODE", "self.shape_pick(tensor['Upsample:out0'])[2:]"],
                    "align_corners": ["BOOL", "VALUE", False],
                   "type":
                       ['STRING',
                        'CODE',
                        "'bilinear' "
                        "if self.attr_pick(node['Upsample'], 'mode') == 'linear' "
                        "else self.attr_pick(node['Upsample'], 'mode')"],
                   }
              },
"blob_map": {"image_resize": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, 8]}
ruler_list.append(r_upsample_l7_to_resize)

r_upsample_l9_to_resize = {
"ruler_name": "upsample_l9_to_resize",
"src_ops_alias": ["Upsample", "Constant"],
"src_inter_flow": [["Constant:out0", "Upsample:in1"]],
"src_in_anchor": [["I:out0", "Upsample:in0"]],
"src_out_tensor": ["Upsample:out0"],
"acu_lys_alias": ["image_resize"],
"src_acu_in_tensor_map": [["I:out0", "image_resize:in0"]],
"src_acu_out_tensor_map": [["Upsample:out0", "image_resize:out0"]],
"param_map": {"image_resize":
                  {"new_size": ["ORIGIN", "CODE", "self.shape_pick(tensor['Upsample:out0'])[2:]"],
                    "align_corners": ["BOOL", "VALUE", False],
                    "type":
                       ['STRING',
                        'CODE',
                        "'bilinear' "
                        "if self.attr_pick(node['Upsample'], 'mode') == 'linear' "
                        "else self.attr_pick(node['Upsample'], 'mode')"],}
              },
"blob_map": {"image_resize": {}},
"acu_inter_flow": [],
"priority_tip": 11,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [9, -1]}
ruler_list.append(r_upsample_l9_to_resize)

r_upsample_l9_scale_to_resize = {
"ruler_name": "r_upsample_l9_scale_to_resize",
"src_ops_alias": ["Upsample", ],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Upsample:in0"], ["I_1:out0", "Upsample:in1"]],
"src_out_tensor": ["Upsample:out0"],
"acu_lys_alias": ["image_resize"],
"src_acu_in_tensor_map": [["I:out0", "image_resize:in0"]],
"src_acu_out_tensor_map": [["Upsample:out0", "image_resize:out0"]],
"param_map": {"image_resize":
                  {"new_size": ["ORIGIN", "CODE", "self.shape_pick(tensor['Upsample:out0'])[2:]"],
                    "align_corners": ["BOOL", "VALUE", False],
                    "type":
                       ['STRING',
                        'CODE',
                        "'bilinear' "
                        "if self.attr_pick(node['Upsample'], 'mode') == 'linear' "
                        "else self.attr_pick(node['Upsample'], 'mode')"],}
              },
"blob_map": {"image_resize": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [9, -1]}
ruler_list.append(r_upsample_l9_scale_to_resize)

r_resize_10 = {
"ruler_name": "r_resize_10",
"src_ops_alias": ["Resize", "Constant"],
"src_inter_flow": [["Constant:out0", "Resize:in1"]],
"src_in_anchor": [["I:out0", "Resize:in0"]],
"src_out_tensor": ["Resize:out0"],
"acu_lys_alias": ["image_resize"],
"src_acu_in_tensor_map": [["I:out0", "image_resize:in0"]],
"src_acu_out_tensor_map": [["Resize:out0", "image_resize:out0"]],
"acu_inter_flow": [],
"param_map":{
    "image_resize":{
        "new_size": ['INTS', 'PYFUNC', r_resize_get_new_size()],
        "align_corners": ['BOOL', 'VALUE', False],
        "half_pixel": ['BOOL', 'VALUE', False],
        "type": ['STRING', 'PYFUNC', r_resize_get_type()],
    }
},
"blob_map": {},
"priority_tip": 0,
"pre_condition": r_resize_10_check(),
"src_ops_main_version": None,
"src_ops_minior_version": [0, 10]
}
ruler_list.append(r_resize_10)

r_resize = {
"ruler_name": "r_resize",
"src_ops_alias": ["Resize", "Constant", "Constant_1"],
"src_inter_flow": [["Constant:out0", "Resize:in1"], ["Constant_1:out0", "Resize:in2"]],
"src_in_anchor": [["I:out0", "Resize:in0"]],
"src_out_tensor": ["Resize:out0"],
"acu_lys_alias": ["image_resize"],
"src_acu_in_tensor_map": [["I:out0", "image_resize:in0"]],
"src_acu_out_tensor_map": [["Resize:out0", "image_resize:out0"]],
"acu_inter_flow": [],
"param_map":{
    "image_resize":{
        "new_size": ['INTS', 'PYFUNC', r_resize_get_new_size()],
        "align_corners": ['BOOL', 'PYFUNC', r_resize_get_align_corners()],
        "half_pixel": ['BOOL', 'PYFUNC', r_resize_get_half_pixel()],
        "type": ['STRING', 'PYFUNC', r_resize_get_type()],
    }
},
"blob_map": {},
"priority_tip": 0,
"pre_condition": r_resize_check(),
"src_ops_main_version": None,
"src_ops_minior_version": [11, -1]
}
ruler_list.append(r_resize)

r_resize_size = {
"ruler_name": "r_resize_size",
"src_ops_alias": ["Resize", "Constant", "Constant_1", "Constant_2"],
"src_inter_flow": [["Constant:out0", "Resize:in1"],
                   ["Constant_1:out0", "Resize:in2"], ["Constant_2:out0", "Resize:in3"]],
"src_in_anchor": [["I:out0", "Resize:in0"]],
"src_out_tensor": ["Resize:out0"],
"acu_lys_alias": ["image_resize"],
"src_acu_in_tensor_map": [["I:out0", "image_resize:in0"]],
"src_acu_out_tensor_map": [["Resize:out0", "image_resize:out0"]],
"acu_inter_flow": [],
"param_map":{
    "image_resize":{
        "new_size": ['INTS', 'PYFUNC', r_resize_get_new_size()],
        "align_corners": ['BOOL', 'PYFUNC', r_resize_get_align_corners()],
        "half_pixel": ['BOOL', 'PYFUNC', r_resize_get_half_pixel()],
        "type": ['STRING', 'PYFUNC', r_resize_get_type()],
    }
},
"blob_map": {},
"priority_tip": 0,
"pre_condition": r_resize_check(),
"src_ops_main_version": None,
"src_ops_minior_version": [11, -1]
}
ruler_list.append(r_resize_size)

r_resize_i0_constant_frozentablei1= {
"ruler_name": "r_resize_i0_constant_frozentablei1",
"src_ops_alias": ["Resize", "Constant"],
"src_inter_flow": [["Constant:out0", "Resize:in1"],
                   ["Constant:out0", "Resize:in2"]],
"src_in_anchor": [["I:out0", "Resize:in0"],["I_1:out0", "Resize:in3"]],
"src_out_tensor": ["Resize:out0"],
"acu_lys_alias": ["image_resize"],
"src_acu_in_tensor_map": [["I:out0", "image_resize:in0"]],
"src_acu_out_tensor_map": [["Resize:out0", "image_resize:out0"]],
"acu_inter_flow": [],
"param_map":{
    "image_resize":{
        "new_size": ['INTS', 'PYFUNC', r_resize_get_new_size()],
        "align_corners": ['BOOL', 'PYFUNC', r_resize_get_align_corners()],
        "half_pixel": ['BOOL', 'PYFUNC', r_resize_get_half_pixel()],
        "type": ['STRING', 'PYFUNC', r_resize_get_type()],
    }
},
"blob_map": {},
"priority_tip": 0,
"pre_condition": r_resize_check(),
"src_ops_main_version": None,
"src_ops_minior_version": [11, -1]
}
ruler_list.append(r_resize_i0_constant_frozentablei1)

r_resize_i0_constant0_constant1= {
"ruler_name": "r_resize_i0_constant0_constant1",
"src_ops_alias": ["Resize", "Constant", "Constant_1"],
"src_inter_flow": [["Constant:out0", "Resize:in1"], ["Constant:out0", "Resize:in2"],
                   ["Constant_1:out0", "Resize:in3"]],
"src_in_anchor": [["I:out0", "Resize:in0"]],
"src_out_tensor": ["Resize:out0"],
"acu_lys_alias": ["image_resize"],
"src_acu_in_tensor_map": [["I:out0", "image_resize:in0"]],
"src_acu_out_tensor_map": [["Resize:out0", "image_resize:out0"]],
"acu_inter_flow": [],
"param_map":{
    "image_resize":{
        "new_size": ['INTS', 'PYFUNC', r_resize_get_new_size()],
        "align_corners": ['BOOL', 'PYFUNC', r_resize_get_align_corners()],
        "half_pixel": ['BOOL', 'PYFUNC', r_resize_get_half_pixel()],
        "type": ['STRING', 'PYFUNC', r_resize_get_type()],
    }
},
"blob_map": {"image_resize": {}},
"priority_tip": 0,
"pre_condition": r_resize_check(),
"src_ops_main_version": None,
"src_ops_minior_version": [11, -1]}
ruler_list.append(r_resize_i0_constant0_constant1)

r_resize_i_frozentablei1= {
"ruler_name": "r_resize_i_frozentablei1",
"src_ops_alias": ["Resize"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Resize:in0"],["I_1:out0", "Resize:in3"]],
"src_out_tensor": ["Resize:out0"],
"acu_lys_alias": ["image_resize"],
"src_acu_in_tensor_map": [["I:out0", "image_resize:in0"]],
"src_acu_out_tensor_map": [["Resize:out0", "image_resize:out0"]],
"acu_inter_flow": [],
"param_map":{
    "image_resize":{
        "new_size": ['INTS', 'PYFUNC', r_resize_get_new_size()],
        "align_corners": ['BOOL', 'PYFUNC', r_resize_get_align_corners()],
        "half_pixel": ['BOOL', 'PYFUNC', r_resize_get_half_pixel()],
        "type": ['STRING', 'PYFUNC', r_resize_get_type()],
    }
},
"blob_map": {},
"priority_tip": 0,
"pre_condition": r_resize_check(),
"src_ops_main_version": None,
"src_ops_minior_version": [11, -1]
}
ruler_list.append(r_resize_i_frozentablei1)

r_resize_dynamic = {
"ruler_name": "r_resize_size",
"src_ops_alias": ["Resize"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Resize:in0"], ["I_1:out0", "Resize:in1"],
                  ["I_2:out0", "Resize:in2"], ["I_3:out0", "Resize:in3"]],
"src_out_tensor": ["Resize:out0"],
"acu_lys_alias": ["image_resize"],
"src_acu_in_tensor_map": [["I:out0", "image_resize:in0"]],
"src_acu_out_tensor_map": [["Resize:out0", "image_resize:out0"]],
"acu_inter_flow": [],
"param_map":{
    "image_resize":{
        "new_size": ['INTS', 'PYFUNC', r_resize_get_new_size()],
        "align_corners": ['BOOL', 'PYFUNC', r_resize_get_align_corners()],
        "half_pixel": ['BOOL', 'PYFUNC', r_resize_get_half_pixel()],
        "type": ['STRING', 'PYFUNC', r_resize_get_type()],
    }
},
"blob_map": {},
"priority_tip": 0,
"pre_condition": r_resize_check(),
"src_ops_main_version": None,
"src_ops_minior_version": [11, -1]
}
ruler_list.append(r_resize_dynamic)

r_resize_i0_scales = {
"ruler_name": "r_resize_i0_scales",
"src_ops_alias": ["Resize", "Constant"],
"src_inter_flow": [["Constant:out0", "Resize:in2"]],
"src_in_anchor": [["I:out0", "Resize:in0"]],
"src_out_tensor": ["Resize:out0"],
"acu_lys_alias": ["image_resize"],
"src_acu_in_tensor_map": [["I:out0", "image_resize:in0"]],
"src_acu_out_tensor_map": [["Resize:out0", "image_resize:out0"]],
"acu_inter_flow": [],
"param_map":{
    "image_resize":{
        "new_size": ['INTS', 'PYFUNC', r_resize_get_new_size()],
        "align_corners": ['BOOL', 'PYFUNC', r_resize_get_align_corners()],
        "half_pixel": ['BOOL', 'PYFUNC', r_resize_get_half_pixel()],
        "type": ['STRING', 'PYFUNC', r_resize_get_type()],
    }
},
"blob_map": {},
"priority_tip": 0,
"pre_condition": r_resize_check(),
"src_ops_main_version": None,
"src_ops_minior_version": [13, -1]
}
ruler_list.append(r_resize_i0_scales)

r_upsample_expand_to_image_resize = {
"ruler_name": "r_upsample_expand_to_image_resize",
"src_ops_alias": ["Reshape", "Expand", "Constant", "Reshape_1", "Constant_1", "Constant_2"],
"src_inter_flow": [["Expand:out0", "Reshape:in0"], ["Constant:out0", "Reshape:in1"],
                   ["Reshape_1:out0", "Expand:in0"], ["Constant_1:out0", "Expand:in1"],
                   ["Constant_2:out0", "Reshape_1:in1"]],
"src_in_anchor": [["I_0:out0", "Reshape_1:in0"]],
"src_out_tensor": ["Reshape:out0"],
"acu_lys_alias": ["image_resize"],
"src_acu_in_tensor_map": [["I_0:out0", "image_resize:in0"]],
"src_acu_out_tensor_map": [["Reshape:out0", "image_resize:out0"]],
"acu_inter_flow": [],
"param_map": {
"image_resize": {
  "new_size": [
     "ORIGIN",
     "CODE",
     "self.tensor_to_numpy(tensor['Constant:out0'])[2:].tolist()"
    ],
  "align_corners": ['BOOL', 'VALUE', False],
  "half_pixel": ['BOOL', 'VALUE', False],
  "type":['STRING','VALUE',"nearest"]}
    },
"blob_map": {"image_resize": {}},
"priority_tip": 0,
"pre_condition": r_upsample_to_resize_check(
    img_in="I_0:out0", img_out="Reshape:out0", expand_in="Reshape_1:out0", expand_out="Expand:out0"),
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_upsample_expand_to_image_resize)

r_upsample_image_resize = {
"ruler_name": "r_upsample_image_resize",
"src_ops_alias": ["Upsample", "Constant"],
"src_inter_flow": [["Constant:out0", "Upsample:in1"]],
"src_in_anchor": [["I:out0", "Upsample:in0"]],
"src_out_tensor": ["Upsample:out0"],
"acu_lys_alias": ["image_resize"],
"src_acu_in_tensor_map": [["I:out0", "image_resize:in0"]],
"src_acu_out_tensor_map": [["Upsample:out0", "image_resize:out0"]],
"param_map": {"image_resize":
                {"new_size": ["ORIGIN", "CODE", "self.shape_pick(tensor['Upsample:out0'])[2:]"],
                    "align_corners": ["BOOL", "VALUE", False],
                    "type": ['STRING', 'CODE', "self.attr_pick(node['Upsample'], 'mode')"],}
              },
"blob_map": {"image_resize": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_upsample_image_resize)

r_pad_g1 = {
"ruler_name": "pad_g1",
"src_ops_alias": ["Pad"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Pad:in0"]],
"src_out_tensor": ["Pad:out0"],
"acu_lys_alias": ["pad"],
"src_acu_in_tensor_map": [["I:out0", "pad:in0"]],
"src_acu_out_tensor_map": [["Pad:out0", "pad:out0"]],
"param_map": {"pad":
                  {'padding_value': ['ORIGIN', 'CODE', "self.map_pad_value(node['Pad'])"],
                   'padding_mode': ['STRING', 'CODE', "self.attr_pick(node['Pad'], 'mode', 'constant')"],
                   'padding_const': ['ORIGIN', 'CODE', "self.attr_pick(node['Pad'], 'value')"],
                   }
              },
"blob_map": {"pad": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [2, 10]}
ruler_list.append(r_pad_g1)

r_pad_11 = {
"ruler_name": "r_pad_11",
"src_ops_alias": ["Pad", "Constant"],
"src_inter_flow": [["Constant:out0", "Pad:in1"]],
"src_in_anchor": [["I:out0", "Pad:in0"]],
"src_out_tensor": ["Pad:out0"],
"acu_lys_alias": ["pad"],
"src_acu_in_tensor_map": [["I:out0", "pad:in0"]],
"src_acu_out_tensor_map": [["Pad:out0", "pad:out0"]],
"param_map": {"pad":
                  {'padding_value': ['ORIGIN', 'PYFUNC', r_pad_value_map()],
                   'padding_mode': ['STRING', 'CODE', "self.attr_pick(node['Pad'], 'mode', 'constant')"],
                   }
              },
"blob_map": {"pad": {}},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [11, -1]}
ruler_list.append(r_pad_11)

r_pad_1 = {
"ruler_name": "r_pad_1",
"src_ops_alias": ["Pad", "Constant", "Constant_1"],
"src_inter_flow": [["Constant:out0", "Pad:in1"], ["Constant_1:out0", "Pad:in2"]],
"src_in_anchor": [["I_0:out0", "Pad:in0"]],
"src_out_tensor": ["Pad:out0"],
"acu_lys_alias": ["pad"],
"src_acu_in_tensor_map": [["I_0:out0", "pad:in0"]],
"src_acu_out_tensor_map": [["Pad:out0", "pad:out0"]],
"acu_inter_flow": [],
"param_map": {"pad": {'padding_value': ['ORIGIN', 'PYFUNC', r_pad_value_map()],
                      'padding_mode': ['STRING', 'CODE', "self.attr_pick(node['Pad'], 'mode', 'constant')"],
                      'padding_const': ['INT', 'CODE', "self.tensor_to_numpy(tensor['Constant_1:out0'])"]
                   }},
"blob_map": {"pad": {}},
"priority_tip": 0,
"pre_condition": r_pad_padding_const_condition(),
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_pad_1)

r_pad_2 = {
"ruler_name": "r_pad_2",
"src_ops_alias": ["Pad", "Constant", "Constant_1"],
"src_inter_flow": [["Constant:out0", "Pad:in1"], ["Constant_1:out0", "Pad:in2"]],
"src_in_anchor": [["I_0:out0", "Pad:in0"]],
"src_out_tensor": ["Pad:out0"],
"acu_lys_alias": ["pad"],
"src_acu_in_tensor_map": [["I_0:out0", "pad:in0"]],
"src_acu_out_tensor_map": [["Pad:out0", "pad:out0"]],
"acu_inter_flow": [],
"param_map": {"pad": {'padding_value': ['ORIGIN', 'PYFUNC', r_pad_value_map()],
                      'padding_mode': ['STRING', 'CODE', "self.attr_pick(node['Pad'], 'mode', 'constant')"],
                      'padding_const': ['ORIGIN', 'PYFUNC', r_pad_padding_const_map()]
                   }},
"blob_map": {"pad": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_pad_2)

r_pad_with_axes = {
"ruler_name": "r_pad_with_axes",
"src_ops_alias": ["Pad", "Constant", "Constant_1", "Constant_2"],
"src_inter_flow": [["Constant:out0", "Pad:in1"], ["Constant_1:out0", "Pad:in2"], ["Constant_2:out0", "Pad:in3"]],
"src_in_anchor": [["I_0:out0", "Pad:in0"]],
"src_out_tensor": ["Pad:out0"],
"acu_lys_alias": ["pad"],
"src_acu_in_tensor_map": [["I_0:out0", "pad:in0"]],
"src_acu_out_tensor_map": [["Pad:out0", "pad:out0"]],
"acu_inter_flow": [],
"param_map": {"pad": {'padding_value': ['ORIGIN', 'PYFUNC', r_pad_value_map()],
                      'padding_mode': ['STRING', 'CODE', "self.attr_pick(node['Pad'], 'mode', 'constant')"],
                      'padding_const': ['ORIGIN', 'PYFUNC', r_pad_padding_const_map()]}},
"blob_map": {"pad": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [18, -1]}
ruler_list.append(r_pad_with_axes)

r_space2depth = {
"ruler_name": "r_space2depth",
"src_ops_alias": ["SpaceToDepth"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "SpaceToDepth:in0"]],
"src_out_tensor": ["SpaceToDepth:out0"],
"acu_lys_alias": ["space2depth"],
"src_acu_in_tensor_map": [["I:out0", "space2depth:in0"]],
"src_acu_out_tensor_map": [["SpaceToDepth:out0", "space2depth:out0"]],
"param_map":
    {"space2depth":
         {"block_size":
["INTS",
 "CODE",
 "[self.attr_pick(node['SpaceToDepth'], 'blocksize'), self.attr_pick(node['SpaceToDepth'], 'blocksize')]"]
          }
     },
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_space2depth)

r_depth2space = {
"ruler_name": "r_depth2space",
"src_ops_alias": ["DepthToSpace"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "DepthToSpace:in0"]],
"src_out_tensor": ["DepthToSpace:out0"],
"acu_lys_alias": ["depth2space"],
"src_acu_in_tensor_map": [["I:out0", "depth2space:in0"]],
"src_acu_out_tensor_map": [["DepthToSpace:out0", "depth2space:out0"]],
"param_map":
{"depth2space":
     {"block_size":
          ["INT",
           "CODE",
           "self.attr_pick(node['DepthToSpace'], 'blocksize')"
           ],
      "mode":
          ["STRING",
           "CODE",
           "self.attr_pick(node['DepthToSpace'], 'mode', 'DCR')"
           ]
      }
 },
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_depth2space)

r_clip_1 = {
"ruler_name": "r_clip_1",
"src_ops_alias": ["Clip"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "Clip:in0"]],
"src_out_tensor": ["Clip:out0"],
"acu_lys_alias": ["clipbyvalue"],
"src_acu_in_tensor_map": [["I_0:out0", "clipbyvalue:in0"]],
"src_acu_out_tensor_map": [["Clip:out0", "clipbyvalue:out0"]],
"acu_inter_flow": [],
"param_map": {'clipbyvalue':
                  {'clip_value_max': ['FLOAT', 'CODE',
                                      "self.attr_pick(node['Clip'], 'max', 1.0)"],
                  'clip_value_min': ['FLOAT', 'CODE',
                                        "self.attr_pick(node['Clip'], 'min', -1.0)"]}},
"blob_map": {"clipbyvalue": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, 5]}
ruler_list.append(r_clip_1)

r_clip_6 = {
"ruler_name": "r_clip_6",
"src_ops_alias": ["Clip"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "Clip:in0"]],
"src_out_tensor": ["Clip:out0"],
"acu_lys_alias": ["clipbyvalue"],
"src_acu_in_tensor_map": [["I_0:out0", "clipbyvalue:in0"]],
"src_acu_out_tensor_map": [["Clip:out0", "clipbyvalue:out0"]],
"acu_inter_flow": [],
"param_map": {'clipbyvalue':
                  {'clip_value_max': ['FLOAT', 'CODE',
                                      "self.attr_pick(node['Clip'], 'max', np.inf)"],
                  'clip_value_min': ['FLOAT', 'CODE',
                                        "self.attr_pick(node['Clip'], 'min', -np.inf)"]}},
"blob_map": {"clipbyvalue": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [6, 10]}
ruler_list.append(r_clip_6)

r_clip_11 = {
"ruler_name": "r_clip_11",
"src_ops_alias": ["Clip", "Constant", "Constant_1"],
"src_inter_flow": [["Constant:out0", "Clip:in1"], ["Constant_1:out0", "Clip:in2"]],
"src_in_anchor": [["I_0:out0", "Clip:in0"]],
"src_out_tensor": ["Clip:out0"],
"acu_lys_alias": ["clipbyvalue"],
"src_acu_in_tensor_map": [["I_0:out0", "clipbyvalue:in0"]],
"src_acu_out_tensor_map": [["Clip:out0", "clipbyvalue:out0"]],
"acu_inter_flow": [],
"param_map": {'clipbyvalue':
                  {'clip_value_max': ['FLOAT', 'CODE',
                                      "self.tensor_to_numpy(tensor['Constant_1:out0'])"],
                  'clip_value_min': ['FLOAT', 'CODE',
                                        "self.tensor_to_numpy(tensor['Constant:out0'])"]}},
"blob_map": {"clipbyvalue": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [11, -1]}
ruler_list.append(r_clip_11)

r_clip_min = {
"ruler_name": "r_clip_min",
"src_ops_alias": ["Clip", "Constant"],
"src_inter_flow": [["Constant:out0", "Clip:in1"]],
"src_in_anchor": [["I_0:out0", "Clip:in0"]],
"src_out_tensor": ["Clip:out0"],
"acu_lys_alias": ["clipbyvalue"],
"src_acu_in_tensor_map": [["I_0:out0", "clipbyvalue:in0"]],
"src_acu_out_tensor_map": [["Clip:out0", "clipbyvalue:out0"]],
"acu_inter_flow": [],
"param_map": {'clipbyvalue':
                  {'clip_value_max': ['FLOAT', 'CODE', "np.inf"],
                   'clip_value_min': ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['Constant:out0'])"]}},
"blob_map": {"clipbyvalue": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_clip_min)

r_clip_max = {
"ruler_name": "r_clip_max",
"src_ops_alias": ["Clip", "Constant"],
"src_inter_flow": [["Constant:out0", "Clip:in2"]],
"src_in_anchor": [["I_0:out0", "Clip:in0"]],
"src_out_tensor": ["Clip:out0"],
"acu_lys_alias": ["clipbyvalue"],
"src_acu_in_tensor_map": [["I_0:out0", "clipbyvalue:in0"]],
"src_acu_out_tensor_map": [["Clip:out0", "clipbyvalue:out0"]],
"acu_inter_flow": [],
"param_map": {'clipbyvalue':
                  {'clip_value_max': ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['Constant:out0'])"],
                   'clip_value_min': ['FLOAT', 'CODE', "-np.inf"]}},
"blob_map": {"clipbyvalue": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_clip_max)

r_clip_3inputs = {
"ruler_name": "r_clip_3inputs",
"src_ops_alias": ["Clip"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "Clip:in0"], ["I_1:out0", "Clip:in1"], ["I_2:out0", "Clip:in2"]],
"src_out_tensor": ["Clip:out0"],
"acu_lys_alias": ["clipbyvalue"],
"src_acu_in_tensor_map": [["I_0:out0", "clipbyvalue:in0"]],
"src_acu_out_tensor_map": [["Clip:out0", "clipbyvalue:out0"]],
"acu_inter_flow": [],
"param_map": {"clipbyvalue": {'clip_value_max': ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['I_2:out0'])"],
                              'clip_value_min': ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['I_1:out0'])"]}},
"blob_map": {"clipbyvalue": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_clip_3inputs)

r_reducemean = {
"ruler_name": "r_reducemean",
"src_ops_alias": ["ReduceMean"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "ReduceMean:in0"]],
"src_out_tensor": ["ReduceMean:out0"],
"acu_lys_alias": ["reducemean"],
"src_acu_in_tensor_map": [["I_0:out0", "reducemean:in0"]],
"src_acu_out_tensor_map": [["ReduceMean:out0", "reducemean:out0"]],
"acu_inter_flow": [],
"param_map": {'reducemean':{'axis_list': ["INTS", "CODE",
                                "self.reducex_axis_list(node['ReduceMean'], self.shape_pick(tensor['I_0:out0']))"],
                           'keep_dims': ['BOOL','CODE', "self.attr_pick(node['ReduceMean'], 'keepdims', 1)"]}},
"blob_map": {"reducemean": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_reducemean)

r_reducemean_with_constant_axes = {
"ruler_name": "r_reducemean_with_constant_axes",
"src_ops_alias": ["ReduceMean", "Constant"],
"src_inter_flow": [["Constant:out0", "ReduceMean:in1"]],
"src_in_anchor": [["I_0:out0", "ReduceMean:in0"]],
"src_out_tensor": ["ReduceMean:out0"],
"acu_lys_alias": ["reducemean"],
"src_acu_in_tensor_map": [["I_0:out0", "reducemean:in0"]],
"src_acu_out_tensor_map": [["ReduceMean:out0", "reducemean:out0"]],
"acu_inter_flow": [],
"param_map": {'reducemean':{'axis_list': ["INTS",
           "CODE",
           "self.reducesum_constant_axis_list(node['ReduceMean'], tensor['Constant:out0'], "
           "self.shape_pick(tensor['I_0:out0']))"],
           'keep_dims': ['BOOL','CODE', "self.attr_pick(node['ReduceMean'], 'keepdims', 1)"]}},
"blob_map": {"reducemean": {}},
"priority_tip": 0,
"pre_condition": "not (len(self.tensor_to_numpy(tensor['Constant:out0']).tolist()) == 0 and "
                 "self.attr_pick(node['ReduceMean'], 'noop_with_empty_axes', 0) == 1)",
"src_ops_main_version": None,
"src_ops_minior_version": [18, -1]}
ruler_list.append(r_reducemean_with_constant_axes)

r_reducemean_to_noop_with_constant_axes = {
"ruler_name": "r_reducemean_to_noop_with_constant_axes",
"src_ops_alias": ["ReduceMean", "Constant"],
"src_inter_flow": [["Constant:out0", "ReduceMean:in1"]],
"src_in_anchor": [["I_0:out0", "ReduceMean:in0"]],
"src_out_tensor": ["ReduceMean:out0"],
"acu_lys_alias": ["noop"],
"src_acu_in_tensor_map": [["I_0:out0", "noop:in0"]],
"src_acu_out_tensor_map": [["ReduceMean:out0", "noop:out0"]],
"acu_inter_flow": [],
"param_map": {},
"blob_map": {},
"priority_tip": 0,
"pre_condition": "len(self.tensor_to_numpy(tensor['Constant:out0']).tolist()) == 0 and "
                 "self.attr_pick(node['ReduceMean'], 'noop_with_empty_axes', 0) == 1",
"src_ops_main_version": None,
"src_ops_minior_version": [18, -1]}
ruler_list.append(r_reducemean_to_noop_with_constant_axes)

r_reducemax = {
"ruler_name": "r_reducemax",
"src_ops_alias": ["ReduceMax"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "ReduceMax:in0"]],
"src_out_tensor": ["ReduceMax:out0"],
"acu_lys_alias": ["reducemax"],
"src_acu_in_tensor_map": [["I_0:out0", "reducemax:in0"]],
"src_acu_out_tensor_map": [["ReduceMax:out0", "reducemax:out0"]],
"acu_inter_flow": [],
"param_map": {'reducemax':{'axis_list': ["INTS", "CODE",
                                "self.reducex_axis_list(node['ReduceMax'], self.shape_pick(tensor['I_0:out0']))"],
                           'keep_dims': ['BOOL','CODE', "self.attr_pick(node['ReduceMax'], 'keepdims', 1)"]}},
"blob_map": {"reducemax": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_reducemax)

r_reducemax_with_constant_axes = {
"ruler_name": "r_reducemax_with_constant_axes",
"src_ops_alias": ["ReduceMax", "Constant"],
"src_inter_flow": [["Constant:out0", "ReduceMax:in1"]],
"src_in_anchor": [["I_0:out0", "ReduceMax:in0"]],
"src_out_tensor": ["ReduceMax:out0"],
"acu_lys_alias": ["reducemax"],
"src_acu_in_tensor_map": [["I_0:out0", "reducemax:in0"]],
"src_acu_out_tensor_map": [["ReduceMax:out0", "reducemax:out0"]],
"acu_inter_flow": [],
"param_map": {'reducemax':{'axis_list': ["INTS",
           "CODE",
           "self.reducesum_constant_axis_list(node['ReduceMax'], tensor['Constant:out0'], "
           "self.shape_pick(tensor['I_0:out0']))"],
           'keep_dims': ['BOOL','CODE', "self.attr_pick(node['ReduceMax'], 'keepdims', 1)"]}},
"blob_map": {"reducemax": {}},
"priority_tip": 0,
"pre_condition": "not (len(self.tensor_to_numpy(tensor['Constant:out0']).tolist()) == 0 and "
                 "self.attr_pick(node['ReduceMax'], 'noop_with_empty_axes', 0) == 1)",
"src_ops_main_version": None,
"src_ops_minior_version": [18, -1]}
ruler_list.append(r_reducemax_with_constant_axes)

r_reducemax_to_noop_with_constant_axes = {
"ruler_name": "r_reducemax_to_noop_with_constant_axes",
"src_ops_alias": ["ReduceMax", "Constant"],
"src_inter_flow": [["Constant:out0", "ReduceMax:in1"]],
"src_in_anchor": [["I_0:out0", "ReduceMax:in0"]],
"src_out_tensor": ["ReduceMax:out0"],
"acu_lys_alias": ["noop"],
"src_acu_in_tensor_map": [["I_0:out0", "noop:in0"]],
"src_acu_out_tensor_map": [["ReduceMax:out0", "noop:out0"]],
"acu_inter_flow": [],
"param_map": {},
"blob_map": {},
"priority_tip": 0,
"pre_condition": "len(self.tensor_to_numpy(tensor['Constant:out0']).tolist()) == 0 and "
                 "self.attr_pick(node['ReduceMax'], 'noop_with_empty_axes', 0) == 1",
"src_ops_main_version": None,
"src_ops_minior_version": [18, -1]}
ruler_list.append(r_reducemax_to_noop_with_constant_axes)

r_reducemin = {
"ruler_name": "r_reducemin",
"src_ops_alias": ["ReduceMin"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "ReduceMin:in0"]],
"src_out_tensor": ["ReduceMin:out0"],
"acu_lys_alias": ["reducemin"],
"src_acu_in_tensor_map": [["I_0:out0", "reducemin:in0"]],
"src_acu_out_tensor_map": [["ReduceMin:out0", "reducemin:out0"]],
"acu_inter_flow": [],
"param_map": {'reducemin':{'axis_list': ["INTS", "CODE",
                                "self.reducex_axis_list(node['ReduceMin'], self.shape_pick(tensor['I_0:out0']))"],
                           'keep_dims': ['BOOL','CODE', "self.attr_pick(node['ReduceMin'], 'keepdims', 1)"]}},
"blob_map": {"reducemin": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_reducemin)

r_reducemin_with_constant_axes = {
"ruler_name": "r_reducemin_with_constant_axes",
"src_ops_alias": ["ReduceMin", "Constant"],
"src_inter_flow": [["Constant:out0", "ReduceMin:in1"]],
"src_in_anchor": [["I_0:out0", "ReduceMin:in0"]],
"src_out_tensor": ["ReduceMin:out0"],
"acu_lys_alias": ["reducemin"],
"src_acu_in_tensor_map": [["I_0:out0", "reducemin:in0"]],
"src_acu_out_tensor_map": [["ReduceMin:out0", "reducemin:out0"]],
"acu_inter_flow": [],
"param_map": {'reducemin':{'axis_list': ["INTS",
           "CODE",
           "self.reducesum_constant_axis_list(node['ReduceMin'], tensor['Constant:out0'], "
           "self.shape_pick(tensor['I_0:out0']))"],
           'keep_dims': ['BOOL','CODE', "self.attr_pick(node['ReduceMin'], 'keepdims', 1)"]}},
"blob_map": {},
"priority_tip": 0,
"pre_condition": "not (len(self.tensor_to_numpy(tensor['Constant:out0']).tolist()) == 0 and "
                 "self.attr_pick(node['ReduceMin'], 'noop_with_empty_axes', 0) == 1)",
"src_ops_main_version": None,
"src_ops_minior_version": [18, -1]}
ruler_list.append(r_reducemin_with_constant_axes)

r_reducemin_to_noop_with_constant_axes = {
"ruler_name": "r_reducemin_to_noop_with_constant_axes",
"src_ops_alias": ["ReduceMin", "Constant"],
"src_inter_flow": [["Constant:out0", "ReduceMin:in1"]],
"src_in_anchor": [["I_0:out0", "ReduceMin:in0"]],
"src_out_tensor": ["ReduceMin:out0"],
"acu_lys_alias": ["noop"],
"src_acu_in_tensor_map": [["I_0:out0", "noop:in0"]],
"src_acu_out_tensor_map": [["ReduceMin:out0", "noop:out0"]],
"acu_inter_flow": [],
"param_map": {},
"blob_map": {},
"priority_tip": 0,
"pre_condition": "len(self.tensor_to_numpy(tensor['Constant:out0']).tolist()) == 0 and "
                 "self.attr_pick(node['ReduceMin'], 'noop_with_empty_axes', 0) == 1",
"src_ops_main_version": None,
"src_ops_minior_version": [18, -1]}
ruler_list.append(r_reducemin_to_noop_with_constant_axes)

r_reducesum = {
"ruler_name": "r_reducesum",
"src_ops_alias": ["ReduceSum"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "ReduceSum:in0"]],
"src_out_tensor": ["ReduceSum:out0"],
"acu_lys_alias": ["reducesum"],
"src_acu_in_tensor_map": [["I_0:out0", "reducesum:in0"]],
"src_acu_out_tensor_map": [["ReduceSum:out0", "reducesum:out0"]],
"acu_inter_flow": [],
"param_map": {'reducesum':{'axis_list': ["INTS", "CODE",
                                "self.reducex_axis_list(node['ReduceSum'], self.shape_pick(tensor['I_0:out0']))"],
                           'keep_dims': ['BOOL','CODE', "self.attr_pick(node['ReduceSum'], 'keepdims', 1)"]}},
"blob_map": {"reducesum": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_reducesum)

r_reducesum_with_constant_axes = {
"ruler_name": "r_reducesum_with_constant_axes",
"src_ops_alias": ["ReduceSum", "Constant"],
"src_inter_flow": [["Constant:out0", "ReduceSum:in1"]],
"src_in_anchor": [["I_0:out0", "ReduceSum:in0"]],
"src_out_tensor": ["ReduceSum:out0"],
"acu_lys_alias": ["reducesum"],
"src_acu_in_tensor_map": [["I_0:out0", "reducesum:in0"]],
"src_acu_out_tensor_map": [["ReduceSum:out0", "reducesum:out0"]],
"acu_inter_flow": [],
"param_map":
{'reducesum':
     {'axis_list':
          ["INTS",
           "CODE",
           "self.reducesum_constant_axis_list(node['ReduceSum'], tensor['Constant:out0'], "
           "self.shape_pick(tensor['I_0:out0']))"],
      'keep_dims': ['BOOL','CODE', "self.attr_pick(node['ReduceSum'], 'keepdims', 1)"]
      }
 },
"blob_map": {"reducesum": {}},
"priority_tip": 0,
"pre_condition": "not (len(self.tensor_to_numpy(tensor['Constant:out0']).tolist()) == 0 and "
                 "self.attr_pick(node['ReduceSum'], 'noop_with_empty_axes', 0) == 1)",
"src_ops_main_version": None,
"src_ops_minior_version": [13, -1]}
ruler_list.append(r_reducesum_with_constant_axes)

r_reducesum_to_noop_with_constant_axes = {
"ruler_name": "r_reducesum_to_noop_with_constant_axes",
"src_ops_alias": ["ReduceSum", "Constant"],
"src_inter_flow": [["Constant:out0", "ReduceSum:in1"]],
"src_in_anchor": [["I_0:out0", "ReduceSum:in0"]],
"src_out_tensor": ["ReduceSum:out0"],
"acu_lys_alias": ["noop"],
"src_acu_in_tensor_map": [["I_0:out0", "noop:in0"]],
"src_acu_out_tensor_map": [["ReduceSum:out0", "noop:out0"]],
"acu_inter_flow": [],
"param_map":{},
"blob_map": {},
"priority_tip": 0,
"pre_condition": "len(self.tensor_to_numpy(tensor['Constant:out0']).tolist()) == 0 and "
                 "self.attr_pick(node['ReduceSum'], 'noop_with_empty_axes', 0) == 1",
"src_ops_main_version": None,
"src_ops_minior_version": [13, -1]}
ruler_list.append(r_reducesum_to_noop_with_constant_axes)

r_reduceprod = {
"ruler_name": "r_reduceprod",
"src_ops_alias": ["ReduceProd"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "ReduceProd:in0"]],
"src_out_tensor": ["ReduceProd:out0"],
"acu_lys_alias": ["reduceprod"],
"src_acu_in_tensor_map": [["I_0:out0", "reduceprod:in0"]],
"src_acu_out_tensor_map": [["ReduceProd:out0", "reduceprod:out0"]],
"acu_inter_flow": [],
"param_map":
{'reduceprod':
     {'axis_list':
          ["INTS",
           "CODE",
           "self.reducex_axis_list(node['ReduceProd'], self.shape_pick(tensor['I_0:out0']))"],
      'keep_dims': ['BOOL','CODE', "self.attr_pick(node['ReduceProd'], 'keepdims', 1)"]
      }
 },
"blob_map": {"reduceprod": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_reduceprod)

r_reduceprod_with_constant_axes = {
"ruler_name": "r_reduceprod_with_constant_axes",
"src_ops_alias": ["ReduceProd", "Constant"],
"src_inter_flow": [["Constant:out0", "ReduceProd:in1"]],
"src_in_anchor": [["I_0:out0", "ReduceProd:in0"]],
"src_out_tensor": ["ReduceProd:out0"],
"acu_lys_alias": ["reduceprod"],
"src_acu_in_tensor_map": [["I_0:out0", "reduceprod:in0"]],
"src_acu_out_tensor_map": [["ReduceProd:out0", "reduceprod:out0"]],
"acu_inter_flow": [],
"param_map":
{'reduceprod':
     {'axis_list':
          ["INTS",
           "CODE",
           "self.reducesum_constant_axis_list(node['ReduceProd'], tensor['Constant:out0'], "
           "self.shape_pick(tensor['I_0:out0']))"],
      'keep_dims': ['BOOL', 'CODE', "self.attr_pick(node['ReduceProd'], 'keepdims', 1)"]
      }
 },
"blob_map": {"reduceprod": {}},
"priority_tip": 0,
"pre_condition": "not (len(self.tensor_to_numpy(tensor['Constant:out0']).tolist()) == 0 and "
                 "self.attr_pick(node['ReduceProd'], 'noop_with_empty_axes', 0) == 1)",
"src_ops_main_version": None,
"src_ops_minior_version": [18, -1]}
ruler_list.append(r_reduceprod_with_constant_axes)

r_reduceprod_to_noop_with_constant_axes = {
"ruler_name": "r_reduceprod_to_noop_with_constant_axes",
"src_ops_alias": ["ReduceProd", "Constant"],
"src_inter_flow": [["Constant:out0", "ReduceProd:in1"]],
"src_in_anchor": [["I_0:out0", "ReduceProd:in0"]],
"src_out_tensor": ["ReduceProd:out0"],
"acu_lys_alias": ["noop"],
"src_acu_in_tensor_map": [["I_0:out0", "noop:in0"]],
"src_acu_out_tensor_map": [["ReduceProd:out0", "noop:out0"]],
"acu_inter_flow": [],
"param_map":{},
"blob_map": {},
"priority_tip": 0,
"pre_condition": "len(self.tensor_to_numpy(tensor['Constant:out0']).tolist()) == 0 and "
                 "self.attr_pick(node['ReduceProd'], 'noop_with_empty_axes', 0) == 1",
"src_ops_main_version": None,
"src_ops_minior_version": [18, -1]}
ruler_list.append(r_reduceprod_to_noop_with_constant_axes)

r_reducel1 = {
"ruler_name": "r_reducel1",
"src_ops_alias": ["ReduceL1"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "ReduceL1:in0"]],
"src_out_tensor": ["ReduceL1:out0"],
"acu_lys_alias": ["abs", "reducesum"],
"src_acu_in_tensor_map": [["I_0:out0", "abs:in0"]],
"src_acu_out_tensor_map": [["ReduceL1:out0", "reducesum:out0"]],
"acu_inter_flow": [["abs:out0", "reducesum:in0"]],
"param_map": {"abs": {},
              "reducesum": {'axis_list': ['INTS', 'CODE', "self.reducex_axis_list(node['ReduceL1'], "
                                                          "self.shape_pick(tensor['I_0:out0']))"],
                            'keep_dims': ['BOOL', 'CODE', "self.attr_pick(node['ReduceL1'], 'keepdims', 1)"]}},
"blob_map": {"abs": {},
             "reducesum": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_reducel1)

r_reducel1_with_constant_axes = {
"ruler_name": "r_reducel1_with_constant_axes",
"src_ops_alias": ["ReduceL1", "Constant"],
"src_inter_flow": [["Constant:out0", "ReduceL1:in1"]],
"src_in_anchor": [["I_0:out0", "ReduceL1:in0"]],
"src_out_tensor": ["ReduceL1:out0"],
"acu_lys_alias": ["abs", "reducesum"],
"src_acu_in_tensor_map": [["I_0:out0", "abs:in0"]],
"src_acu_out_tensor_map": [["ReduceL1:out0", "reducesum:out0"]],
"acu_inter_flow": [["abs:out0", "reducesum:in0"]],
"param_map": {"reducesum": {'axis_list': ["INTS", "CODE",
              "self.reducesum_constant_axis_list(node['ReduceL1'], tensor['Constant:out0'], "
              "self.shape_pick(tensor['I_0:out0']))"],
                            'keep_dims': ['BOOL', 'CODE', "self.attr_pick(node['ReduceL1'], 'keepdims', 1)"]}},
"blob_map": {},
"priority_tip": 0,
"pre_condition": "not (len(self.tensor_to_numpy(tensor['Constant:out0']).tolist()) == 0 and "
                 "self.attr_pick(node['ReduceL1'], 'noop_with_empty_axes', 0) == 1)",
"src_ops_main_version": None,
"src_ops_minior_version": [18, -1]
}
ruler_list.append(r_reducel1_with_constant_axes)

r_reducel1_to_noop_with_constant_axes = {
"ruler_name": "r_reducel1_to_noop_with_constant_axes",
"src_ops_alias": ["ReduceL1", "Constant"],
"src_inter_flow": [["Constant:out0", "ReduceL1:in1"]],
"src_in_anchor": [["I_0:out0", "ReduceL1:in0"]],
"src_out_tensor": ["ReduceL1:out0"],
"acu_lys_alias": ["noop"],
"src_acu_in_tensor_map": [["I_0:out0", "noop:in0"]],
"src_acu_out_tensor_map": [["ReduceL1:out0", "noop:out0"]],
"acu_inter_flow": [],
"param_map": {},
"blob_map": {},
"priority_tip": 0,
"pre_condition": "len(self.tensor_to_numpy(tensor['Constant:out0']).tolist()) == 0 and "
                 "self.attr_pick(node['ReduceL1'], 'noop_with_empty_axes', 0) == 1",
"src_ops_main_version": None,
"src_ops_minior_version": [18, -1]
}
ruler_list.append(r_reducel1_to_noop_with_constant_axes)

r_reducel2 = {
"ruler_name": 'r_reducel2',
"src_ops_alias": ["ReduceL2"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "ReduceL2:in0"]],
"src_out_tensor": ["ReduceL2:out0"],
"acu_lys_alias": ["reducesum", "multiply", "sqrt"],
"src_acu_in_tensor_map": [["I_0:out0", "multiply:in0"], ["I_0:out0", "multiply:in1"]],
"src_acu_out_tensor_map": [["ReduceL2:out0", "sqrt:out0"]],
"acu_inter_flow": [["multiply:out0", "reducesum:in0"], ["reducesum:out0", "sqrt:in0"]],
"param_map": {"reducesum": {'axis_list': ['INTS', 'CODE', "self.reducex_axis_list(node['ReduceL2'], "
                                                          "self.shape_pick(tensor['I_0:out0']))"],
                            'keep_dims': ['BOOL','CODE', "self.attr_pick(node['ReduceL2'], 'keepdims', 1)"]},
              "multiply":{},
              "sqrt":{}},
"blob_map": {"reducesum": {},
             "multiply": {},
             "sqrt":{}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_reducel2)

r_reducel2_with_constant_axes = {
"ruler_name": "r_reducel2_with_constant_axes",
"src_ops_alias": ["ReduceL2", "Constant"],
"src_inter_flow": [["Constant:out0", "ReduceL2:in1"]],
"src_in_anchor": [["I_0:out0", "ReduceL2:in0"]],
"src_out_tensor": ["ReduceL2:out0"],
"acu_lys_alias": ["reducesum", "multiply", "sqrt"],
"src_acu_in_tensor_map": [["I_0:out0", "multiply:in0"], ["I_0:out0", "multiply:in1"]],
"src_acu_out_tensor_map": [["ReduceL2:out0", "sqrt:out0"]],
"acu_inter_flow": [["multiply:out0", "reducesum:in0"], ["reducesum:out0", "sqrt:in0"]],
"param_map": {"reducesum": {'axis_list': ["INTS", "CODE",
              "self.reducesum_constant_axis_list(node['ReduceL2'], tensor['Constant:out0'], "
              "self.shape_pick(tensor['I_0:out0']))"],
              'keep_dims': ['BOOL','CODE', "self.attr_pick(node['ReduceL2'], 'keepdims', 1)"]}},
"blob_map": {},
"priority_tip": 0,
"pre_condition": "not (len(self.tensor_to_numpy(tensor['Constant:out0']).tolist()) == 0 and "
                 "self.attr_pick(node['ReduceL2'], 'noop_with_empty_axes', 0) == 1)",
"src_ops_main_version": None,
"src_ops_minior_version": [18, -1]
}
ruler_list.append(r_reducel2_with_constant_axes)

r_reducel2_to_noop_with_constant_axes = {
"ruler_name": "r_reducel2_to_noop_with_constant_axes",
"src_ops_alias": ["ReduceL2", "Constant"],
"src_inter_flow": [["Constant:out0", "ReduceL2:in1"]],
"src_in_anchor": [["I_0:out0", "ReduceL2:in0"]],
"src_out_tensor": ["ReduceL2:out0"],
"acu_lys_alias": ["noop"],
"src_acu_in_tensor_map": [["I_0:out0", "noop:in0"]],
"src_acu_out_tensor_map": [["ReduceL2:out0", "noop:out0"]],
"acu_inter_flow": [],
"param_map": {},
"blob_map": {},
"priority_tip": 0,
"pre_condition": "len(self.tensor_to_numpy(tensor['Constant:out0']).tolist()) == 0 and "
                 "self.attr_pick(node['ReduceL2'], 'noop_with_empty_axes', 0) == 1",
"src_ops_main_version": None,
"src_ops_minior_version": [18, -1]
}
ruler_list.append(r_reducel2_to_noop_with_constant_axes)

r_reducelogsum = {
"ruler_name": "r_reducelogsum",
"src_ops_alias": ["ReduceLogSum"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "ReduceLogSum:in0"]],
"src_out_tensor": ["ReduceLogSum:out0"],
"acu_lys_alias": ["reducesum", "log"],
"src_acu_in_tensor_map": [["I_0:out0", "reducesum:in0"]],
"src_acu_out_tensor_map": [["ReduceLogSum:out0", "log:out0"]],
"acu_inter_flow": [["reducesum:out0", "log:in0"]],
"param_map": {"reducesum": {'axis_list': ['INTS', 'CODE', "self.reducex_axis_list(node['ReduceLogSum'], "
                                                          "self.shape_pick(tensor['I_0:out0']))"],
                            'keep_dims': ['BOOL','CODE', "self.attr_pick(node['ReduceLogSum'], 'keepdims', 1)"]},
              "log": {}},
"blob_map": {"reducesum": {},
             "log": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_reducelogsum)

r_reducelogsum_with_constant_axes = {
"ruler_name": "r_reducelogsum_with_constant_axes",
"src_ops_alias": ["ReduceLogSum", "Constant"],
"src_inter_flow": [["Constant:out0", "ReduceLogSum:in1"]],
"src_in_anchor": [["I_0:out0", "ReduceLogSum:in0"]],
"src_out_tensor": ["ReduceLogSum:out0"],
"acu_lys_alias": ["reducesum", "log"],
"src_acu_in_tensor_map": [["I_0:out0", "reducesum:in0"]],
"src_acu_out_tensor_map": [["ReduceLogSum:out0", "log:out0"]],
"acu_inter_flow": [["reducesum:out0", "log:in0"]],
"param_map": {"reducesum": {'axis_list': ["INTS", "CODE",
              "self.reducesum_constant_axis_list(node['ReduceLogSum'], tensor['Constant:out0'], "
              "self.shape_pick(tensor['I_0:out0']))"],
              'keep_dims': ['BOOL', 'CODE', "self.attr_pick(node['ReduceLogSum'], 'keepdims', 1)"]}},
"blob_map": {},
"priority_tip": 0,
"pre_condition": "not (len(self.tensor_to_numpy(tensor['Constant:out0']).tolist()) == 0 and "
                 "self.attr_pick(node['ReduceLogSum'], 'noop_with_empty_axes', 0) == 1)",
"src_ops_main_version": None,
"src_ops_minior_version": [18, -1]
}
ruler_list.append(r_reducelogsum_with_constant_axes)

r_reducelogsum_to_noop_with_constant_axes = {
"ruler_name": "r_reducelogsum_to_noop_with_constant_axes",
"src_ops_alias": ["ReduceLogSum", "Constant"],
"src_inter_flow": [["Constant:out0", "ReduceLogSum:in1"]],
"src_in_anchor": [["I_0:out0", "ReduceLogSum:in0"]],
"src_out_tensor": ["ReduceLogSum:out0"],
"acu_lys_alias": ["noop"],
"src_acu_in_tensor_map": [["I_0:out0", "noop:in0"]],
"src_acu_out_tensor_map": [["ReduceLogSum:out0", "noop:out0"]],
"acu_inter_flow": [],
"param_map": {},
"blob_map": {},
"priority_tip": 0,
"pre_condition": "len(self.tensor_to_numpy(tensor['Constant:out0']).tolist()) == 0 and "
                 "self.attr_pick(node['ReduceLogSum'], 'noop_with_empty_axes', 0) == 1",
"src_ops_main_version": None,
"src_ops_minior_version": [18, -1]
}
ruler_list.append(r_reducelogsum_to_noop_with_constant_axes)

r_reducelogsumexp = {
"ruler_name": "r_reducelogsumexp",
"src_ops_alias": ["ReduceLogSumExp"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "ReduceLogSumExp:in0"]],
"src_out_tensor": ["ReduceLogSumExp:out0"],
"acu_lys_alias": ["exp", "reducesum", "log"],
"src_acu_in_tensor_map": [["I_0:out0", "exp:in0"]],
"src_acu_out_tensor_map": [["ReduceLogSumExp:out0", "log:out0"]],
"acu_inter_flow": [["exp:out0", "reducesum:in0"], ["reducesum:out0", "log:in0"]],
"param_map": {"reducesum": {'axis_list': ['INTS', 'CODE', "self.reducex_axis_list(node['ReduceLogSumExp'], "
                                                          "self.shape_pick(tensor['I_0:out0']))"],
                            'keep_dims': ['BOOL','CODE', "self.attr_pick(node['ReduceLogSumExp'], 'keepdims', 1)"]},
              "exp": {},
              "log": {}},
"blob_map": {"reducesum": {},
             "exp": {},
             "log": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_reducelogsumexp)

r_reducelogsumexp_with_constant_axes = {
"ruler_name": "r_reducelogsumexp_with_constant_axes",
"src_ops_alias": ["ReduceLogSumExp", "Constant"],
"src_inter_flow": [["Constant:out0", "ReduceLogSumExp:in1"]],
"src_in_anchor": [["I_0:out0", "ReduceLogSumExp:in0"]],
"src_out_tensor": ["ReduceLogSumExp:out0"],
"acu_lys_alias": ["exp", "reducesum", "log"],
"src_acu_in_tensor_map": [["I_0:out0", "exp:in0"]],
"src_acu_out_tensor_map": [["ReduceLogSumExp:out0", "log:out0"]],
"acu_inter_flow": [["exp:out0", "reducesum:in0"], ["reducesum:out0", "log:in0"]],
"param_map": {"reducesum": {'axis_list': ["INTS", "CODE",
              "self.reducesum_constant_axis_list(node['ReduceLogSumExp'], tensor['Constant:out0'], "
              "self.shape_pick(tensor['I_0:out0']))"],
              'keep_dims': ['BOOL','CODE', "self.attr_pick(node['ReduceLogSumExp'], 'keepdims', 1)"]},
              },
"blob_map": {},
"priority_tip": 0,
"pre_condition": "not (len(self.tensor_to_numpy(tensor['Constant:out0']).tolist()) == 0 and "
                 "self.attr_pick(node['ReduceLogSum'], 'noop_with_empty_axes', 0) == 1)",
"src_ops_main_version": None,
"src_ops_minior_version": [18, -1]}
ruler_list.append(r_reducelogsumexp_with_constant_axes)

r_reducelogsumexp_to_noop_with_constant_axes = {
"ruler_name": "r_reducelogsumexp_to_noop_with_constant_axes",
"src_ops_alias": ["ReduceLogSumExp", "Constant"],
"src_inter_flow": [["Constant:out0", "ReduceLogSumExp:in1"]],
"src_in_anchor": [["I_0:out0", "ReduceLogSumExp:in0"]],
"src_out_tensor": ["ReduceLogSumExp:out0"],
"acu_lys_alias": ["noop"],
"src_acu_in_tensor_map": [["I_0:out0", "noop:in0"]],
"src_acu_out_tensor_map": [["ReduceLogSumExp:out0", "noop:out0"]],
"acu_inter_flow": [],
"param_map": {},
"blob_map": {},
"priority_tip": 0,
"pre_condition": "len(self.tensor_to_numpy(tensor['Constant:out0']).tolist()) == 0 and "
                 "self.attr_pick(node['ReduceLogSum'], 'noop_with_empty_axes', 0) == 1",
"src_ops_main_version": None,
"src_ops_minior_version": [18, -1]}
ruler_list.append(r_reducelogsumexp_to_noop_with_constant_axes)

r_ReduceSumSquare = {
"ruler_name": "r_ReduceSumSquare",
"src_ops_alias": ["ReduceSumSquare"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "ReduceSumSquare:in0"]],
"src_out_tensor": ["ReduceSumSquare:out0"],
"acu_lys_alias": ["square", "reducesum"],
"src_acu_in_tensor_map": [["I_0:out0", "square:in0"]],
"src_acu_out_tensor_map": [["ReduceSumSquare:out0", "reducesum:out0"]],
"acu_inter_flow": [["square:out0", "reducesum:in0"]],
"param_map": {"reducesum": {'axis_list': ['INTS', 'CODE', "self.reducex_axis_list(node['ReduceSumSquare'], "
                                                          "self.shape_pick(tensor['I_0:out0']))"],
                            'keep_dims': ['BOOL','CODE', "self.attr_pick(node['ReduceSumSquare'], 'keepdims', 1)"]},
              },
"blob_map": {},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_ReduceSumSquare)

r_reducesumsquare_with_constant_axes = {
"ruler_name": "r_reducesumsquare_with_constant_axes",
"src_ops_alias": ["ReduceSumSquare", "Constant"],
"src_inter_flow": [["Constant:out0", "ReduceSumSquare:in1"]],
"src_in_anchor": [["I_0:out0", "ReduceSumSquare:in0"]],
"src_out_tensor": ["ReduceSumSquare:out0"],
"acu_lys_alias": ["square", "reducesum"],
"src_acu_in_tensor_map": [["I_0:out0", "square:in0"]],
"src_acu_out_tensor_map": [["ReduceSumSquare:out0", "reducesum:out0"]],
"acu_inter_flow": [["square:out0", "reducesum:in0"]],
"param_map": {"reducesum": {'axis_list': ["INTS", "CODE",
              "self.reducesum_constant_axis_list(node['ReduceSumSquare'], tensor['Constant:out0'], "
              "self.shape_pick(tensor['I_0:out0']))"],
              'keep_dims': ['BOOL','CODE', "self.attr_pick(node['ReduceSumSquare'], 'keepdims', 1)"]},
              },
"blob_map": {},
"priority_tip": 0,
"pre_condition": "not (len(self.tensor_to_numpy(tensor['Constant:out0']).tolist()) == 0 and "
                 "self.attr_pick(node['ReduceSumSquare'], 'noop_with_empty_axes', 0) == 1)",
"src_ops_main_version": None,
"src_ops_minior_version": [18, -1]
}
ruler_list.append(r_reducesumsquare_with_constant_axes)

r_reducesumsquare_to_noop_with_constant_axes = {
"ruler_name": "r_reducesumsquare_to_noop_with_constant_axes",
"src_ops_alias": ["ReduceSumSquare", "Constant"],
"src_inter_flow": [["Constant:out0", "ReduceSumSquare:in1"]],
"src_in_anchor": [["I_0:out0", "ReduceSumSquare:in0"]],
"src_out_tensor": ["ReduceSumSquare:out0"],
"acu_lys_alias": ["noop"],
"src_acu_in_tensor_map": [["I_0:out0", "noop:in0"]],
"src_acu_out_tensor_map": [["ReduceSumSquare:out0", "noop:out0"]],
"acu_inter_flow": [],
"param_map": {},
"blob_map": {},
"priority_tip": 0,
"pre_condition": "len(self.tensor_to_numpy(tensor['Constant:out0']).tolist()) == 0 and "
                 "self.attr_pick(node['ReduceSumSquare'], 'noop_with_empty_axes', 0) == 1",
"src_ops_main_version": None,
"src_ops_minior_version": [18, -1]}
ruler_list.append(r_reducesumsquare_to_noop_with_constant_axes)

r_gather = {
"ruler_name": "r_gather",
"src_ops_alias": ["Gather"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Gather:in0"], ["I_1:out0", "Gather:in1"]],
"src_out_tensor": ["Gather:out0"],
"acu_lys_alias": ["gather"],
"src_acu_in_tensor_map": [["I:out0", "gather:in0"], ["I_1:out0", "gather:in1"]],
"src_acu_out_tensor_map": [["Gather:out0", "gather:out0"]],
"acu_inter_flow": [],
"param_map": {"gather": {'axis': ['INT', 'CODE', "self.attr_pick(node['Gather'], 'axis', 0)"]}},
"blob_map": {"gather": {}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_gather)

r_gather_elements = {
"ruler_name": "r_gather_elements",
"src_ops_alias": ["GatherElements"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "GatherElements:in0"], ["I_1:out0", "GatherElements:in1"]],
"src_out_tensor": ["GatherElements:out0"],
"acu_lys_alias": ["gather_elements"],
"src_acu_in_tensor_map": [["I:out0", "gather_elements:in0"], ["I_1:out0", "gather_elements:in1"]],
"src_acu_out_tensor_map": [["GatherElements:out0", "gather_elements:out0"]],
"acu_inter_flow": [],
"param_map": {"gather_elements": {'axis': ['INT', 'CODE', "self.attr_pick(node['GatherElements'], 'axis', 0)"]}},
"blob_map": {"gather_elements": {}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_gather_elements)

r_gather_nd = {
"ruler_name": "r_gather_nd",
"src_ops_alias": ["GatherND"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "GatherND:in0"], ["I_1:out0", "GatherND:in1"]],
"src_out_tensor": ["GatherND:out0"],
"acu_lys_alias": ["gathernd"],
"src_acu_in_tensor_map": [["I:out0", "gathernd:in0"], ["I_1:out0", "gathernd:in1"]],
"src_acu_out_tensor_map": [["GatherND:out0", "gathernd:out0"]],
"acu_inter_flow": [],
"param_map": {"gathernd": {'batch_dims': ['INT', 'CODE', "self.attr_pick(node['GatherND'], 'batch_dims', 0)"]}},
"blob_map": {"gathernd": {}},
"priority_tip": 0,
"pre_condition": None}
ruler_list.append(r_gather_nd)

r_softplus = {
"ruler_name": 'r_softplus',
"src_ops_alias": ["Softplus"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "Softplus:in0"]],
"src_out_tensor": ["Softplus:out0"],
"acu_lys_alias": ["softrelu"],
"src_acu_in_tensor_map": [["I_0:out0", "softrelu:in0"]],
"src_acu_out_tensor_map": [["Softplus:out0", "softrelu:out0"]],
"acu_inter_flow": [],
"param_map": {"softrelu": {}},
"blob_map": {"softrelu": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_softplus)

r_tile = {
"ruler_name": "r_tile",
"src_ops_alias": ["Tile", "Constant_0"],
"src_inter_flow": [["Constant_0:out0", "Tile:in1"]],
"src_in_anchor": [["I:out0", "Tile:in0"]],
"src_out_tensor": ["Tile:out0"],
"acu_lys_alias": ["tile"],
"src_acu_in_tensor_map": [["I:out0", "tile:in0"]],
"src_acu_out_tensor_map": [["Tile:out0", "tile:out0"]],
"param_map":
    {"tile":
         {"multiples": ["INTS", "CODE", "self.tensor_to_numpy(tensor['Constant_0:out0'])"]}
     },
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_tile)

r_tile_2_input = {
"ruler_name": "r_tile_2_input",
"src_ops_alias": ["Tile"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Tile:in0"], ["I_1:out0", "Tile:in1"]],
"src_out_tensor": ["Tile:out0"],
"acu_lys_alias": ["tile"],
"src_acu_in_tensor_map": [["I:out0", "tile:in0"], ["I_1:out0", "tile:in1"]],
"src_acu_out_tensor_map": [["Tile:out0", "tile:out0"]],
"param_map":
    {"tile":
         {"multiples": ["INTS", "CODE", "self.tile_get_multiples(node['Tile'], "
                                        "tensor['I:out0'], tensor['Tile:out0'])"]}
     },
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_tile_2_input)

r_argmin = {
"ruler_name": "r_argmin",
"src_ops_alias": ["ArgMin"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "ArgMin:in0"]],
"src_out_tensor": ["ArgMin:out0"],
"acu_lys_alias": ["argmin"],
"src_acu_in_tensor_map":[["I:out0", "argmin:in0"]],
"src_acu_out_tensor_map": [["ArgMin:out0", "argmin:out0"]],
"param_map":{
  "argmin":
    {
      'axis': ['INT', 'CODE', "self.attr_pick(node['ArgMin'], 'axis', 0)"],
      'keepdims': ['BOOL', 'CODE', "self.attr_pick(node['ArgMin'], 'keepdims', 1)"],
      'select_last_index': ['BOOL', 'CODE', "self.attr_pick(node['ArgMin'], 'select_last_index', 0)"],
      'output_type': ['STRING', 'CODE', "self.dtype_pick(tensor['ArgMin:out0'])"]
    }
},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_argmin)

r_argmax = {
"ruler_name": "r_argmax",
"src_ops_alias": ["ArgMax"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "ArgMax:in0"]],
"src_out_tensor": ["ArgMax:out0"],
"acu_lys_alias": ["argmax"],
"src_acu_in_tensor_map":[["I:out0", "argmax:in0"]],
"src_acu_out_tensor_map": [["ArgMax:out0", "argmax:out0"]],
"param_map":{
  "argmax": {
              'output_type': ['STRING', 'CODE', "self.dtype_pick(tensor['ArgMax:out0'])"],
              'keepdims': ['BOOL', 'CODE', "self.attr_pick(node['ArgMax'], 'keepdims', 1)"],
            }
},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_argmax)

r_neg = {
"ruler_name": "r_neg",
"src_ops_alias": ["Neg"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Neg:in0"]],
"src_out_tensor": ["Neg:out0"],
"acu_lys_alias": ["neg"],
"src_acu_in_tensor_map": [["I:out0", "neg:in0"]],
"src_acu_out_tensor_map": [["Neg:out0", "neg:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": None}
ruler_list.append(r_neg)

r_sin = {
"ruler_name": "r_sin",
"src_ops_alias": ["Sin"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "Sin:in0"]],
"src_out_tensor": ["Sin:out0"],
"acu_lys_alias": ["sin"],
"src_acu_in_tensor_map": [["I_0:out0", "sin:in0"]],
"src_acu_out_tensor_map": [["Sin:out0", "sin:out0"]],
"acu_inter_flow": [],
"param_map": {"sin": {}},
"blob_map": {"sin": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_sin)

r_reverse_sequence = {
"ruler_name": "r_reverse_sequence",
"src_ops_alias": ["ReverseSequence"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "ReverseSequence:in0"], ['I_0:out0', "ReverseSequence:in1"]],
"src_out_tensor": ["ReverseSequence:out0"],
"acu_lys_alias": ["reverse_sequence"],
"src_acu_in_tensor_map":[["I:out0", "reverse_sequence:in0"], ['I_0:out0', "reverse_sequence:in1"]],
"src_acu_out_tensor_map": [["ReverseSequence:out0", "reverse_sequence:out0"]],
"param_map": {
    "reverse_sequence":{
        "seq_axis":["INT", "CODE", "self.attr_pick(node['ReverseSequence'], 'time_axis', 0)"],
        "batch_axis":["INT", "CODE", "self.attr_pick(node['ReverseSequence'], 'batch_axis', 1)"],
    },
},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_reverse_sequence)

r_where = {
"ruler_name": "r_where",
"src_ops_alias": ["Where"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "Where:in0"], ["I_1:out0", "Where:in1"], ["I_2:out0", "Where:in2"]],
"src_out_tensor": ["Where:out0"],
"acu_lys_alias": ["where"],
"src_acu_in_tensor_map": [["I_0:out0", "where:in0"], ["I_1:out0", "where:in1"], ["I_2:out0", "where:in2"]],
"src_acu_out_tensor_map": [["Where:out0", "where:out0"]],
"acu_inter_flow": [],
"param_map": {"where": {}},
"blob_map": {"where": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_where)

r_matmul = {
"ruler_name": "r_matmul",
"src_ops_alias": ["MatMul"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "MatMul:in0"], ['I_1:out0', "MatMul:in1"]],
"src_out_tensor": ["MatMul:out0"],
"acu_lys_alias": ["matmul"],
"src_acu_in_tensor_map":[["I:out0", "matmul:in0"], ['I_1:out0', "matmul:in1"]],
"src_acu_out_tensor_map": [["MatMul:out0", "matmul:out0"]],
"param_map": {"matmul": {'transpose_a': ['BOOL', 'CODE', "self.attr_pick(node['MatMul'], 'transpose_a', False)"],
                         'transpose_b': ['BOOL', 'CODE', "self.attr_pick(node['MatMul'], 'transpose_b', False)"],}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_matmul)

r_xor = {
"ruler_name": "r_logical_xor",
"src_ops_alias": ["Xor"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "Xor:in0"], ["I_1:out0", "Xor:in1"]],
"src_out_tensor": ["Xor:out0"],
"acu_lys_alias": ["logical_xor"],
"src_acu_in_tensor_map": [["I_0:out0", "logical_xor:in0"], ["I_1:out0", "logical_xor:in1"]],
"src_acu_out_tensor_map": [["Xor:out0", "logical_xor:out0"]],
"acu_inter_flow": [],
"param_map": {"logical_xor": {}},
"blob_map": {"logical_xor": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_xor)

r_not = {
"ruler_name": "r_not",
"src_ops_alias": ["Not"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "Not:in0"]],
"src_out_tensor": ["Not:out0"],
"acu_lys_alias": ["logical_not"],
"src_acu_in_tensor_map": [["I_0:out0", "logical_not:in0"]],
"src_acu_out_tensor_map": [["Not:out0", "logical_not:out0"]],
"acu_inter_flow": [],
"param_map": {"logical_not": {}},
"blob_map": {"logical_not": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_not)

r_round = {
"ruler_name": "r_round",
"src_ops_alias": ["Round"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "Round:in0"]],
"src_out_tensor": ["Round:out0"],
"acu_lys_alias": ["round"],
"src_acu_in_tensor_map": [["I_0:out0", "round:in0"]],
"src_acu_out_tensor_map": [["Round:out0", "round:out0"]],
"acu_inter_flow": [],
"param_map": {"round": {}},
"blob_map": {"round": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_round)

r_expand = {
"ruler_name": "r_expand",
"src_ops_alias": ["Expand", "Constant"],
"src_inter_flow": [["Constant:out0", "Expand:in1"]],
"src_in_anchor": [["I_0:out0", "Expand:in0"]],
"src_out_tensor": ["Expand:out0"],
"acu_lys_alias": ["expand_broadcast"],
"src_acu_in_tensor_map": [["I_0:out0", "expand_broadcast:in0"]],
"src_acu_out_tensor_map": [["Expand:out0", "expand_broadcast:out0"]],
"acu_inter_flow": [],
"param_map": {
    "expand_broadcast": {
        "shape": ["INTS", "PYFUNC", r_expand_broadcast_shape(in_tensor="I_0:out0")]
    }
},
"blob_map": {"expand_broadcast": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_expand)

r_expand_dynamic = {
"ruler_name": "r_expand_dynamic",
"src_ops_alias": ["Expand"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "Expand:in0"],["I_1:out0", "Expand:in1"]],
"src_out_tensor": ["Expand:out0"],
"acu_lys_alias": ["expand_broadcast"],
"src_acu_in_tensor_map": [["I_0:out0", "expand_broadcast:in0"]],
"src_acu_out_tensor_map": [["Expand:out0", "expand_broadcast:out0"]],
"acu_inter_flow": [],
"param_map": {
    "expand_broadcast": {
        "shape": ["INTS", "CODE", "self.tensor_to_numpy(tensor['I_1:out0'])"]
    }
},
"blob_map": {"expand_broadcast": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_expand_dynamic)

r_quantizelinear = {
"ruler_name": "r_quantizelinear",
"src_ops_alias": ["QuantizeLinear", "Constant", "Constant_1"],
"src_inter_flow": [["Constant:out0", "QuantizeLinear:in1"], ["Constant_1:out0", "QuantizeLinear:in2"]],
"src_in_anchor": [["I_0:out0", "QuantizeLinear:in0"]],
"src_out_tensor": ["QuantizeLinear:out0"],
"acu_lys_alias": ["quantize"],
"src_acu_in_tensor_map": [["I_0:out0", "quantize:in0"]],
"src_acu_out_tensor_map": [["QuantizeLinear:out0", "quantize:out0"]],
"acu_inter_flow": [],
"param_map": {"quantize": {}},
"blob_map": {"quantize": {}},
"extension": [
    ["CODE", "self.qnt_out_tensor(acu_ly['quantize'], tensor['Constant:out0'], tensor['Constant_1:out0'], 0)"]
],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_quantizelinear)
#QuantizeLinear:QuantizeLinear_Y;Constant:Initializer_X_SCALE;Constant_1:Initializer_X_ZP

r_dequantizelinear = {
"ruler_name": "r_dequantizelinear",
"src_ops_alias": ["DequantizeLinear", "Constant", "Constant_1"],
"src_inter_flow": [["Constant:out0", "DequantizeLinear:in1"], ["Constant_1:out0", "DequantizeLinear:in2"]],
"src_in_anchor": [["I_0:out0", "DequantizeLinear:in0"]],
"src_out_tensor": ["DequantizeLinear:out0"],
"acu_lys_alias": ["dequantize"],
"src_acu_in_tensor_map": [["I_0:out0", "dequantize:in0"]],
"src_acu_out_tensor_map": [["DequantizeLinear:out0", "dequantize:out0"]],
"acu_inter_flow": [],
"param_map": {"dequantize": {}},
"blob_map": {"dequantize": {}},
"extension": [
    ["CODE", "self.qnt_in_tensor(acu_ly['dequantize'], tensor['Constant:out0'], tensor['Constant_1:out0'], 0)"]
],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_dequantizelinear)
#DequantizeLinear:DequantizeLinear_Y;Constant:Initializer_scale;Constant_1:Initializer_zp

r_qlinearmatmul = {
"ruler_name": "qlinearmatmul",
"src_ops_alias": ["QLinearMatMul", "Constant", "Constant_1", "Constant_2", "Constant_3", "Constant_4", "Constant_5"],
"src_inter_flow": [["Constant:out0", "QLinearMatMul:in1"], ["Constant_1:out0", "QLinearMatMul:in2"],
                   ["Constant_2:out0", "QLinearMatMul:in4"], ["Constant_3:out0", "QLinearMatMul:in5"],
                   ["Constant_4:out0", "QLinearMatMul:in6"], ["Constant_5:out0", "QLinearMatMul:in7"]],
"src_in_anchor": [["I_0:out0", "QLinearMatMul:in0"], ["I_1:out0", "QLinearMatMul:in3"]],
"src_out_tensor": ["QLinearMatMul:out0"],
"acu_lys_alias": ["matmul"],
"src_acu_in_tensor_map": [["I_0:out0", "matmul:in0"], ["I_1:out0", "matmul:in1"]],
"src_acu_out_tensor_map": [["QLinearMatMul:out0", "matmul:out0"]],
"acu_inter_flow": [],
"param_map": {"matmul": {}},
"blob_map": {"matmul": {}},
"extension": [
    ["CODE", "self.qnt_in_tensor(acu_ly['matmul'], tensor['Constant:out0'], tensor['Constant_1:out0'], 0)"],
    ["CODE", "self.qnt_in_tensor(acu_ly['matmul'], tensor['Constant_2:out0'], tensor['Constant_3:out0'], 1)"],
    ["CODE", "self.qnt_out_tensor(acu_ly['matmul'], tensor['Constant_4:out0'], tensor['Constant_5:out0'], 0)"],
],
"priority_tip": 0,
"pre_condition": "self.tensor_to_numpy(tensor['Constant_2:out0']).shape[0] == 1",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_qlinearmatmul)
#QLinearMatMul:QLinearMatMul_Y;Constant:Initializer_X0_SCALE;Constant_1:Initializer_X0_ZP;
#Constant_2:Initializer_X1_SCALE;Constant_3:Initializer_X1_ZP;Constant_4:Initializer_Y_SCALE;
#Constant_5:Initializer_Y_ZP

r_qlinearmatmul_var = {
"ruler_name": "r_qlinearmatmul_var",
"src_ops_alias": ["QLinearMatMul", "Constant", "Constant_1", "Constant_2", "Constant_3",
                  "Constant_4", "Constant_5", "Constant_6"],
"src_inter_flow": [
    ["Constant:out0", "QLinearMatMul:in1"],
    ["Constant_1:out0", "QLinearMatMul:in2"],
    ["Constant_2:out0", "QLinearMatMul:in3"],
    ["Constant_3:out0", "QLinearMatMul:in4"],
    ["Constant_4:out0", "QLinearMatMul:in5"],
    ["Constant_5:out0", "QLinearMatMul:in6"],
    ["Constant_6:out0", "QLinearMatMul:in7"]
],
"src_in_anchor": [["I_0:out0", "QLinearMatMul:in0"]],
"src_out_tensor": ["QLinearMatMul:out0"],
"acu_lys_alias": ["matmul", "variable"],
"src_acu_in_tensor_map": [["I_0:out0", "matmul:in0"]],
"src_acu_out_tensor_map": [["QLinearMatMul:out0", "matmul:out0"]],
"acu_inter_flow": [["variable:out0", "matmul:in1"]],
"param_map": {
    "matmul": {
        'transpose_a': ['BOOL', 'VALUE', False],
        'transpose_b': ['BOOL', 'VALUE', False],
    },
    "variable": {
        'shape': ['ORIGIN', 'CODE', "self.shape_pick(tensor['Constant_2:out0'])"],
    }
},
"blob_map": {
    "variable": {
        'data': ["CODE", "self.qtensor_to_numpy("
                         "tensor['Constant_2:out0'],"
                         "self.tensor_to_numpy(tensor['Constant_3:out0']),"
                         "self.tensor_to_numpy(tensor['Constant_4:out0']))"]
    }
},
"extension": [
    ["CODE", "self.qnt_in_tensor(acu_ly['matmul'], tensor['Constant:out0'], tensor['Constant_1:out0'], 0)"],
    ["CODE", "self.qnt_in_tensor(acu_ly['matmul'], tensor['Constant_3:out0'], tensor['Constant_4:out0'], 1)"],
    ["CODE", "self.qnt_out_tensor(acu_ly['matmul'], tensor['Constant_5:out0'], tensor['Constant_6:out0'], 0)"],
],
"priority_tip": 0,
"pre_condition": "self.tensor_to_numpy(tensor['Constant_3:out0']).shape[0] == 1", #only support per-tensor scale
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_qlinearmatmul_var)
# QLinearMatMul:QLinearMatMul_QLinearMatMul_104;Constant:Initializer_109;Constant_1:Initializer_110;
# Constant_2:Initializer_111;Constant_3:Initializer_112;Constant_4:Initializer_113;
# Constant_5:Initializer_114;Constant_6:Initializer_115

r_qlinearconv_no_bias = {
"ruler_name": "qlinearconv_no_bias",
"src_ops_alias": ["QLinearConv", "Constant", "Constant_1", "Constant_2",
                  "Constant_3", "Constant_4", "Constant_5", "Constant_6"],
"src_inter_flow": [["Constant:out0", "QLinearConv:in1"], ["Constant_1:out0", "QLinearConv:in2"],
                   ["Constant_2:out0", "QLinearConv:in3"], ["Constant_3:out0", "QLinearConv:in4"],
                   ["Constant_4:out0", "QLinearConv:in5"], ["Constant_5:out0", "QLinearConv:in6"],
                   ["Constant_6:out0", "QLinearConv:in7"]],
"src_in_anchor": [["I_0:out0", "QLinearConv:in0"]],
"src_out_tensor": ["QLinearConv:out0"],
"acu_lys_alias": ["convolution"],
"src_acu_in_tensor_map": [["I_0:out0", "convolution:in0"]],
"src_acu_out_tensor_map": [["QLinearConv:out0", "convolution:out0"]],
"acu_inter_flow": [],
"param_map":
{"convolution":
{"weights": ["INT", "CODE", "self.shape_pick(tensor['Constant_2:out0'])[0]"],
# "pad_method":
#    ["STRING", "CODE", "'auto' if self.attr_pick(node['QLinearConv'], 'pads', None) == None else 'padding_const'"],
"pad_method":
   ["STRING", "CODE",
    "'auto' if self.attr_pick(node['QLinearConv'], 'auto_pad', 'NOTSET') != 'NOTSET' else 'padding_const'"],
"bias": ["BOOL", "VALUE", False],
"ksize_w": ["INT", "CODE", "self.shape_pick(tensor['Constant_2:out0'])[3]"],
"group_number": ["INT", "CODE", "self.attr_pick(node['QLinearConv'], 'group', 1)"],
"ksize_h": ["INT", "CODE", "self.shape_pick(tensor['Constant_2:out0'])[2]"],
"stride_w": ["INT", "CODE", "self.attr_pick(node['QLinearConv'], 'strides', default =[1,1])[1]"],
"stride_h": ["INT", "CODE", "self.attr_pick(node['QLinearConv'], 'strides', default=[1,1])[0]"],
"dilation":
     ['INT',
      'CODE',
      "self.attr_pick(node['QLinearConv'], 'dilations')"\
      " if isinstance(self.attr_pick(node['QLinearConv'], 'dilations'), int)"\
      " else self.attr_pick(node['QLinearConv'], 'dilations')[0]"],
"padding":
   ["STRING",
    "CODE",
    "'SAME' if self.attr_pick(node['QLinearConv'], 'auto_pad', 'NOTSET') in ['SAME_UPPER', 'SAME_LOWER'] "
    "else 'VALID' "],
"pad":
   ["INTS",
    "CODE",
    "[p for p in self.array_layout(self.attr_pick(node['QLinearConv'], 'pads', [ 0, 0, 0, 0]), [0, 2, 1, 3])]"]
}
},
"blob_map": {"convolution":
                 {"weight": ["CODE", "self.qtensor_to_numpy("
                                     "tensor['Constant_2:out0'],"
                                     "self.tensor_to_numpy(tensor['Constant_3:out0']),"
                                     "self.tensor_to_numpy(tensor['Constant_4:out0']))"]}},
"extension": [
    ["CODE", "self.qnt_coef_tensor("
             "'weight', acu_ly['convolution'],"
             "tensor['Constant_3:out0'],"
             "tensor['Constant_4:out0'])"],
    ["CODE", "self.qnt_out_tensor(acu_ly['convolution'], tensor['Constant_5:out0'], tensor['Constant_6:out0'], 0)"],
    ["CODE", "self.qnt_in_tensor(acu_ly['convolution'], tensor['Constant:out0'], tensor['Constant_1:out0'], 0)"]
],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_qlinearconv_no_bias)
#QLinearConv:QLinearConv_Y;Constant:Initializer_X_SCALE;Constant_1:Initializer_X_ZP;Constant_2:Initializer_W;
#Constant_3:Initializer_W_SCALE;Constant_4:Initializer_W_ZP;Constant_5:Initializer_Y_SCALE;Constant_6:Initializer_Y_ZP

r_qlinearconv_with_bias = {
"ruler_name": "qlinearconv_with_bias",
"src_ops_alias": ["QLinearConv", "Constant", "Constant_1", "Constant_2",
                  "Constant_3", "Constant_4", "Constant_5", "Constant_6", "Constant_7"],
"src_inter_flow": [["Constant:out0", "QLinearConv:in1"], ["Constant_1:out0", "QLinearConv:in2"],
                   ["Constant_2:out0", "QLinearConv:in3"], ["Constant_3:out0", "QLinearConv:in4"],
                   ["Constant_4:out0", "QLinearConv:in5"], ["Constant_5:out0", "QLinearConv:in6"],
                   ["Constant_6:out0", "QLinearConv:in7"], ["Constant_7:out0", "QLinearConv:in8"]],
"src_in_anchor": [["I_0:out0", "QLinearConv:in0"]],
"src_out_tensor": ["QLinearConv:out0"],
"acu_lys_alias": ["convolution"],
"src_acu_in_tensor_map": [["I_0:out0", "convolution:in0"]],
"src_acu_out_tensor_map": [["QLinearConv:out0", "convolution:out0"]],
"acu_inter_flow": [],
"param_map":
{"convolution":
{"weights": ["INT", "CODE", "self.shape_pick(tensor['Constant_2:out0'])[0]"],
"pad_method":
   ["STRING", "CODE",
    "'auto' if self.attr_pick(node['QLinearConv'], 'auto_pad', 'NOTSET') != 'NOTSET' else 'padding_const'"],
"bias": ["BOOL", "VALUE", True],
"ksize_w": ["INT", "CODE", "self.shape_pick(tensor['Constant_2:out0'])[3]"],
"group_number": ["INT", "CODE", "self.attr_pick(node['QLinearConv'], 'group', 1)"],
"ksize_h": ["INT", "CODE", "self.shape_pick(tensor['Constant_2:out0'])[2]"],
"stride_w": ["INT", "CODE", "self.attr_pick(node['QLinearConv'], 'strides', default =[1,1])[1]"],
"stride_h": ["INT", "CODE", "self.attr_pick(node['QLinearConv'], 'strides', default=[1,1])[0]"],
"dilation":
     ['INT',
      'CODE',
      "self.attr_pick(node['QLinearConv'], 'dilations')"\
      " if isinstance(self.attr_pick(node['QLinearConv'], 'dilations'), int)"\
      " else self.attr_pick(node['QLinearConv'], 'dilations')[0]"],
"padding":
   ["STRING",
    "CODE",
    "'SAME' if self.attr_pick(node['QLinearConv'], 'auto_pad', 'NOTSET') in ['SAME_UPPER', 'SAME_LOWER'] "
    "else 'VALID' "],
"pad":
   ["INTS",
    "CODE",
    "[p for p in self.array_layout(self.attr_pick(node['QLinearConv'], 'pads', [ 0, 0, 0, 0]), [0, 2, 1, 3])]"]
}
},
"blob_map": {"convolution":
                 {"weight": ["CODE", "self.qtensor_to_numpy("
                                     "tensor['Constant_2:out0'],"
                                     "self.tensor_to_numpy(tensor['Constant_3:out0']),"
                                     "self.tensor_to_numpy(tensor['Constant_4:out0']))"],
                  "bias": ["CODE", "self.qtensor_to_numpy("
                                     "tensor['Constant_7:out0'],"
                                     "self.tensor_to_numpy(tensor['Constant:out0'])"
                                     "*self.tensor_to_numpy(tensor['Constant_3:out0']),"
                                     "np.array([0]*self.shape_pick(tensor['Constant_2:out0'])[0],dtype=np.int32))"]
                  }
             },
"extension": [
    ["CODE", "self.qnt_coef_tensor("
             "'weight', acu_ly['convolution'],"
             "tensor['Constant_3:out0'],"
             "tensor['Constant_4:out0'])"],
    ["CODE", "self.qnt_bias_tensor(acu_ly['convolution'], tensor['Constant:out0'], tensor['Constant_3:out0'])"],
    ["CODE", "self.qnt_out_tensor(acu_ly['convolution'], tensor['Constant_5:out0'], tensor['Constant_6:out0'], 0)"],
    ["CODE", "self.qnt_in_tensor(acu_ly['convolution'], tensor['Constant:out0'], tensor['Constant_1:out0'], 0)"]
],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_qlinearconv_with_bias)
#QLinearConv:QLinearConv_Y;Constant:Initializer_X_SCALE;Constant_1:Initializer_X_ZP;Constant_2:Initializer_W;
#Constant_3:Initializer_W_SCALE;Constant_4:Initializer_W_ZP;Constant_5:Initializer_Y_SCALE;
#Constant_6:Initializer_Y_ZP;Constant_7:Initializer_B

r_qlinearconv_1d_with_bias_share_zp = {
"ruler_name": "qlinearconv_1d_with_bias_share_zp",
"src_ops_alias": ["QLinearConv", "Constant", "Constant_1", "Constant_2",
                  "Constant_3", "Constant_4", "Constant_5"],
"src_inter_flow": [["Constant:out0", "QLinearConv:in1"], ["Constant_1:out0", "QLinearConv:in2"],
                   ["Constant_2:out0", "QLinearConv:in3"], ["Constant_3:out0", "QLinearConv:in4"],
                   ["Constant_1:out0", "QLinearConv:in5"], ["Constant_4:out0", "QLinearConv:in6"],
                   ["Constant_1:out0", "QLinearConv:in7"], ["Constant_5:out0", "QLinearConv:in8"]],
"src_in_anchor": [["I_0:out0", "QLinearConv:in0"]],
"src_out_tensor": ["QLinearConv:out0"],
"acu_lys_alias": ["conv1d"],
"src_acu_in_tensor_map": [["I_0:out0", "conv1d:in0"]],
"src_acu_out_tensor_map": [["QLinearConv:out0", "conv1d:out0"]],
"acu_inter_flow": [],
"param_map":
{"conv1d":
{"weights": ["INT", "CODE", "self.shape_pick(tensor['Constant_2:out0'])[0]"],
"pad_method":
   ["STRING", "CODE",
    "'auto' if self.attr_pick(node['QLinearConv'], 'auto_pad', 'NOTSET') != 'NOTSET' else 'padding_const'"],
"bias": ["BOOL", "VALUE", True],
"group_number": ["INT", "CODE", "self.attr_pick(node['QLinearConv'], 'group', 1)"],
"ksize": ["INT", "CODE", "self.shape_pick(tensor['Constant_2:out0'])[2]"],
"stride": ["INT", "CODE", "self.attr_pick(node['QLinearConv'], 'strides', default=[1])[0]"],
"dilation":
     ['INT',
      'CODE',
      "self.attr_pick(node['QLinearConv'], 'dilations')"\
      " if isinstance(self.attr_pick(node['QLinearConv'], 'dilations'), int)"\
      " else self.attr_pick(node['QLinearConv'], 'dilations')[0]"],
"padding":
   ["STRING",
    "CODE",
    "'SAME' if self.attr_pick(node['QLinearConv'], 'auto_pad', 'NOTSET') in ['SAME_UPPER', 'SAME_LOWER'] "
    "else 'VALID' "],
"pad":
   ["INTS",
    "CODE",
    "[p for p in self.array_layout(self.attr_pick(node['QLinearConv'], 'pads', [ 0, 0,]), [0, 1])]"]
}
},
"blob_map": {"conv1d":
                 {"weight": ["CODE", "self.qtensor_to_numpy("
                                     "tensor['Constant_2:out0'],"
                                     "self.tensor_to_numpy(tensor['Constant_3:out0']),"
                                     "self.tensor_to_numpy(tensor['Constant_1:out0']))"],
                  "bias": ["CODE", "self.qtensor_to_numpy("
                                     "tensor['Constant_5:out0'],"
                                     "self.tensor_to_numpy(tensor['Constant:out0'])"
                                     "*self.tensor_to_numpy(tensor['Constant_3:out0']),"
                                     "np.array([0]*self.shape_pick(tensor['Constant_2:out0'])[0],dtype=np.int32))"]
                  }
             },
"extension": [
    ["CODE", "self.qnt_coef_tensor("
             "'weight', acu_ly['convolution'],"
             "tensor['Constant_3:out0'],"
             "tensor['Constant_1:out0'])"],
    ["CODE", "self.qnt_bias_tensor(acu_ly['convolution'], tensor['Constant:out0'], tensor['Constant_3:out0'])"],
    ["CODE", "self.qnt_out_tensor(acu_ly['convolution'], tensor['Constant_4:out0'], tensor['Constant_1:out0'], 0)"],
    ["CODE", "self.qnt_in_tensor(acu_ly['convolution'], tensor['Constant:out0'], tensor['Constant_1:out0'], 0)"]
],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_qlinearconv_1d_with_bias_share_zp)

r_qlinearadd = {
"ruler_name": "r_qlinearadd",
"src_ops_alias": ["QLinearAdd", "Constant", "Constant_1", "Constant_2", "Constant_3", "Constant_4", "Constant_5"],
"src_inter_flow": [["Constant:out0", "QLinearAdd:in1"], ["Constant_1:out0", "QLinearAdd:in2"],
                   ["Constant_2:out0", "QLinearAdd:in4"], ["Constant_3:out0", "QLinearAdd:in5"],
                   ["Constant_4:out0", "QLinearAdd:in6"], ["Constant_5:out0", "QLinearAdd:in7"]],
"src_in_anchor": [["I_0:out0", "QLinearAdd:in0"], ["I_1:out0", "QLinearAdd:in3"]],
"src_out_tensor": ["QLinearAdd:out0"],
"acu_lys_alias": ["add"],
"src_acu_in_tensor_map": [["I_0:out0", "add:in0"], ["I_1:out0", "add:in1"]],
"src_acu_out_tensor_map": [["QLinearAdd:out0", "add:out0"]],
"acu_inter_flow": [],
"param_map": {"add": {}},
"blob_map": {"add": {}},
"extension": [
    ["CODE", "self.qnt_in_tensor(acu_ly['add'], tensor['Constant:out0'], tensor['Constant_1:out0'], 0)"],
    ["CODE", "self.qnt_in_tensor(acu_ly['add'], tensor['Constant_2:out0'], tensor['Constant_3:out0'], 1)"],
    ["CODE", "self.qnt_out_tensor(acu_ly['add'], tensor['Constant_4:out0'], tensor['Constant_5:out0'], 0)"],
],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_qlinearadd)

r_qlinearmul = {
"ruler_name": "r_qlinearmul",
"src_ops_alias": ["QLinearMul", "Constant", "Constant_1", "Constant_2", "Constant_3", "Constant_4", "Constant_5"],
"src_inter_flow": [["Constant:out0", "QLinearMul:in1"], ["Constant_1:out0", "QLinearMul:in2"],
                   ["Constant_2:out0", "QLinearMul:in4"], ["Constant_3:out0", "QLinearMul:in5"],
                   ["Constant_4:out0", "QLinearMul:in6"], ["Constant_5:out0", "QLinearMul:in7"]],
"src_in_anchor": [["I_0:out0", "QLinearMul:in0"], ["I_1:out0", "QLinearMul:in3"]],
"src_out_tensor": ["QLinearMul:out0"],
"acu_lys_alias": ["multiply"],
"src_acu_in_tensor_map": [["I_0:out0", "multiply:in0"], ["I_1:out0", "multiply:in1"]],
"src_acu_out_tensor_map": [["QLinearMul:out0", "multiply:out0"]],
"acu_inter_flow": [],
"param_map": {"multiply": {}},
"blob_map": {"multiply": {}},
"extension": [
    ["CODE", "self.qnt_in_tensor(acu_ly['multiply'], tensor['Constant:out0'], tensor['Constant_1:out0'], 0)"],
    ["CODE", "self.qnt_in_tensor(acu_ly['multiply'], tensor['Constant_2:out0'], tensor['Constant_3:out0'], 1)"],
    ["CODE", "self.qnt_out_tensor(acu_ly['multiply'], tensor['Constant_4:out0'], tensor['Constant_5:out0'], 0)"],
],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_qlinearmul)

r_qlinearsigmoid = {
"ruler_name": "r_qlinearsigmoid",
"src_ops_alias": ["QLinearSigmoid", "Constant", "Constant_1", "Constant_2", "Constant_3"],
"src_inter_flow": [["Constant:out0", "QLinearSigmoid:in1"], ["Constant_1:out0", "QLinearSigmoid:in2"],
                   ["Constant_2:out0", "QLinearSigmoid:in3"], ["Constant_3:out0", "QLinearSigmoid:in4"]],
"src_in_anchor": [["I_0:out0", "QLinearSigmoid:in0"]],
"src_out_tensor": ["QLinearSigmoid:out0"],
"acu_lys_alias": ["sigmoid"],
"src_acu_in_tensor_map": [["I_0:out0", "sigmoid:in0"]],
"src_acu_out_tensor_map": [["QLinearSigmoid:out0", "sigmoid:out0"]],
"acu_inter_flow": [],
"param_map": {"sigmoid": {}},
"blob_map": {"sigmoid": {}},
"extension": [
    ["CODE", "self.qnt_in_tensor(acu_ly['sigmoid'], tensor['Constant:out0'], tensor['Constant_1:out0'], 0)"],
    ["CODE", "self.qnt_out_tensor(acu_ly['sigmoid'], tensor['Constant_2:out0'], tensor['Constant_3:out0'], 0)"],
],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_qlinearsigmoid)

r_qlinearglobalavgpool = {
"ruler_name": 'r_qlinearglobalavgpool',
"src_ops_alias": ["QLinearGlobalAveragePool", "Constant", "Constant_1", "Constant_2", "Constant_3"],
"src_inter_flow": [["Constant:out0", "QLinearGlobalAveragePool:in1"],
                   ["Constant_1:out0", "QLinearGlobalAveragePool:in2"],
                   ["Constant_2:out0", "QLinearGlobalAveragePool:in3"],
                   ["Constant_3:out0", "QLinearGlobalAveragePool:in4"]],
"src_in_anchor": [["I_0:out0", "QLinearGlobalAveragePool:in0"]],
"src_out_tensor": ["QLinearGlobalAveragePool:out0"],
"acu_lys_alias": ["pooling"],
"src_acu_in_tensor_map": [["I_0:out0", "pooling:in0"]],
"src_acu_out_tensor_map": [["QLinearGlobalAveragePool:out0", "pooling:out0"]],
"acu_inter_flow": [],
"param_map": {"pooling": {"global_pooling": ["BOOL", "VALUE", True],
                          "type": ["STRING", "VALUE", 'AVG'],
                          "round_type": ["STRING", "VALUE", "floor"],
                          }},
"blob_map": {"pooling": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_qlinearglobalavgpool)

r_qlinearavgpool = {
"ruler_name": 'r_qlinearavgpool',
"src_ops_alias": ["QLinearAveragePool", "Constant", "Constant_1", "Constant_2", "Constant_3"],
"src_inter_flow": [["Constant:out0", "QLinearAveragePool:in1"],
                   ["Constant_1:out0", "QLinearAveragePool:in2"],
                   ["Constant_2:out0", "QLinearAveragePool:in3"],
                   ["Constant_3:out0", "QLinearAveragePool:in4"]],
"src_in_anchor": [["I_0:out0", "QLinearAveragePool:in0"]],
"src_out_tensor": ["QLinearAveragePool:out0"],
"acu_lys_alias": ["pooling"],
"src_acu_in_tensor_map": [["I_0:out0", "pooling:in0"]],
"src_acu_out_tensor_map": [["QLinearAveragePool:out0", "pooling:out0"]],
"acu_inter_flow": [],
"param_map": {
    "pooling":{
        "type": ["STRING", "VALUE", "AVG"],
        "pad_method": ["STRING", "VALUE", "padding_const"],
        "round_type": ["STRING", "CODE", "'ceil' if self.attr_pick(node['QLinearAveragePool'],"
                                         "'ceil_mode') == 1 else 'floor'"],
        "pad": ["INTS", "CODE", "[str(p) for p in self.array_layout(self.attr_pick(node['QLinearAveragePool'], 'pads',"
                                "[ 0, 0, 0, 0]), [0, 1])]"],
        "global_pooling": ["BOOL", "VALUE", False],
        "ksize_h": ["INT", "CODE", "self.attr_pick(node['QLinearAveragePool'], 'kernel_shape', [1])[0]"],
        "ksize_w": ["INT", "CODE", "self.attr_pick(node['QLinearAveragePool'], 'kernel_shape', [1])[1]"],
        "stride": ["INT", "CODE", "self.attr_pick(node['QLinearAveragePool'], 'strides', [1])[0]"],
        "padding": [
            "STRING",
            "CODE",
         "'SAME' if self.attr_pick(node['QLinearAveragePool'], 'auto_pad', 'NOTSET') in ['SAME_UPPER',"
         "'SAME_LOWER'] else 'VALID'"
        ],
}},
"blob_map": {"pooling": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_qlinearavgpool)

r_hard_swish_2 = {
"ruler_name": "r_hard_swish_2",
"src_ops_alias": ["Div", "Mul", "Constant", "Clip", "Add", "Constant_1", "Constant_2", "Constant_3"],
"src_inter_flow": [["Mul:out0", "Div:in0"], ["Constant:out0", "Div:in1"], ["Clip:out0", "Mul:in1"],
                   ["Add:out0", "Clip:in0"], ["Constant_1:out0", "Clip:in1"], ["Constant_2:out0", "Clip:in2"],
                   ["Constant_3:out0", "Add:in1"]],
"src_in_anchor": [["I_0:out0", "Add:in0"], ["I_0:out0", "Mul:in0"]],
"src_out_tensor": ["Div:out0"],
"acu_lys_alias": ["hard_swish"],
"src_acu_in_tensor_map": [["I_0:out0", "hard_swish:in0"]],
"src_acu_out_tensor_map": [["Div:out0", "hard_swish:out0"]],
"acu_inter_flow": [],
"param_map": {"hard_swish": {}},
"blob_map": {"hard_swish": {}},
"priority_tip": 0,
"pre_condition": "(self.tensor_to_numpy(tensor['Constant:out0']) == 6.0).all() and "\
                "(self.tensor_to_numpy(tensor['Constant_1:out0']) == 0.0).all() and "\
                "(self.tensor_to_numpy(tensor['Constant_2:out0']) == 6.0).all() and "\
                "(self.tensor_to_numpy(tensor['Constant_3:out0']) == 3.0).all()",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_hard_swish_2)

r_hard_swish_3 = {
"ruler_name": "r_hard_swish_3",
"src_ops_alias": ["Mul", "Div", "Clip", "Constant", "Add", "Constant_1", "Constant_2"],
"src_inter_flow": [["Div:out0", "Mul:in1"], ["Clip:out0", "Div:in0"],
                   ["Constant:out0", "Div:in1"], ["Add:out0", "Clip:in0"],
                   ["Constant_1:out0", "Clip:in1"], ["Constant:out0", "Clip:in2"],
                   ["Constant_2:out0", "Add:in1"]],
"src_in_anchor": [["I_0:out0", "Add:in0"], ["I_0:out0", "Mul:in0"]],
"src_out_tensor": ["Mul:out0"],
"acu_lys_alias": ["hard_swish"],
"src_acu_in_tensor_map": [["I_0:out0", "hard_swish:in0"]],
"src_acu_out_tensor_map": [["Mul:out0", "hard_swish:out0"]],
"acu_inter_flow": [],
"param_map": {"hard_swish": {}},
"blob_map": {"hard_swish": {}},
"priority_tip": 0,
"pre_condition": "(self.tensor_to_numpy(tensor['Constant:out0']) == 6.0).all() and "\
                "(self.tensor_to_numpy(tensor['Constant_1:out0']) == 0.0).all() and "\
                "(self.tensor_to_numpy(tensor['Constant_2:out0']) == 3.0).all()",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_hard_swish_3)

r_nonzero = {
"ruler_name": "r_nonzero",
"src_ops_alias": ["NonZero"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "NonZero:in0"]],
"src_out_tensor": ["NonZero:out0"],
"acu_lys_alias": ["nonzero"],
"src_acu_in_tensor_map": [["I:out0", "nonzero:in0"]],
"src_acu_out_tensor_map": [["NonZero:out0", "nonzero:out0"]],
"param_map": {"nonzero": {'output_type': ['STRING', 'CODE', "self.dtype_pick(tensor['NonZero:out0'])"]}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_nonzero)

r_one_hot = {
"ruler_name": "r_one_hot",
"src_ops_alias": ["OneHot", "Constant", "Constant_1"],
"src_inter_flow": [["Constant:out0", "OneHot:in1"], ["Constant_1:out0", "OneHot:in2"]],
"src_in_anchor": [["I_0:out0", "OneHot:in0"]],
"src_out_tensor": ["OneHot:out0"],
"acu_lys_alias": ["one_hot"],
"src_acu_in_tensor_map": [["I_0:out0", "one_hot:in0"]],
"src_acu_out_tensor_map": [["OneHot:out0", "one_hot:out0"]],
"acu_inter_flow": [],
"param_map": {
    "one_hot": {
        'depth': ['INT', 'CODE', "self.tensor_to_numpy(tensor['Constant:out0'])[0]"],
        'on_value': ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['Constant_1:out0'])[1]"],
        'off_value': ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['Constant_1:out0'])[0]"],
        'axis': ['INT', 'CODE', "self.attr_pick(node['OneHot'], 'axis', -1)"],
        'dtype': ['STRING', 'CODE', "self.tensor_to_numpy(tensor['Constant_1:out0'])[0].dtype"]
    }
},
"blob_map": {"one_hot": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]
}
ruler_list.append(r_one_hot)

r_shape = {
"ruler_name": "r_shape",
"src_ops_alias": ["Shape"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Shape:in0"]],
"src_out_tensor": ["Shape:out0"],
"acu_lys_alias": ["shapelayer"],
"src_acu_in_tensor_map": [["I:out0", "shapelayer:in0"]],
"src_acu_out_tensor_map": [["Shape:out0", "shapelayer:out0"]],
"param_map": {"shapelayer": {"out_type": ["STRING", "CODE", "self.dtype_pick(tensor['Shape:out0'])"]}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_shape)

r_sign = {
"ruler_name": "r_sign",
"src_ops_alias": ["Sign"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Sign:in0"]],
"src_out_tensor": ["Sign:out0"],
"acu_lys_alias": ["sign"],
"src_acu_in_tensor_map": [["I:out0", "sign:in0"]],
"src_acu_out_tensor_map": [["Sign:out0", "sign:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_sign)

r_size = {
"ruler_name": "r_size",
"src_ops_alias": ["Size"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Size:in0"]],
"src_out_tensor": ["Size:out0"],
"acu_lys_alias": ["size"],
"src_acu_in_tensor_map": [["I:out0", "size:in0"]],
"src_acu_out_tensor_map": [["Size:out0", "size:out0"]],
"param_map": {"size": {"out_type": ["STRING", "CODE", "self.dtype_pick(tensor['Size:out0'])"]}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_size)

r_scatter_nd = {
"ruler_name": "r_scatter_nd",
"src_ops_alias": ["ScatterND"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "ScatterND:in0"], ["I_1:out0", "ScatterND:in1"], ["I_2:out0", "ScatterND:in2"]],
"src_out_tensor": ["ScatterND:out0"],
"acu_lys_alias": ["scatter_nd_update"],
"src_acu_in_tensor_map": [["I:out0", "scatter_nd_update:in0"], ["I_1:out0", "scatter_nd_update:in1"],
                          ["I_2:out0", "scatter_nd_update:in2"]],
"src_acu_out_tensor_map": [["ScatterND:out0", "scatter_nd_update:out0"]],
"acu_inter_flow": [],
"param_map":{
    "scatter_nd_update":{
        'reduction':["STRING", "CODE", "self.attr_pick(node['ScatterND'], 'reduction', 'none')"],
    }
},
"blob_map": {},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_scatter_nd)

r_mean_variance_normalization = {
"ruler_name": "r_mean_variance_normalization",
"src_ops_alias": ["MeanVarianceNormalization"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "MeanVarianceNormalization:in0"]],
"src_out_tensor": ["MeanVarianceNormalization:out0"],
"acu_lys_alias": ["instancenormalize"],
"src_acu_in_tensor_map": [["I:out0", "instancenormalize:in0"]],
"src_acu_out_tensor_map": [["MeanVarianceNormalization:out0", "instancenormalize:out0"]],
"param_map":{
    "instancenormalize":{
        'axis':['INTS', 'CODE', "[p for p in self.attr_pick(node['MeanVarianceNormalization'], 'axes', [0, 2, 3])]"],
        'eps':['FLOAT', 'VALUE', 1e-9],
    }
},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": "self.attr_pick(node['MeanVarianceNormalization'], 'axes', [0, 2, 3]) == [0, 2, 3]",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_mean_variance_normalization)

r_cumsum = {
"ruler_name": "r_cumsum",
"src_ops_alias": ["CumSum", "Constant"],
"src_inter_flow": [["Constant:out0", "CumSum:in1"]],
"src_in_anchor": [["I:out0", "CumSum:in0"]],
"src_out_tensor": ["CumSum:out0"],
"acu_lys_alias": ["cumsum"],
"src_acu_in_tensor_map": [["I:out0", "cumsum:in0"]],
"src_acu_out_tensor_map": [["CumSum:out0", "cumsum:out0"]],
"acu_inter_flow": [],
"param_map":{
    'cumsum':
     {'axis':["INT","CODE","self.tensor_to_numpy(tensor['Constant:out0']).item()"],
      'exclusive': ['BOOL','CODE', "self.attr_pick(node['CumSum'], 'exclusive', 0)"],
      'reverse': ['BOOL','CODE', "self.attr_pick(node['CumSum'], 'reverse', 0)"]
      }
 },
"blob_map": {},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [11, -1]}
ruler_list.append(r_cumsum)

r_max_roi_pooling = {
"ruler_name": "r_roi_pooling",
"src_ops_alias": ["MaxRoiPool"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "MaxRoiPool:in0"], ["I_1:out0", "MaxRoiPool:in1"]],
"src_out_tensor": ["MaxRoiPool:out0"],
"acu_lys_alias": ["roipooling"],
"src_acu_in_tensor_map": [["I_0:out0", "roipooling:in0"], ["I_1:out0", "roipooling:in1"]],
"src_acu_out_tensor_map": [["MaxRoiPool:out0", "roipooling:out0"]],
"acu_inter_flow": [],
"param_map": {"roipooling": {
    'pooled_h':['INT', 'CODE', "self.shape_pick(tensor['MaxRoiPool:out0'])[2]"],
    'pooled_w': ['INT', 'CODE', "self.shape_pick(tensor['MaxRoiPool:out0'])[3]"],
    'spatial_scale': ['INT', 'CODE', "self.attr_pick(node['MaxRoiPool'], 'spatial_scale', 1)"],
    }
},
"blob_map": {},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_max_roi_pooling)
#MaxRoiPool:MaxRoiPool_Y;Constant:Initializer_rois

r_topk_10_const = {
"ruler_name": "r_topk_10_const",
"src_ops_alias": ["TopK", "Constant"],
"src_inter_flow": [["Constant:out0", "TopK:in1"]],
"src_in_anchor": [["I:out0", "TopK:in0"]],
"src_out_tensor": ["TopK:out0", "TopK:out1"],
"acu_lys_alias": ["topk","cast"],
"acu_inter_flow": [['topk:out1', 'cast:in0']],
"src_acu_in_tensor_map": [["I:out0", "topk:in0"]],
"src_acu_out_tensor_map": [["TopK:out0", "topk:out0"], ["TopK:out1", "cast:out0"]],
"param_map": {
    "topk":{
        "k": ['INT', "CODE", "self.tensor_to_numpy(tensor['Constant:out0']).item()"],
        'axis': ['INT', 'CODE', "self.attr_pick(node['TopK'], 'axis', -1)"],
    },
    "cast":{
        "out_data_type": ['STRING', 'VALUE', 'int64']
    }
},
"blob_map": {},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [10, -1]}
ruler_list.append(r_topk_10_const)

r_topk = {
"ruler_name": "r_topk",
"src_ops_alias": ["TopK"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "TopK:in0"]],
"src_out_tensor": ["TopK:out0", "TopK:out1"],
"acu_lys_alias": ["topk"],
"src_acu_in_tensor_map": [["I:out0", "topk:in0"]],
"src_acu_out_tensor_map": [["TopK:out0", "topk:out0"], ["TopK:out1", "topk:out1"]],
"param_map": {"topk":{"k": ['INT', "CODE", "self.attr_pick(node['TopK'], 'k', 1)"]}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, 9]}
ruler_list.append(r_topk)

r_layernormalize = {
"ruler_name": "r_layernormalize",
"src_ops_alias": ["Add", "Mul", "Constant", "Div", "Constant_1", "Sub", "Sqrt", "ReduceMean", "Add_1",
                  "ReduceMean_1", "Constant_2", "Pow", "Cast", "Constant_3"],
"src_inter_flow": [["Mul:out0", "Add:in0"], ["Constant:out0", "Add:in1"], ["Div:out0", "Mul:in0"],
                   ["Constant_1:out0", "Mul:in1"], ["Sub:out0", "Div:in0"], ["Sqrt:out0", "Div:in1"],
                   ["ReduceMean:out0", "Sub:in1"], ["Add_1:out0", "Sqrt:in0"], ["ReduceMean_1:out0", "Add_1:in0"],
                   ["Constant_2:out0", "Add_1:in1"], ["Pow:out0", "ReduceMean_1:in0"], ["Cast:out0", "Pow:in0"],
                   ["Constant_3:out0", "Pow:in1"], ["Sub:out0", "Cast:in0"]],
"src_in_anchor": [["I_0:out0", "ReduceMean:in0"], ["I_0:out0", "Sub:in0"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["layernormalize"],
"src_acu_in_tensor_map": [["I_0:out0", "layernormalize:in0"]],
"src_acu_out_tensor_map": [["Add:out0", "layernormalize:out0"]],
"acu_inter_flow": [],
"param_map": {
    "layernormalize": {
        "axis_list": ['INTS', 'CODE', "self.attr_pick(node['ReduceMean'], 'axes', [-1])"],
        "eps": ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['Constant_2:out0'])"],
    }
},
"blob_map": {
    "layernormalize": {
        "scale": ['CODE', "self.tensor_to_numpy(tensor['Constant_1:out0'])"],
        "bias": ['CODE', "self.tensor_to_numpy(tensor['Constant:out0'])"]
    }
},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_layernormalize)

r_layernormalize_1 = {
"ruler_name": "r_layernormalize_1",
"src_ops_alias": ["Add", "Mul", "Constant", "Div", "Constant_1", "Sub", "Sqrt", "ReduceMean", "Add_1",
                  "ReduceMean_1", "Constant_2", "Pow", "Constant_3"],
"src_inter_flow": [["Mul:out0", "Add:in0"], ["Constant:out0", "Add:in1"], ["Div:out0", "Mul:in0"],
                   ["Constant_1:out0", "Mul:in1"], ["Sub:out0", "Div:in0"], ["Sqrt:out0", "Div:in1"],
                   ["ReduceMean:out0", "Sub:in1"], ["Add_1:out0", "Sqrt:in0"], ["ReduceMean_1:out0", "Add_1:in0"],
                   ["Constant_2:out0", "Add_1:in1"], ["Pow:out0", "ReduceMean_1:in0"], ["Sub:out0", "Pow:in0"],
                   ["Constant_3:out0", "Pow:in1"]],
"src_in_anchor": [["I_0:out0", "ReduceMean:in0"], ["I_0:out0", "Sub:in0"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["layernormalize"],
"src_acu_in_tensor_map": [["I_0:out0", "layernormalize:in0"]],
"src_acu_out_tensor_map": [["Add:out0", "layernormalize:out0"]],
"acu_inter_flow": [],
"param_map": {
    "layernormalize": {
        "axis_list": ['INTS', 'CODE', "self.attr_pick(node['ReduceMean'], 'axes', [-1])"],
        "eps": ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['Constant_2:out0'])"],
    }
},
"blob_map": {
    "layernormalize": {
        "scale": ['CODE', "self.tensor_to_numpy(tensor['Constant_1:out0'])"],
        "bias": ['CODE', "self.tensor_to_numpy(tensor['Constant:out0'])"]
    }
},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_layernormalize_1)

r_layernormalize_2 = {
"ruler_name": "r_layernormalize_2",
"src_ops_alias": ["Add", "Div", "ReduceMean","Constant", "Mul", "Pow", "Constant_1", "Sub", "Add_1", "Constant_2",
                  "ReduceMean_1", "Constant_3", "Pow_1", "Sub_1", "Constant_4"],
"src_inter_flow": [["Div:out0", "Add:in0"], ["Constant:out0", "Add:in1"], ["Mul:out0", "Div:in0"],
                   ["Pow:out0", "Div:in1"], ["Constant_1:out0", "Mul:in0"], ["Sub:out0", "Mul:in1"],
                   ["Add_1:out0", "Pow:in0"], ["Constant_2:out0", "Pow:in1"], ["ReduceMean_1:out0", "Add_1:in0"],
                   ["Constant_3:out0", "Add_1:in1"], ["Pow_1:out0", "ReduceMean_1:in0"], ["Sub_1:out0", "Pow_1:in0"],
                   ["Constant_4:out0", "Pow_1:in1"],["ReduceMean:out0", "Sub:in1"], ["ReduceMean:out0", "Sub_1:in1"]],
"src_in_anchor": [["I_0:out0", "Sub_1:in0"], ["I_0:out0", "Sub:in0"], ["I_0:out0", "ReduceMean:in0"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["layernormalize"],
"src_acu_in_tensor_map": [["I_0:out0", "layernormalize:in0"]],
"src_acu_out_tensor_map": [["Add:out0", "layernormalize:out0"]],
"acu_inter_flow": [],
"param_map": {
    "layernormalize": {
        "axis_list": ['INTS', 'CODE', "self.attr_pick(node['ReduceMean'], 'axes', [-1])"],
        "eps": ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['Constant_3:out0'])"],
    }
},
"blob_map": {
    "layernormalize": {
        "scale": ['CODE', "self.tensor_to_numpy(tensor['Constant_1:out0'])"],
        "bias": ['CODE', "self.tensor_to_numpy(tensor['Constant:out0'])"]
    }
},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_layernormalize_2)

r_layernormalize_3 = {
"ruler_name": "r_layernormalize_3",
"src_ops_alias": ["GlobalAveragePool", "Sub", "Mul", "GlobalAveragePool_1", "Add_1", "Constant","Sqrt",
                  "Reciprocal","Mul_1", "Constant_1", "Mul_2", "Mul_3", "Sub_1", "Constant_2", "Add_2"],
"src_inter_flow": [["GlobalAveragePool:out0", "Sub:in1"], ["Sub:out0", "Mul:in0"], ["Sub:out0", "Mul:in1"],
                   ["Mul:out0", "GlobalAveragePool_1:in0"], ["GlobalAveragePool_1:out0", "Add_1:in0"],
                   ["Constant:out0", "Add_1:in1"], ["Add_1:out0", "Sqrt:in0"], ["Sqrt:out0", "Reciprocal:in0"],
                   ["Reciprocal:out0", "Mul_1:in0"], ["Constant_1:out0", "Mul_1:in1"], ["Mul_1:out0", "Mul_2:in1"],
                   ["GlobalAveragePool:out0", "Mul_2:in0"],["Mul_1:out0", "Mul_3:in1"], ["Mul_2:out0", "Sub_1:in1"],
                   ["Constant_2:out0", "Sub_1:in0"], ["Sub_1:out0", "Add_2:in1"], ["Mul_3:out0", "Add_2:in0"]],
"src_in_anchor": [["I_0:out0", "GlobalAveragePool:in0"], ["I_0:out0", "Sub:in0"], ["I_0:out0", "Mul_3:in0"]],
"src_out_tensor": ["Add_2:out0"],
"acu_lys_alias": ["layernormalize"],
"src_acu_in_tensor_map": [["I_0:out0", "layernormalize:in0"]],
"src_acu_out_tensor_map": [["Add_2:out0", "layernormalize:out0"]],
"param_map": {
    "layernormalize": {
        "eps": ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['Constant:out0'])"],
    }
},
"blob_map": {
    "layernormalize": {
        "scale": ['CODE', "self.tensor_to_numpy(tensor['Constant_1:out0'])"],
        "bias": ['CODE', "self.tensor_to_numpy(tensor['Constant_2:out0'])"]
    }
},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_layernormalize_3)

r_layernormalize_4 = {
"ruler_name": "r_layernormalize_4",
"src_ops_alias": ["ReduceMean", "Sub", "Mul", "ReduceMean_1", "Add_1", "Constant","Sqrt", "Reciprocal",
                  "Mul_1", "Constant_1", "Mul_2", "Mul_3", "Sub_1", "Constant_2", "Add_2"],
"src_inter_flow": [["ReduceMean:out0", "Sub:in1"],
                   ["Sub:out0", "Mul:in0"], ["Sub:out0", "Mul:in1"],["Mul:out0", "ReduceMean_1:in0"],
                   ["ReduceMean_1:out0", "Add_1:in0"],["Constant:out0", "Add_1:in1"], ["Add_1:out0", "Sqrt:in0"],
                   ["Sqrt:out0", "Reciprocal:in0"], ["Reciprocal:out0", "Mul_1:in0"], ["Constant_1:out0", "Mul_1:in1"],
                   ["Mul_1:out0", "Mul_2:in1"],  ["ReduceMean:out0", "Mul_2:in0"],["Mul_1:out0", "Mul_3:in1"],
                   ["Mul_2:out0", "Sub_1:in1"], ["Constant_2:out0", "Sub_1:in0"], ["Sub_1:out0", "Add_2:in1"],
                   ["Mul_3:out0", "Add_2:in0"]],
"src_in_anchor": [["I_0:out0", "ReduceMean:in0"], ["I_0:out0", "Sub:in0"], ["I_0:out0", "Mul_3:in0"]],
"src_out_tensor": ["Add_2:out0"],
"acu_lys_alias": ["layernormalize"],
"src_acu_in_tensor_map": [["I_0:out0", "layernormalize:in0"]],
"src_acu_out_tensor_map": [["Add_2:out0", "layernormalize:out0"]],
"param_map": {
    "layernormalize": {
        "axis_list": ['INTS', 'CODE', "self.attr_pick(node['ReduceMean'], 'axes', [1])"],
        "eps": ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['Constant:out0'])"],
    }
},
"blob_map": {
    "layernormalize": {
        "scale": ['CODE', "self.tensor_to_numpy(tensor['Constant_1:out0'])"],
        "bias": ['CODE', "self.tensor_to_numpy(tensor['Constant_2:out0'])"]
    }
},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_layernormalize_4)

r_layernormalize_5 = {
"ruler_name": "r_layernormalize_5",
"src_ops_alias": ["LayerNormalization", "Constant_0", "Constant_1"],
"src_inter_flow": [ ["Constant_1:out0", "LayerNormalization:in2"], ["Constant_0:out0", "LayerNormalization:in1"]],
"src_in_anchor": [["I:out0", "LayerNormalization:in0"]],
"src_out_tensor": ["LayerNormalization:out0"],
"acu_lys_alias": ["layernormalize"],
"src_acu_in_tensor_map": [["I:out0", "layernormalize:in0"]],
"src_acu_out_tensor_map": [["LayerNormalization:out0", "layernormalize:out0"]],
"param_map": {
    "layernormalize":
        {"axis_list": ['INTS', 'CODE', "self.layernorm_axis_param(self.shape_pick(tensor['LayerNormalization:out0']),"
                                       "self.attr_pick(node['LayerNormalization'], 'axis', [-1]))"],
         "eps": ["FLOAT", "CODE", "self.attr_pick(node['LayerNormalization'], 'epsilon', 1e-5)"]}
},
"blob_map": {
    "layernormalize": {
        "scale": ['CODE', "self.tensor_to_numpy(tensor['Constant_0:out0'])"],
        "bias": ['CODE', "self.tensor_to_numpy(tensor['Constant_1:out0'])"]
    }
},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [17, -1]}
ruler_list.append(r_layernormalize_5)

r_layernormalize_6 = {
"ruler_name": "r_layernormalize_6",
"src_ops_alias": ["Add", "Div", "Constant", "Mul", "Add_1", "Constant_1", "Sub", "Sqrt", "Constant_2",
                  "ReduceMean", "Div_1", "Mul_1", "Constant_3", "ReduceMean_1", "Constant_4", "Mul_2",
                  "Sub_1", "ReduceMean_2"],
"src_inter_flow": [["Div:out0", "Add:in0"], ["Constant:out0", "Add:in1"], ["Mul:out0", "Div:in0"],
                   ["Add_1:out0", "Div:in1"], ["Constant_1:out0", "Mul:in0"], ["Sub:out0", "Mul:in1"],
                   ["Sqrt:out0", "Add_1:in0"], ["Constant_2:out0", "Add_1:in1"], ["ReduceMean:out0", "Sub:in1"],
                   ["Div_1:out0", "Sqrt:in0"], ["Mul_1:out0", "Div_1:in0"], ["Constant_3:out0", "Div_1:in1"],
                   ["ReduceMean_1:out0", "Mul_1:in0"], ["Constant_4:out0", "Mul_1:in1"],
                   ["Mul_2:out0", "ReduceMean_1:in0"], ["Sub_1:out0", "Mul_2:in0"], ["Sub_1:out0", "Mul_2:in1"],
                   ["ReduceMean_2:out0", "Sub_1:in1"]],
"src_in_anchor": [["I_0:out0", "ReduceMean:in0"], ["I_0:out0", "Sub:in0"], ["I_0:out0", "ReduceMean_2:in0"],
                  ["I_0:out0", "Sub_1:in0"]],
"src_out_tensor": ["Add:out0"],
"acu_lys_alias": ["layernormalize"],
"src_acu_in_tensor_map": [["I_0:out0", "layernormalize:in0"]],
"src_acu_out_tensor_map": [["Add:out0", "layernormalize:out0"]],
"acu_inter_flow": [],
"param_map": {
    "layernormalize":
        {"axis_list": ['INTS', 'CODE', "self.attr_pick(node['ReduceMean'], 'axes', [-1])"],
         "eps": ['FLOAT', 'CODE', "self.tensor_to_numpy(tensor['Constant_2:out0'])*"
                                  "self.tensor_to_numpy(tensor['Constant_3:out0'])/"
                                  "self.tensor_to_numpy(tensor['Constant_4:out0'])"]}},
"blob_map": {
    "layernormalize": {
        "scale": ['CODE', "self.tensor_to_numpy(tensor['Constant_1:out0'])"],
        "bias": ['CODE', "self.tensor_to_numpy(tensor['Constant:out0'])"]
    }},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_layernormalize_6)


r_gelu = {
"ruler_name": "r_gelu",
"src_ops_alias": ["Mul", "Mul_1", "Constant", "Add", "Erf", "Constant_1", "Div", "Constant_2"],
"src_inter_flow": [["Mul_1:out0", "Mul:in0"], ["Constant:out0", "Mul:in1"],
["Add:out0", "Mul_1:in1"], ["Erf:out0", "Add:in0"], ["Constant_1:out0", "Add:in1"],
["Div:out0", "Erf:in0"],
["Constant_2:out0", "Div:in1"]],
"src_in_anchor": [["I_0:out0", "Div:in0"], ["I_0:out0", "Mul_1:in0"]],
"src_out_tensor": ["Mul:out0"],
"acu_lys_alias": ["gelu"],
"src_acu_in_tensor_map": [["I_0:out0", "gelu:in0"]],
"src_acu_out_tensor_map": [["Mul:out0", "gelu:out0"]],
"acu_inter_flow": [],
"param_map": {"gelu": {'approximate':['BOOL', 'VALUE', False]}},
"blob_map": {"gelu": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_gelu)

r_gelu_1 = {
"ruler_name": "r_gelu_1",
"src_ops_alias": ["Mul", "Mul_1", "Constant", "Add", "Constant_1", "Tanh", "Mul_2",
                  "Constant_2", "Add_1", "Mul_3", "Constant_3", "Pow", "Constant_4"],
"src_inter_flow": [["Mul_1:out0", "Mul:in1"], ["Constant:out0", "Mul_1:in0"], ["Add:out0", "Mul_1:in1"],
                   ["Constant_1:out0", "Add:in0"], ["Tanh:out0", "Add:in1"], ["Mul_2:out0", "Tanh:in0"],
                   ["Constant_2:out0", "Mul_2:in0"], ["Add_1:out0", "Mul_2:in1"], ["Mul_3:out0", "Add_1:in1"],
                   ["Constant_3:out0", "Mul_3:in0"], ["Pow:out0", "Mul_3:in1"], ["Constant_4:out0", "Pow:in1"]],
"src_in_anchor": [["I_0:out0", "Add_1:in0"], ["I_0:out0", "Pow:in0"], ["I_0:out0", "Mul:in0"]],
"src_out_tensor": ["Mul:out0"],
"acu_lys_alias": ["gelu"],
"src_acu_in_tensor_map": [["I_0:out0", "gelu:in0"]],
"src_acu_out_tensor_map": [["Mul:out0", "gelu:out0"]],
"acu_inter_flow": [],
"param_map": {"gelu": {'approximate':['BOOL', 'VALUE', False]}},
"blob_map": {"gelu": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_gelu_1)

r_gelu_2 = {
"ruler_name": "r_gelu",
"src_ops_alias": ["Mul", "Mul_1", "Add", "Constant", "Erf", "Constant_1", "Div", "Constant_2"],
"src_inter_flow": [["Mul_1:out0", "Mul:in0"], ["Add:out0", "Mul:in1"],
["Constant:out0", "Mul_1:in1"], ["Erf:out0", "Add:in0"],
["Constant_1:out0", "Add:in1"], ["Div:out0", "Erf:in0"],
["Constant_2:out0", "Div:in1"]],
"src_in_anchor": [["I_0:out0", "Mul_1:in0"], ["I_0:out0", "Div:in0"]],
"src_out_tensor": ["Mul:out0"],
"acu_lys_alias": ["gelu"],
"src_acu_in_tensor_map": [["I_0:out0", "gelu:in0"]],
"src_acu_out_tensor_map": [["Mul:out0", "gelu:out0"]],
"acu_inter_flow": [],
"param_map": {"gelu": {'approximate':['BOOL', 'VALUE', False]}},
"blob_map": {"gelu": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_gelu_2)

r_mish = {
"ruler_name": "r_mish",
"src_ops_alias": ["Mul", "Tanh", "Softplus"],
"src_inter_flow": [["Tanh:out0", "Mul:in1"], ["Softplus:out0", "Tanh:in0"]],
"src_in_anchor": [["I_0:out0", "Softplus:in0"], ["I_0:out0", "Mul:in0"]],
"src_out_tensor": ["Mul:out0"],
"acu_lys_alias": ["mish"],
"src_acu_in_tensor_map": [["I_0:out0", "mish:in0"]],
"src_acu_out_tensor_map": [["Mul:out0", "mish:out0"]],
"acu_inter_flow": [],
"param_map": {"mish": {}},
"blob_map": {"mish": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_mish)

r_mish_1 = {
"ruler_name": "r_mish_1",
"src_ops_alias": ["Mul", "Tanh", "Log", "Add", "Constant", "Exp"],
"src_inter_flow": [["Tanh:out0", "Mul:in1"], ["Log:out0", "Tanh:in0"],
                   ["Add:out0", "Log:in0"], ["Constant:out0", "Add:in0"], ["Exp:out0", "Add:in1"]],
"src_in_anchor": [["I_0:out0", "Exp:in0"], ["I_0:out0", "Mul:in0"]],
"src_out_tensor": ["Mul:out0"],
"acu_lys_alias": ["mish"],
"src_acu_in_tensor_map": [["I_0:out0", "mish:in0"]],
"src_acu_out_tensor_map": [["Mul:out0", "mish:out0"]],
"acu_inter_flow": [],
"param_map": {"mish": {}},
"blob_map": {"mish": {}},
"priority_tip": 0,
"pre_condition": "(self.tensor_to_numpy(tensor['Constant:out0']) == 1.0).all()",
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_mish_1)

r_mish_2 = {
"ruler_name": "r_mish_2",
"src_ops_alias": ["Mish"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "Mish:in0"]],
"src_out_tensor": ["Mish:out0"],
"acu_lys_alias": ["mish"],
"src_acu_in_tensor_map": [["I_0:out0", "mish:in0"]],
"src_acu_out_tensor_map": [["Mish:out0", "mish:out0"]],
"acu_inter_flow": [],
"param_map": {"mish": {}},
"blob_map": {"mish": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [18, -1]
}
ruler_list.append(r_mish_2)

r_reshape = {
"ruler_name": "r_reshape",
"src_ops_alias": ["Reshape", "Constant_0"],
"src_inter_flow": [["Constant_0:out0", "Reshape:in1"]],
"src_in_anchor": [["I_0:out0", "Reshape:in0"]],
"src_out_tensor": ["Reshape:out0"],
"acu_lys_alias": ["reshape"],
"src_acu_in_tensor_map": [["I_0:out0", "reshape:in0"]],
"src_acu_out_tensor_map": [["Reshape:out0", "reshape:out0"]],
"param_map":
{"reshape":{"shape":["INTS", "CODE", "self.tensor_to_numpy(tensor['Constant_0:out0'])",]}},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_reshape)

r_acosh = {
"ruler_name": "r_acosh",
"src_ops_alias": ["Acosh"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Acosh:in0"]],
"src_out_tensor": ["Acosh:out0"],
"acu_lys_alias": ["acosh"],
"src_acu_in_tensor_map": [["I:out0", "acosh:in0"]],
"src_acu_out_tensor_map": [["Acosh:out0", "acosh:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_acosh)

r_cos = {
"ruler_name": "r_cos",
"src_ops_alias": ["Cos"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "Cos:in0"]],
"src_out_tensor": ["Cos:out0"],
"acu_lys_alias": ["cos"],
"src_acu_in_tensor_map": [["I:out0", "cos:in0"]],
"src_acu_out_tensor_map": [["Cos:out0", "cos:out0"]],
"param_map": {},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_cos)

r_dft = {
"ruler_name": "r_dft",
"src_ops_alias": ["DFT"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "DFT:in0"]],
"src_out_tensor": ["DFT:out0"],
"acu_lys_alias": ["dft"],
"src_acu_in_tensor_map": [["I_0:out0", "dft:in0"]],
"src_acu_out_tensor_map": [["DFT:out0", "dft:out0"]],
"acu_inter_flow": [],
"param_map": {},
"blob_map": {},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [17, -1]
}
ruler_list.append(r_dft)

r_stft = {
"ruler_name": "r_stft",
"src_ops_alias": ["STFT"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "STFT:in0"], ["I_1:out0", "STFT:in1"],
                  ["I_2:out0", "STFT:in2"], ["I_3:out0", "STFT:in3"]],
"src_out_tensor": ["STFT:out0"],
"acu_lys_alias": ["stft"],
"src_acu_in_tensor_map": [["I:out0", "stft:in0"], ["I_1:out0", "stft:in1"],
                          ["I_2:out0", "stft:in2"], ["I_3:out0", "stft:in3"]],
"src_acu_out_tensor_map": [["STFT:out0", "stft:out0"]],
"acu_inter_flow": [],
"param_map": {"stft":{
        'onesided':["INT", "CODE", "self.attr_pick(node['STFT'], 'onesided', 1)"],
    }},
"blob_map": {},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [17, -1]
}
ruler_list.append(r_stft)

r_hannwindow = {
"ruler_name": "r_hannwindow",
"src_ops_alias": ["HannWindow"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "HannWindow:in0"]],
"src_out_tensor": ["HannWindow:out0"],
"acu_lys_alias": ["hannwindow"],
"src_acu_in_tensor_map": [["I_0:out0", "hannwindow:in0"]],
"src_acu_out_tensor_map": [["HannWindow:out0", "hannwindow:out0"]],
"acu_inter_flow": [],
"param_map": {
"hannwindow":
        {"periodic": ['INT', 'CODE', "self.attr_pick(node['HannWindow'], 'periodic', 1)"]}
},
"blob_map": {},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [17, -1]
}
ruler_list.append(r_hannwindow)

r_hammingwindow = {
"ruler_name": "r_hammingwindow",
"src_ops_alias": ["HammingWindow"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "HammingWindow:in0"]],
"src_out_tensor": ["HammingWindow:out0"],
"acu_lys_alias": ["hammingwindow"],
"src_acu_in_tensor_map": [["I_0:out0", "hammingwindow:in0"]],
"src_acu_out_tensor_map": [["HammingWindow:out0", "hammingwindow:out0"]],
"acu_inter_flow": [],
"param_map": {
"hannwindow":
        {"periodic": ['INT', 'CODE', "self.attr_pick(node['HammingWindow'], 'periodic', 1)"]}
},
"blob_map": {},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [17, -1]
}
ruler_list.append(r_hammingwindow)

r_gridsample = {
"ruler_name": "r_grid_samples",
"src_ops_alias": ["GridSample"],
"src_inter_flow": [],
"src_in_anchor": [["I:out0", "GridSample:in0"], ["I_1:out0", "GridSample:in1"]],
"src_out_tensor": ["GridSample:out0"],
"acu_lys_alias": ["gridsample"],
"src_acu_in_tensor_map": [["I:out0", "gridsample:in0"], ["I_1:out0", "gridsample:in1"]],
"src_acu_out_tensor_map": [["GridSample:out0", "gridsample:out0"]],
"param_map": {"gridsample": {
    "mode": ['STRING', 'CODE', "self.attr_pick(node['GridSample'], 'mode', 'bilinear')"],
    "align_corners": ['BOOL', 'CODE', "self.attr_pick(node['GridSample'], 'align_corners', False)"],
    "padding_mode": ['STRING', 'CODE', "self.attr_pick(node['GridSample'], 'padding_mode', 'zeros')"],
},},
"blob_map": {},
"acu_inter_flow": [],
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [1, -1]}
ruler_list.append(r_gridsample)

r_group_norm = {
"ruler_name": 'r_group_norm',
"src_ops_alias": ["GroupNormalization", "Constant", "Constant_1"],
"src_inter_flow": [["Constant:out0", "GroupNormalization:in1"], ["Constant_1:out0", "GroupNormalization:in2"]],
"src_in_anchor": [["I_0:out0", "GroupNormalization:in0"]],
"src_out_tensor": ["GroupNormalization:out0"],
"acu_lys_alias": ["groupnormalize"],
"src_acu_in_tensor_map": [["I_0:out0", "groupnormalize:in0"]],
"src_acu_out_tensor_map": [["GroupNormalization:out0", "groupnormalize:out0"]],
"acu_inter_flow": [],
"param_map": {"groupnormalize": {'eps': ['FLOAT', 'CODE',
                                         "self.attr_pick(node['GroupNormalization'], 'epsilon', 1e-5)"],
                                 'num_groups':
                                     ['INT', 'CODE', "self.attr_pick(node['GroupNormalization'], 'num_groups', 1)"]}},
"blob_map": {"groupnormalize": {'scale': ['CODE', "self.tensor_to_numpy(tensor['Constant:out0'])"],
                                'bias': ['CODE', "self.tensor_to_numpy(tensor['Constant_1:out0'])"]}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [18, -1]
}
ruler_list.append(r_group_norm)

r_bitwise_and = {
"ruler_name": "r_bitwise_and",
"src_ops_alias": ["BitwiseAnd"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "BitwiseAnd:in0"], ["I_1:out0", "BitwiseAnd:in1"]],
"src_out_tensor": ["BitwiseAnd:out0"],
"acu_lys_alias": ["bitwise_and"],
"src_acu_in_tensor_map": [["I_0:out0", "bitwise_and:in0"], ["I_1:out0", "bitwise_and:in1"]],
"src_acu_out_tensor_map": [["BitwiseAnd:out0", "bitwise_and:out0"]],
"acu_inter_flow": [],
"param_map": {},
"blob_map": {},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [18, -1]
}
ruler_list.append(r_bitwise_and)

r_bitwise_or = {
"ruler_name": "r_bitwise_or",
"src_ops_alias": ["BitwiseOr"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "BitwiseOr:in0"], ["I_1:out0", "BitwiseOr:in1"]],
"src_out_tensor": ["BitwiseOr:out0"],
"acu_lys_alias": ["bitwise_or"],
"src_acu_in_tensor_map": [["I_0:out0", "bitwise_or:in0"], ["I_1:out0", "bitwise_or:in1"]],
"src_acu_out_tensor_map": [["BitwiseOr:out0", "bitwise_or:out0"]],
"acu_inter_flow": [],
"param_map": {},
"blob_map": {},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [18, -1]
}
ruler_list.append(r_bitwise_or)

r_bitwise_xor = {
"ruler_name": "r_bitwise_xor",
"src_ops_alias": ["BitwiseXor"],
"src_inter_flow": [],
"src_in_anchor": [["I_0:out0", "BitwiseXor:in0"], ["I_1:out0", "BitwiseXor:in1"]],
"src_out_tensor": ["BitwiseXor:out0"],
"acu_lys_alias": ["bitwise_xor"],
"src_acu_in_tensor_map": [["I_0:out0", "bitwise_xor:in0"], ["I_1:out0", "bitwise_xor:in1"]],
"src_acu_out_tensor_map": [["BitwiseXor:out0", "bitwise_xor:out0"]],
"acu_inter_flow": [],
"param_map": {},
"blob_map": {},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [18, -1]
}
ruler_list.append(r_bitwise_xor)

r_center_crop_pad = {
"ruler_name": "r_center_crop_pad",
"src_ops_alias": ["CenterCropPad", "Constant"],
"src_inter_flow": [["Constant:out0", "CenterCropPad:in1"]],
"src_in_anchor": [["I_0:out0", "CenterCropPad:in0"]],
"src_out_tensor": ["CenterCropPad:out0"],
"acu_lys_alias": ["center_crop_pad"],
"src_acu_in_tensor_map": [["I_0:out0", "center_crop_pad:in0"]],
"src_acu_out_tensor_map": [["CenterCropPad:out0", "center_crop_pad:out0"]],
"acu_inter_flow": [],
"param_map": {"center_crop_pad": {"shape": ["INTS", "PYFUNC", r_center_crop_pad_shape(in_tensor='I_0:out0')]}},
"blob_map": {"center_crop_pad": {}},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [18, -1]
}
ruler_list.append(r_center_crop_pad)

r_col2im = {
"ruler_name": "r_col2im",
"src_ops_alias": ["Col2Im", "Constant", "Constant_1"],
"src_inter_flow": [["Constant:out0", "Col2Im:in1"], ["Constant_1:out0", "Col2Im:in2"]],
"src_in_anchor": [["I_0:out0", "Col2Im:in0"]],
"src_out_tensor": ["Col2Im:out0"],
"acu_lys_alias": ["col2im"],
"src_acu_in_tensor_map": [["I_0:out0", "col2im:in0"]],
"src_acu_out_tensor_map": [["Col2Im:out0", "col2im:out0"]],
"acu_inter_flow": [],
"param_map": {"col2im": {'image_shape': ["INTS", "CODE", "self.tensor_to_numpy(tensor['Constant:out0']).tolist()"],
                         'block_shape': ["INTS", "CODE", "self.tensor_to_numpy(tensor['Constant_1:out0']).tolist()"],
                         'pads': ["ORIGIN", "PYFUNC", r_col2im_pads()],
                         'strides': ["INTS", "CODE", "self.attr_pick(node['Col2Im'], 'strides', [1, 1])"],
                         'dilations': ["INTS", "CODE", "self.attr_pick(node['Col2Im'], 'dilations', [1, 1])"]}},
"blob_map": {},
"priority_tip": 0,
"pre_condition": None,
"src_ops_main_version": None,
"src_ops_minior_version": [18, -1]
}
ruler_list.append(r_col2im)


def gen_onnx_ruler(dst_path):
    # print(json.dumps(ruler_list))
    dst_path = os.path.join(dst_path, 'onnx_ruler_db.json')

    with open(dst_path, 'w+') as f:
        json.dump(ruler_list, f, indent=1)

    # To Verify ruler follow synatx
    with open(dst_path, 'r') as f:
        x_val = json.load(f)
def main():
    gen_onnx_ruler(sys.argv[1])

if  __name__ == '__main__':
    main()

