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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore
from acuitylib.acuitylog import AcuityLog as al

def get_pad_shape(auto_pad,  # type: str
                  input_spatial_shape,  # type: list[int]
                  kernel_spatial_shape,  # type: list[int]
                  strides_spatial,  # type: list[int]
                  dilations_spatial,  # type: list[int]
                  output_spatial_shape,  # type: list[int]
                  spatial_pad, #type: list[int]
                  ):  # type: (...) -> list[int]
    spatial_rank = len(input_spatial_shape)
    pad_shape = [[0, 0]] * spatial_rank
    pad_shape = np.array(pad_shape)
    if auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
        for i in range(spatial_rank):
            pad_size = (output_spatial_shape[i] - 1) * strides_spatial[i] + (
                        (kernel_spatial_shape[i] - 1) * dilations_spatial[i] + 1) - input_spatial_shape[i]
            pad_shape[i][0] = pad_size // 2 if auto_pad == 'SAME_UPPER' else pad_size - pad_size // 2
            pad_shape[i][1] = pad_size - pad_size // 2 if auto_pad == 'SAME_UPPER' else pad_size // 2
    else:
        for i in range(spatial_rank):
            pad_shape[i][0] = spatial_pad[i]
            pad_shape[i][1] = spatial_pad[i + spatial_rank]
    return pad_shape.tolist()


def get_output_shape(auto_pad,  # type: str
                     input_spatial_shape,  # type: list[int]
                     kernel_spatial_shape,  # type: list[int]
                     strides_spatial,  # type: list[int]
                     dilations,  # type: list[int]
                     spatial_pad, # type: list[int]
                     ceil_mode=False, # type:bool
                     ):  # type: (...) -> list[int]
    out_shape = [0] * len(input_spatial_shape)
    round_func = np.ceil if ceil_mode else np.floor
    spatial_rank = len(input_spatial_shape)
    if auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
        for i in range(len(input_spatial_shape)):
            out_shape[i] = int(np.ceil(float(input_spatial_shape[i]) / float(strides_spatial[i])))
    elif auto_pad in ('VALID', 'NOTSET'):
        for i in range(spatial_rank):
            out_shape[i] = int(round_func(float(
                input_spatial_shape[i] + spatial_pad[i] + spatial_pad[i + spatial_rank] - kernel_spatial_shape[i]) / float(strides_spatial[i])) + 1)
            if (out_shape[i] - 1) * strides_spatial[i] >= input_spatial_shape[i] + spatial_pad[i]:
                out_shape[i] = out_shape[i] - 1
    return out_shape

def pool(x,  # type: np.ndarray
         x_shape,  # type: Sequence[int]
         kernel_shape,  # type: Sequence[int]
         strides_shape,  # type: Sequence[int]
         out_shape,  # type: Sequence[int]
         pad_shape,  # type: Sequence[int]
         pooling_type,  # type: Text
         count_include_pad=0,  # type: int
         ):  # type: (...) -> np.ndarray
    spatial_size = len(x_shape) - 2
    y = np.zeros([x_shape[0], x_shape[1]] + list(out_shape))

    if pooling_type == 'AVG':
        f = np.average
    elif pooling_type == 'MAX':
        f = np.max
    elif pooling_type.startswith('LP'):
        p = int(pooling_type.replace('LP', ''))
        axis = tuple([i for i in range(2, 2+spatial_size)])
        f = lambda x: np.linalg.norm(x=x, ord=p)
    else:
        raise NotImplementedError(
            'Pooling type {} does not support. Should be AVG, MAX'.format(pooling_type))

    for location in np.ndindex(y.shape):
        n,c = location[0], location[1]
        spatial_index = location[2:]
        index_arg = [n, c]
        for s_r in range(spatial_size):
            l = spatial_index[s_r]
            if pooling_type == 'MAX':
                start = l * strides_shape[s_r] - pad_shape[s_r][0]
                end = min(start + kernel_shape[s_r], x_shape[s_r + 2])
            elif pooling_type == 'AVG' or 'LP' in pooling_type:
                if count_include_pad == 0:
                    start = l * strides_shape[s_r] - pad_shape[s_r][0]
                    end = min(start + kernel_shape[s_r], x_shape[s_r + 2] + pad_shape[s_r][1])
                    end = min(end, x_shape[s_r + 2])
                else:
                    start = l * strides_shape[s_r]
                    end = min(start + kernel_shape[s_r], x_shape[s_r + 2] + pad_shape[s_r][0] + pad_shape[s_r][1])
                    # end = min(end, x_shape[s_r + 2])
            start = max(start, 0)
            if start != end:
                index_arg.append((start, end))
            else:
                index_arg.append((start, start+1))
        if spatial_size == 1:
            to_process_data = x[n, c, index_arg[2][0]:index_arg[2][1]]
        elif spatial_size == 2:
            to_process_data = x[n, c, index_arg[2][0]:index_arg[2][1], index_arg[3][0]:index_arg[3][1]]
        else:
            index_arg_str = str(n) + ',' + str(c) + ',' + ','.join('{}:{}'.format(s,e) for s,e in index_arg[2:] )
            to_process_data = eval('x[{}]'.format(index_arg_str))

        y[location] = f(to_process_data)

    return y.astype(np.float32)


def MaxPool(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    x = inputs[0]
    x_shape = np.shape(x)
    spatial_rank = len(x_shape) - 2
    kernel_shape = attr['kernel_shape']
    strides = attr.get('strides', [1]*spatial_rank)
    dilations = attr.get('dilations', [1] * spatial_rank)
    ceil_mode = attr.get('ceil_mode', 0) == 1
    spatial_pad = attr.get('pads', [0]*2*spatial_rank)
    store_order = attr.get('storage_order', 0)
    if store_order == 1:
        al.e('Unsupport storage_order is 1')
    store_location = True if len(outputs) == 2 else False
    # TODO: need handle this case.
    out_spatial_shape = get_output_shape(attr.get('auto_pad', 'VALID', ), x_shape[2:], kernel_shape, strides,
                                         dilations=dilations, ceil_mode=ceil_mode, spatial_pad=spatial_pad)
    spatial_pads = get_pad_shape(attr.get('auto_pad', 'VALID'), x_shape[2:], kernel_shape, strides, dilations,
                                 out_spatial_shape, spatial_pad)
    # pads = [(0, 0), (0, 0)] + spatial_pads
    # padded = np.pad(x, pads, mode='constant', constant_values=np.nan)
    if 'infer_shape' in attr:
        shape = list(x_shape[0:2])
        shape.extend(out_spatial_shape)
        return np.ones(shape, x.dtype)

    np_pads = [[0,0],[0,0]] + spatial_pads
    x = np.pad(x, np_pads, 'constant', constant_values=-np.inf)
    torch_pad = [0] * spatial_rank

    import torch
    max_pool_model = None
    if spatial_rank == 1:
        max_pool_model = torch.nn.MaxPool1d(kernel_shape, strides, torch_pad, dilations, store_location, ceil_mode)
    elif spatial_rank == 2:
        max_pool_model = torch.nn.MaxPool2d(kernel_shape, strides, torch_pad, dilations, store_location, ceil_mode)
    elif spatial_rank == 3:
        max_pool_model = torch.nn.MaxPool3d(kernel_shape, strides, torch_pad, dilations, store_location, ceil_mode)
    else:
        al.e("Not support {} dim max pool".format(str(spatial_rank)))

    in_torch_tensor = torch.tensor(x, requires_grad=False)

    with torch.no_grad():
        res_torch_tensor = max_pool_model(in_torch_tensor)

    if store_location:
        res = res_torch_tensor[0].detach().numpy()
        index = res_torch_tensor[1].detach().numpy()
        def dot(l):
            d_r = 1
            for v in l:
                d_r = d_r * v
            return d_r

        def index_update():
            in_spatial_shape = x_shape[2:]
            spatial_pad_shape = [s + p[0] + p[1] for s, p in zip(in_spatial_shape, spatial_pads)]
            index_list = []
            for s in range(spatial_rank):
                if s != spatial_rank - 1:
                    index_list.append(index // dot(spatial_pad_shape[0:s+1]) - spatial_pads[s][0])
                else:
                    index_list.append(np.mod(index, dot(spatial_pad_shape[0:s])) - spatial_pads[s][0])
            index_in_orig = np.ones_like(index)
            for s in range(spatial_rank):
                if s != spatial_rank - 1:
                    index_in_orig = index_in_orig + index_list[s] * in_spatial_shape[s-1]
                else:
                    index_in_orig = index_in_orig + index_list[s]
            index_in_orig = index_in_orig - 1
            return index_in_orig

        return [res, index_update()]
    else:
        return res_torch_tensor.detach().numpy()
