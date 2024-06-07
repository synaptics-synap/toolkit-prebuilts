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

from acuitylib.converter.paragraph import param_map_ops, ddict, param_types_convert_func, \
                                          paragraph_serialize, match_type
import json
import sys
from acuitylib.acuitylog import AcuityLog as al
from acuitylib.acuitynet import AcuityNet
import copy
import numpy as np
import os
from acuitylib.converter.tensorflow.tensorflowloader import TF_Graph_Preprocess
from acuitylib.converter.tensorflow.tensorflowloader import Quantize_Util as quant_util
from acuitylib.converter.tensorflow.tf_gen_ruler_utils import SubMatchMap
from acuitylib.converter.tensorflow.tf_util import TFProto_Util
from acuitylib.converter.tensorflow.nx_for_acu import NX_ACU
from acuitylib.converter.ruler_tool import RulerDB, Ruler
from acuitylib.converter.tensor_util import TensorLabelOperator as tlo
from acuitylib.converter.tensor_util import OP_ALIAS as oa
from acuitylib.converter.tensor_model import TensorModel, IN, OUT
from acuitylib.layer.customlayer import CustomLayer
from acuitylib.core.dtype import DType
from acuitylib.xtf import xtf as tf
from collections import OrderedDict
import platform
import re
import math

from acuitylib.converter.pyfunc_builder import pyfunc_call


class Match_Map:
    def __init__(self, ruler, alias_node_map, acu_ly_alias_map):
        self.ruler = ruler
        self.alias_node_map = alias_node_map
        self.acu_ly_alias_map = acu_ly_alias_map


class convert_tf():
    # setup paragraph
    def __init__(self, tf_ruler_db_file, args):
        self.args = args
        self.nx_and_acu = NX_ACU('tensorflow')
        self.tp_u = TFProto_Util()
        self.ready_tensors = list()
        self.processed_tensor = list()
        self.input_is_scalar_list = list()
        # Global Attribute
        # dirname = os.path.dirname(args.tf_pb)
        inputs = args.inputs
        outputs = args.outputs

        input_size_list = list()
        if args.input_size_list is not None:
            input_size_list = args.input_size_list
            for size in input_size_list:
                if len(size) == 0:  # scalar
                    self.input_is_scalar_list.append(True)
                    size.append(1)
                else:
                    self.input_is_scalar_list.append(False)
            if len(input_size_list) != 0 and len(input_size_list) != len(inputs):
                al.e('length of input_size_list is not equal to inputs')
        else:
            args.input_size_list = [[None] for _ in range(len(inputs))]

        # Define global define
        self.net_inputs = inputs
        self.net_outputs = outputs
        self.net_input_size_list = list()  # the first dim of shape is zero, which is meaningful for acuitynet
        self.acu_layer_tbl = ddict()
        self.net_name = os.path.splitext(os.path.basename(args.tf_pb))[0]

        self.anet = AcuityNet(self.net_name, platform = 'tensorflow')
        # Store matched layers, nodes, and match pargraph_id for each match result.
        self.ruler_matched_tbl = list()
        with_batch_size_list = list()  # for tensorflow, the first dim of shape should not be zero
        for i in range(len(input_size_list)):
            with_batch_size_list.append(input_size_list[i])
            self.net_input_size_list.append(input_size_list[i])

        self.tfg_preprocess = TF_Graph_Preprocess(args.tf_pb, self.net_inputs, self.net_outputs,
                                                  with_batch_size_list,
                                                  args.predef_file, args.subgraphs)
        self.sub_match_map = SubMatchMap(self.tfg_preprocess, None, None, None)
        self.input_tensor_list = self.tfg_preprocess.input_tensors
        self.output_tensor_list = self.tfg_preprocess.output_tensors
        self.main_version = self.tfg_preprocess.graph.versions.producer
        self.minor_version = self.tfg_preprocess.graph.versions.min_consumer
        al.i('Current TF Model producer version {} min consumer version {} bad consumer version []'.
             format(self.main_version, self.minor_version, self.tfg_preprocess.graph.versions.bad_consumers))

        # Load convert_ruler database
        filter = None
        self.ruler_db = self.build_ruler_db(tf_ruler_db_file, self.main_version, self.minor_version, filter)

    '''
    define build-in functions
    '''
    def is_const(self, node_name):
        return self.tfg_preprocess.is_const_node(node_name)
    # check inputs have const
    def have_const_in_inputs(self, tensor):
        for in_tensor in self.tfg_preprocess.node(tensor).input:
            if self.is_const(in_tensor):
                return True
        return False

    def is_const_tensor(self, tensor_name):
        node, dir, port = tlo().tensor_label_split(tensor_name)
        if dir == 'in':
            return False
        assert(dir == 'out')
        return self.tfg_preprocess.is_const_node(node)


    # check input is const
    def input_port_is_const(self, tensor, pt):
        port = int(pt.split('in')[1])
        return self.is_const(tensor.input[port])
    # Check tensor shape
    def shape_pick(self, tensor_name):
        node_name, dir, port = tlo().tensor_label_split(tensor_name)
        if self.is_const(node_name):
            if self.tensor_to_numpy(tensor_name).shape == ():
                return [1]
            return self.tensor_to_numpy(tensor_name).shape
        return self.tfg_preprocess.query_tensors(tensor_name, ret='shape')[0]

    def tensor_is_scalar(self, tensor_ref):
        return self.shape_pick(tensor_ref) == ()

    def attr_pick(self, in_tsr, key, default=0):
        if isinstance(in_tsr, str):
            node = self.tensor_model.tensor_factory(in_tsr)
        else:
            node = in_tsr
        if key not in node.params:
            return default
        return node.params[key]

    def tf_type_enum_to_ac_type(self, tf_type):
        # map tf type enum to acuity type
        return DType.map_to_acuity_dtype(tf_type, 'tensorflow')

    def array_layout(self, array, layout):
        new_array = copy.copy(array)
        for id, map_id in enumerate(layout):
            new_array[id] = array[map_id]
        return new_array

    def tensor_to_numpy(self, tensor_name, trans=None, dtype=None):
        name, dir, port = tlo().tensor_label_split(tensor_name)
        assert(dir=='out' and port==0)
        np_data = self.tfg_preprocess.load_const_tensor(tensor_name)
        if trans != None:
            np_data = np.transpose(np_data, trans)
        if dtype != None:
            np_data = np_data.astype(dtype)
        #convert a none shape ndarry to a shape [1] ndarray
        #this is a workaround to let functions such as INTS, FLOATS in paragraph.py
        #could process array smoothly without meet issue like:
        #'iteration over a 0-d array'
        if np_data.shape is tuple():
            np_data.shape = (1)
        return np_data

    def squeeze_shapes(self, axes, shapes):
        ret = list()
        for i, axis in enumerate(axes):
            if axis < 0:
                axes[i] = len(shapes)+axis
        for id in range(len(shapes)):
            if len(axes) == 0:
                if shapes[id] != 1:
                    ret.append(shapes[id])
            else:
                if id not in axes:
                    ret.append(shapes[id])
        if ret[0] == 1 and len(axes) != 0:
            ret[0] = 0
        return ret

    def reshape_shape(self, tensor_name):
        np_data = self.tensor_to_numpy(tensor_name)
        reshape_size = np_data
        if reshape_size[0] == 1:
            if -1 not in reshape_size:
                reshape_size[0] = -1
            else:
                reshape_size[0] = 0
                al.w('Network may not support batch > 1 !')
        return reshape_size

    def split_slice(self, dim, in_shape, split_num):
        return [int(in_shape[dim[0]] / split_num) * i for i in range(1, split_num)]

    def splitv_slice(self, const , split_num):
        slices = []
        total = 0
        for i in range(split_num-1):
            total += const[i]
            slices.append(total)
        return slices

    def deconv_output_shape(self, output_shape):
        ret_output_shape = list()
        for item in output_shape:
            ret_output_shape.append(int(item))
        if ret_output_shape[0] == 1:
            ret_output_shape[0] = 0
        return ret_output_shape

    def gru_kernel(self, gates_name, candidate_name):
        # gates_kernel:     [hidden_size + input_size, hidden_size * 2]
        # candidate_kernel: [hidden_size + input_size, hidden_size]
        gates_kernel = self.tensor_to_numpy(gates_name)
        candi_kernel = self.tensor_to_numpy(candidate_name)
        gates_shape = gates_kernel.shape
        hidden_size = int(gates_shape[1] / 2)
        input_size = gates_shape[0] - hidden_size

        kr, kz = np.split(gates_kernel, indices_or_sections=2, axis=1)
        k_ir, k_rr = np.split(kr, indices_or_sections=[input_size], axis=0)
        k_iz, k_rz = np.split(kz, indices_or_sections=[input_size], axis=0)
        k_ih, k_rh = np.split(candi_kernel, indices_or_sections=[input_size], axis=0)

        input_kernel = [k_iz, k_ir, k_ih]
        recurrent_kernel = [k_rz, k_rr, k_rh]
        return [input_kernel, recurrent_kernel]

    def gru_bias(self, gates_name, candidate_name):
        gates_bias = self.tensor_to_numpy(gates_name)
        candi_bias = self.tensor_to_numpy(candidate_name)
        hidden_size = candi_bias.shape[0]

        b_ir, b_iz = np.split(gates_bias, 2, axis=0)
        b_rr = b_rz = np.zeros([hidden_size], dtype=np.float32)
        b_ih = candi_bias
        b_rh = np.zeros([hidden_size], dtype=np.float32)
        input_bias = [b_iz, b_ir, b_ih]
        recurrent_bias = [b_rz, b_rr, b_rh]
        return [input_bias, recurrent_bias]

    '''
    build-in functions done
    '''
    def check_out_branch(self, ruler, nodes):
        src_out_tensors = ruler.src_out_tensor
        all_nodes = list()
        to_check_nodes = list()
        all_flows = list()

        for name in nodes.values():
            # refer: tensorflowload.TF_Graph_Preprocess.calc_2_const
            if self.tensor_model.node(name).op not in ['Const'] or\
                            name.endswith(TF_Graph_Preprocess.new_const_node_name_suffix):
                all_nodes.append(name)
                to_check_nodes.append(name)

        for out_tensor in src_out_tensors:
            out_node_alias = out_tensor.split(':')[0]
            out_node_name = nodes[out_node_alias]
            if out_node_name in to_check_nodes:
                to_check_nodes.remove(out_node_name)

        for node in to_check_nodes:
            flows = self.tensor_model.flow(node + ':out0', 'to')
            if flows is not None:
                all_flows.extend(flows)

        for flow in all_flows:
            dst_node = flow[1].split(':')[0]
            if dst_node not in all_nodes:
                return False

        return True

    def _tf_match_flow_process(self, cur_node_name, ruler):
        def real_name_to_alias(tensor):
            real_map = {value: key for key, value in alias_map.items()}
            name, dir, port = tlo().tensor_label_split(tensor)
            return tlo().tensor_label(real_map.get(name, '__UNKNOWN'), dir, port)
        tensor = dict()
        out_nodes = set()
        for out_tensor in ruler.tensor_mode.out_tensors:
            out_nodes.add(ruler.tensor_mode.product_by(out_tensor))
        for out_root_tensor, flow_in_scan_order in ruler.tensor_mode.flow_in_scan_order.items():
            # Step 1, pick a unique graph to compare
            sub_graph_items = set()
            op = ruler.tensor_mode.tensor_factory(ruler.tensor_mode.product_by(out_root_tensor)).op
            real_tensor_factory = self.tensor_model.tensor_factory(cur_node_name)
            if real_tensor_factory.op != op:
                continue
            # Step 2, try build graph
            root_node_name, orig_dir, orig_port = tlo().tensor_label_split(out_root_tensor)
            alias_map = dict()
            alias_map[root_node_name] = cur_node_name
            tensor_map = dict()
            root_tensor_label = tlo().tensor_label(cur_node_name, orig_dir, orig_port)
            tensor_map[out_root_tensor] = root_tensor_label

            real_tensor = self.tensor_model.tensor(root_tensor_label)
            if real_tensor == None:
                continue
            # sub_graph_items.add(real_tensor)
            sub_graph_items.add(real_tensor_factory)
            sub_graph_items.update(self.tensor_model.model.successors(real_tensor_factory))

            # out_nodes = set()
            # for out_tensor in ruler.src_out_tensor:
            #     node, dir, port = tlo().tensor_label_split(out_tensor)
            #     out_nodes.add(node)

            attach_dict = dict()
            match = True
            processed_flow = set()
            for flow in flow_in_scan_order:
                r_f_dst, r_f_dst_dir, r_f_dst_port = tlo().tensor_label_split(flow[IN])
                r_f_src, r_f_src_dir, r_f_src_port = tlo().tensor_label_split(flow[OUT])
                # Scan down
                if r_f_dst in alias_map:
                    real_dst_node_name = alias_map[r_f_dst]
                    real_dst_tensor_label = tlo().tensor_label(real_dst_node_name, r_f_dst_dir, r_f_dst_port)
                    if self.tensor_model.tensor(real_dst_tensor_label) == None:
                        match = False
                        break
                    real_src_tensor_label = self.tensor_model.flow_from(real_dst_tensor_label)
                    real_src_node_name = self.tensor_model.product_by(real_src_tensor_label)

                    tensor_map[flow[OUT]] = real_src_tensor_label
                    tensor_map[flow[IN]] = real_dst_tensor_label
                    alias_map[r_f_src] = real_src_node_name

                    sub_graph_items.add(self.tensor_model.tensor_factory(real_dst_node_name))
                    for t in self.tensor_model.consume_from(real_dst_node_name):
                        sub_graph_items.add(self.tensor_model.tensor(t))
                    if oa().alias_op(r_f_src) == oa().alias_op('I'):
                        if flow[OUT] not in attach_dict:
                            attach_dict[flow[OUT]] = set()
                        attach_dict[flow[OUT]].add(tensor_map[flow[OUT]])
                        sub_graph_items.add(self.tensor_model.tensor(real_src_tensor_label))
                        continue

                    if self.tensor_model.tensor_factory(real_src_node_name).op != \
                            ruler.tensor_mode.tensor_factory(r_f_src).op:
                        match = False
                        break
                    sub_graph_items.add(self.tensor_model.tensor_factory(real_src_node_name))
                    for t in self.tensor_model.product_to(real_src_node_name):
                        sub_graph_items.add(self.tensor_model.tensor(t))
                    if self.tensor_model.tensor_factory(cur_node_name).op in \
                            ['FakeQuantWithMinMaxVars', 'FakeQuantWithMinMaxVarsPerChannel']:
                        match = True
                # Scan up
                elif r_f_src in alias_map:
                    real_src_node_name = alias_map[r_f_src]
                    real_src_tensor_label = tlo().tensor_label(real_src_node_name, r_f_src_dir, r_f_src_port)
                    out_flows = self.tensor_model.flow(real_src_tensor_label, 'to')
                    if out_flows == None:
                        match = False
                        break
                    real_out_flow = list()
                    for out_f in out_flows:
                        alias_out_f = (real_name_to_alias(out_f[OUT]), real_name_to_alias(out_f[IN]))
                        if alias_out_f in processed_flow:
                            continue
                        else:
                            real_out_flow.append(out_f)

                    for out_f in real_out_flow:
                        real_dst_tensor_label = out_f[IN]
                        real_dst, real_dst_dir, real_dst_port = tlo().tensor_label_split(real_dst_tensor_label)
                        if real_dst in self.processed_nodes or self.tfg_preprocess.node(real_dst).op != \
                                oa().alias_op(r_f_dst) or real_dst_port != r_f_dst_port:
                            continue
                        real_dst_node_name = self.tensor_model.provide_to(real_dst_tensor_label)

                        tensor_map[flow[OUT]] = real_src_tensor_label
                        tensor_map[flow[IN]] = real_dst_tensor_label
                        alias_map[r_f_dst] = real_dst_node_name

                        if real_src_tensor_label == None:
                            match = False
                            break

                        sub_graph_items.add(self.tensor_model.tensor_factory(real_dst_node_name))
                        for t in self.tensor_model.consume_from(real_dst_node_name):
                            sub_graph_items.add(self.tensor_model.tensor(t))
                        if oa().alias_op(r_f_src) == oa().alias_op('I'):
                            if flow[OUT] not in attach_dict:
                                attach_dict[flow[OUT]] = set()
                            attach_dict[flow[OUT]].add(tensor_map[flow[OUT]])
                            sub_graph_items.add(self.tensor_model.tensor(real_src_tensor_label))
                            continue

                        if self.tensor_model.tensor_factory(real_src_node_name).op != \
                                ruler.tensor_mode.tensor_factory(r_f_src).op:
                            match = False
                            break
                        sub_graph_items.add(self.tensor_model.tensor_factory(real_src_node_name))
                        for t in self.tensor_model.product_to(real_src_node_name):
                            sub_graph_items.add(self.tensor_model.tensor(t))
                        if r_f_dst in out_nodes:
                            for tensor_name in self.tensor_model.product_to(real_dst_node_name):
                                tensor_obj = self.tensor_model.tensor(tensor_name)
                                tensor_map[tlo.tensor_label(r_f_dst, tensor_obj.dir, tensor_obj.port)] = tensor_name
                            sub_graph_items.update(
                                self.tensor_model.model.successors(
                                    self.tensor_model.tensor_factory(real_dst_node_name)
                                )
                            )

                        break
                else:
                    match = False
                    break
                processed_flow.add(flow)

            if match == True:
                for attach_alias, attach_set in attach_dict.items():
                    if len(attach_set) > 1:
                        continue
                ruler_base_graph = self.tensor_model.sub_model(sub_graph_items)
                if ruler.isomorphism(ruler_base_graph) == False:
                    continue
                else:
                    return True, alias_map, tensor_map
        return False, dict(), dict()

    def _tf_try_match_ruler(self, node_name):
        match = False
        cur_node_name = node_name
        ruler_id = 0
        next_flows = list()
        op_name = self.tensor_model.node(node_name).op
        for ruler_id, ruler in enumerate(self.ruler_db.ruler_list):
            match_id = ruler_id
            match, node, tensor = self._tf_match_flow_process(cur_node_name, ruler)
            real_node = copy.copy(node)
            if match == True:
                for flow in ruler.tensor_mode.in_flows:
                    src_node, src_dir, src_port = tlo().tensor_label_split(flow[OUT])
                    dst_node, dst_dir, dst_port = tlo().tensor_label_split(flow[IN])
                    next_flows.append(self.tensor_model.flow(tlo().tensor_label(node[dst_node],
                                                                                dst_dir, dst_port), 'from'))
                    if src_node in real_node:
                        real_node.pop(src_node)

            if match == True and ruler.pre_condition != None:
                if type(ruler.pre_condition) == list:
                    match = pyfunc_call(self, ruler.pre_condition, node, tensor)
                else:
                    match = eval(ruler.pre_condition)

            if match == True:
                matched_ruler = self.ruler_db.ruler_list[match_id]
                if len(matched_ruler.src_internal_flow) > 1:
                    match = self.check_out_branch(matched_ruler, real_node)

            if match == True:
                return True, self.ruler_db.ruler_list[match_id], real_node, tensor, next_flows

        # if not match the ruler, generate ruler
        if match == False:
            ruler = self.generate_ruler(self.tensor_model.tensor_factory(cur_node_name))
            if ruler is not None:
                al.i("Dynamically generate ruler '{}' for node '{}'".format(ruler.name, cur_node_name))
                match, node, tensor = self._tf_match_flow_process(cur_node_name, ruler)
                real_node = copy.copy(node)
                if match == True:
                    al.i("Match newly generated ruler '{}' successfully for node '{}'"
                         .format(ruler.name, cur_node_name))
                    for flow in ruler.tensor_mode.in_flows:
                        src_node, src_dir, src_port = tlo().tensor_label_split(flow[OUT])
                        dst_node, dst_dir, dst_port = tlo().tensor_label_split(flow[IN])
                        next_flows.append(self.tensor_model.flow(
                            tlo().tensor_label(node[dst_node], dst_dir, dst_port), 'from'))
                        if src_node in real_node:
                            real_node.pop(src_node)
                    return True, ruler, real_node, tensor, next_flows

        #If not match the ruler, convert it to custom layer.
        if match == False:
            ruler_id += 1
            node_name_scope = self.tp_u.query_name_scope(cur_node_name)
            self.tfg_preprocess.import_tf_ruler_module()
            #process while block
            if op_name == 'Exit' and node_name_scope in self.tfg_preprocess.while_block_dict:
                while_block_idx = 0
                for while_name_scope, value in self.tfg_preprocess.while_block_dict.items():
                    if while_name_scope == node_name_scope:
                        self.ruler_db.add_ruler(self.tfg_preprocess.tf_ruler_module.r_while_custom_template(
                            value, while_block_idx), 'tensorflow')
                    while_block_idx += 1
            #process normal custom layer
            else:
                node_input_cnt = len(self.tensor_model.node(node_name).input)
                node_output_cnt, out_port_list = self.tensor_model.node_out_port_count(node_name)
                self.ruler_db.add_ruler(self.tfg_preprocess.tf_ruler_module.r_custom_template(
                    op_name, node_input_cnt, node_output_cnt, out_port_list), 'tensorflow')

            #The last ruler in the ruler list must be the added ruler.
            ruler = self.ruler_db.ruler_list[ruler_id]
            match, node, tensor = self._tf_match_flow_process(cur_node_name, ruler)
            real_node = copy.copy(node)
            if match == True:
                for flow in ruler.tensor_mode.in_flows:
                    src_node, src_dir, src_port = tlo().tensor_label_split(flow[OUT])
                    dst_node, dst_dir, dst_port = tlo().tensor_label_split(flow[IN])
                    next_flows.append(self.tensor_model.flow(
                        tlo().tensor_label(node[dst_node], dst_dir, dst_port), 'from'))
                    if src_node in real_node:
                        real_node.pop(src_node)

            if match == True and ruler.pre_condition != None:
                if type(ruler.pre_condition) == list:
                    match = pyfunc_call(self, ruler.pre_condition, node, tensor)
                else:
                    match = eval(ruler.pre_condition)

            if match == True:
                al.w('Convert {} {} to {} customlayer'.format(cur_node_name, op_name, ruler.name))
                return True, ruler, real_node, tensor, next_flows
        return False, None, None, None, None

    def _tf_parse_param(self, process, node, tensor):
        # type = process[0]
        exec_str = process[1:]
        return param_types_convert_func[process[0]](self._tf_parase_execute(exec_str, node, tensor))

    def _tf_parase_execute(self, process, node, tensor):
        if process[0] == param_map_ops.value:
            return process[1]
        elif process[0] == param_map_ops.code:
            # print(process[1])
            return eval(process[1])
        elif process[0] == param_map_ops.pyfunc:
            return pyfunc_call(self, process[1], node, tensor)
        else:
            al.e('Only support {} and {} in define convert db'.format(param_map_ops.value, param_map_ops.code))

    def _tf_acu_blob_assign(self, aculayer, ruler, alayer_alias, node, tensor):
        if ruler.blob_map != None:
            for coef, process in ruler.blob_map.get(alayer_alias, dict()).items():
                aculayer.put_data_with_key(coef, self._tf_parase_execute(process, node, tensor))
                # Process Quantize Coef
                if process[0] == param_map_ops.code:
                    for match_str in re.findall(r'tensor[[].*?[]]', process[1]):
                        match_str = match_str.replace('tensor', '')
                        to_process_tensor = match_str.strip('[]\'\"')
                        for flow in ruler.tensor_mode.flow(to_process_tensor, 'to'):
                            dst_node = node[flow[1].split(":")[0]]
                            dst_tensor_factory = self.tensor_model.tensor_factory(dst_node)
                            if dst_tensor_factory.op in \
                                    ['FakeQuantWithMinMaxVars', 'FakeQuantWithMinMaxArgs',
                                     'FakeQuantWithMinMaxVarsPerChannel']:
                                num_bits = dst_tensor_factory.params['num_bits']
                                narrow_range = dst_tensor_factory.params['narrow_range']
                                quant_util().quantize_coef_tensor_set(
                                    self.tfg_preprocess,
                                    aculayer,
                                    coef,
                                    dst_node,
                                    num_bits,
                                    narrow_range
                                )
                                break
                if process[0] == param_map_ops.value and len(process) > 2:
                    quant_node = process[-1]
                    quant_tensor_factory = self.tensor_model.tensor_factory(quant_node)
                    if quant_tensor_factory.op in ['FakeQuantWithMinMaxVars', 'FakeQuantWithMinMaxArgs',
                                                   'FakeQuantWithMinMaxVarsPerChannel']:
                        num_bits = quant_tensor_factory.params['num_bits']
                        narrow_range = quant_tensor_factory.params['narrow_range']
                        quant_util().quantize_coef_tensor_set(
                            self.tfg_preprocess,
                            aculayer,
                            coef,
                            quant_node,
                            num_bits,
                            narrow_range
                        )


    def _tf_acu_param_assign(self, alayer, ruler, alayer_alias, node, tensor):
        if ruler.param_map != None:
            params = dict()
            for parm, process in ruler.param_map.get(alayer_alias, dict()).items():
                params[parm] = self._tf_parse_param(process, node, tensor)
            alayer.set_params(params)
        self._tf_acu_blob_assign(alayer, ruler, alayer_alias, node, tensor)

    def _tf_alias_to_op(self, alias):
        if alias.rfind('_') == -1:
            return alias
        if alias.split('_')[-1].isdigit() == False:
            return alias
        return '_'.join(alias.split('_')[0:-1])

    def _tf_build_acu_layer(self, ruler, nodes_map, tensor_map):
        # build alias name and node map
        node = nodes_map
        tensor = tensor_map
        ml_ops_alias_map = dict()
        ml_node_name_list = list()
        for alias, node_name in node.items():
            ml_ops_alias_map[alias] = node_name
            ml_node_name_list.append(node_name)

        acu_layer_alias_map = dict()
        for acu_layer_alias in ruler.acu_lys_alias:
            op = self._tf_alias_to_op(acu_layer_alias)
            name = ml_ops_alias_map[ruler.tensor_mode.product_by(ruler.src_out_tensor[0])]
            if len(ruler.acu_lys_alias) > 1 and acu_layer_alias != ruler.acu_lys_alias[0]:
                name = acu_layer_alias
            alayer = self.anet.new_layer(op, ret='layer', name=name)
            acu_layer_alias_map[acu_layer_alias] = alayer
            self._tf_acu_param_assign(alayer, ruler, acu_layer_alias, node, tensor)

            # Process Quantize Output
            if len(ruler.src_out_tensor) == 1 and oa().alias_op(ruler.tensor_mode.product_by(
                    ruler.src_out_tensor[0])) in ['FakeQuantWithMinMaxVars', 'FakeQuantWithMinMaxArgs',
                                                  'FakeQuantWithMinMaxVarsPerChannel']:
                if len(ruler.acu_lys_alias) != 1:
                    al.e('Donot merge quant ruler with several acuity layers.'
                         ' Please refine the ruler, split {} to different rulers.'.format(ruler.acu_lys_alias))
                #dst_node = node[flow[1].split(":")[0]]
                #dst_tensor_factory = self.tensor_model.tensor_factory(dst_node)
                # = dst_tensor_factory.params.num_bits
                fake_quant_node = node[ruler.src_out_tensor[0].strip(':out0')]
                fake_quant_tensor_factory = self.tensor_model.tensor_factory(fake_quant_node)
                num_bits = fake_quant_tensor_factory.params['num_bits']
                narrow_range = fake_quant_tensor_factory.params['narrow_range']
                if narrow_range:
                    al.e('currently not support quant tensorflow with narrow_range')
                qtype = 'u' + str(num_bits)
                quant_util().quantize_output_tensor_set(self.tfg_preprocess, alayer,
                                                        node[ruler.tensor_mode.product_by(ruler.src_out_tensor[0])],
                                                        qtype)

        # Build internal connect
        if (len(ruler.acu_lys_alias) > 1):
            for flow in ruler.acu_inter_flow:
                src_node, src_dir, src_port = tlo().tensor_label_split(flow[OUT])
                dst_node, dst_dir, dst_port = tlo().tensor_label_split(flow[IN])
                al.d('connect {}'.format(flow))
                self.anet.connect(acu_layer_alias_map[src_node], src_port, acu_layer_alias_map[dst_node], dst_port)

        return ml_ops_alias_map, acu_layer_alias_map, ml_node_name_list

    def build_ruler_db(self, db_file, main_version, minor_version, filters):
        ruler_db = RulerDB()
        ruler_db.setup_db(db_file, platform='tensorflow')
        return ruler_db

    def pre_process(self):
        self.is_quantize_model = False
        for node in self.tfg_preprocess.graph.node:
            if node.op in ['FakeQuantWithMinMaxVars', 'FakeQuantWithMinMaxArgs',
                           'FakeQuantWithMinMaxVarsPerChannel']:
                self.is_quantize_model = True
                break

        self.tfg_preprocess.pre_proces(ruler_db=self.ruler_db)
        self.tensor_model = self.tfg_preprocess.tensor_model
        self.sub_graph_id = 0

    def _tf_push_ready_node(self, next_flows):
        next_tensors = list()
        for flow in next_flows:
            if flow[OUT] not in next_flows:
                next_tensors.append(flow[OUT])
        self.ready_tensors.extend(next_tensors)
        ready_nodes = self.check_ready_factorys(next_tensors)
        for node in ready_nodes:
            self.ready_nodes.insert(0, node)

    def check_ready_factorys(self, marked_tensors):
        node_list = list()
        for marked_tensor in marked_tensors:
            node_name, dir, port, = tlo().tensor_label_split(marked_tensor)
            gen_tensors = self.tensor_model.product_to(node_name)
            ready = True
            for tensor in gen_tensors:
                if tensor not in self.ready_tensors:
                    ready = False
            if ready:
                if node_name not in node_list:
                    node_list.append(node_name)
        return list(reversed(node_list))

    def match_paragraph_and_param(self):
        for out_tensor in self.output_tensor_list:
            alayer = self.anet.new_layer('output', ret='layer', name='attach_' + out_tensor)
            self.acu_layer_tbl[out_tensor] = alayer
            al.i('build output layer {}'.format('attach_' + out_tensor))
            self.ready_tensors.append(out_tensor)

        self.ready_nodes = self.check_ready_factorys(self.output_tensor_list)

        self.processed_nodes = set()
        for indx, in_tensor in enumerate(copy.copy(self.input_tensor_list)):
            params = dict()
            node_name, dir, port = tlo().tensor_label_split(in_tensor)
            if dir == 'out':
                attach_node_name = '{}_{}_{}_placeholder'.format(node_name, dir, port)
                if self.tfg_preprocess.node(node_name) != None and \
                                self.tfg_preprocess.node(node_name).op == 'Placeholder':
                    attach_node_name = node_name
                real_in_tensor = tlo().tensor_label(attach_node_name, dir, port)
                self.processed_nodes.add(attach_node_name)
                alayer = self.anet.new_layer('input', ret='layer', name=node_name)
                alayer.params.is_scalar = self.input_is_scalar_list[indx]
                params['shape'] = copy.copy(self.net_input_size_list[indx])
                try:
                    node = self.tensor_model.tensorfactory_dict.get(node_name)
                    if node is None: # If we specify a custom input node, use attach_node_name
                        node = self.tensor_model.tensorfactory_dict.get(attach_node_name)
                    tf_type = node.nodeObj.attr.get('dtype').type
                except:
                    tf_type = tf.float32
                    al.w('Input node miss type, set a tf.float32 as a default type.')
                if not DType.is_backend_dtype_support(tf_type, 'tensorflow'):
                    al.w('Input type [{}] is not supported'.format(tf_type))
                else:
                    params['type'] = DType.map_to_acuity_dtype(tf_type, 'tensorflow')
                self.acu_layer_tbl[real_in_tensor] = alayer
                self.input_tensor_list[indx] = real_in_tensor
                alayer.set_params(params)
                al.i('build input layer {}'.format(real_in_tensor))
            else:
                src_node_name, src_dir, src_port =tlo().tensor_label_split(self.tensor_model.flow_from(in_tensor))
                self.processed_nodes.add(src_node_name)
                alayer = self.anet.new_layer('input', ret='layer', name='attach_' + in_tensor)
                params['shape'] = copy.copy(self.net_input_size_list[indx])
                try:
                    tf_type = self.tensor_model.tensorfactory_dict.get(node_name).nodeObj.attr.get('dtype').type
                except:
                    tf_type = None
                if tf_type is None or not DType.is_backend_dtype_support(tf_type, 'tensorflow'):
                    al.w('Input type [{}] is not supported'.format(tf_type))
                else:
                    params['type'] = DType.map_to_acuity_dtype(tf_type, 'tensorflow')
                self.acu_layer_tbl[in_tensor] = alayer
                alayer.set_params(params)
                al.i('build input layer {}'.format(in_tensor))

        current_node = self.ready_nodes.pop()
        # self.processed_nodes.add(self.input_tensor_list)

        while True:
            if current_node == None:
                break
            if current_node in self.processed_nodes:
                if len(self.ready_nodes) == 0:
                    current_node = None
                else:
                    current_node = self.ready_nodes.pop()
                continue

            al.d('Try match {} {}'.format(self.tensor_model.tensor_factory(current_node).op, current_node))
            matched, ruler, match_nodes_map, match_tensor_map, next_flow = self._tf_try_match_ruler(current_node)

            if matched:
                al.i('Match {} [{}] [{}] to [{}]'.format(
                    ruler.name,
                    [node_name for node_name in match_nodes_map.values()],
                    ruler.src_op_alias, ruler.acu_lys_alias))
                ml_op_alias_map, acu_layer_alias_map, ml_node_list = self._tf_build_acu_layer(ruler, match_nodes_map,
                                                                                              match_tensor_map)
                self.ruler_matched_tbl.append(
                    Match_Map(
                        ruler=ruler,
                        alias_node_map=ml_op_alias_map,
                        acu_ly_alias_map=acu_layer_alias_map
                    ))
                self.processed_nodes.update(match_nodes_map.values())
                for layer_alias, layer in acu_layer_alias_map.items():
                    if isinstance(layer, CustomLayer) and self.tfg_preprocess.node(current_node).op != 'Exit':
                        tensor_data_map = dict()
                        for tensor_alias, tensor in match_tensor_map.items():
                            if self.is_const_tensor(tensor):
                                tensor_data_map[tensor_alias] = self.tensor_to_numpy(tensor)
                        layer.load_params_from_tf(ruler, layer_alias, ml_op_alias_map, tensor_data_map)

            else:
                al.w('Not match node {} {} '.format(current_node, self.tfg_preprocess.node(
                    current_node).op))
                # TODO: add api for customers to add ruler
            self._tf_push_ready_node(next_flow)
            if len(self.ready_nodes) > 0:
                current_node = self.ready_nodes.pop()
            else:
                current_node = None

    def _tf_build_src_acu_map(self, src_acu_map, alias_value):
        if src_acu_map[0] == 'CODE' or src_acu_map[0] == 'VALUE':
            tensor = dict()
            for alias, op_label in alias_value.items():
                tensor[alias] = self.tfg_preprocess.node(op_label)
            value =  eval(src_acu_map[1])
            return value
        else:
            return src_acu_map

    def __query_src_acu_ly(self, achor_out_tensor, map_identify = 'in_map'):
        r"""Find the tensor match with achor_out_tensor in ruler_matched_tbl,
        and the matched tensor in acuity, return the acuity layer and the port.
        :param achor_out_tensor: the tensor to find in ruler_matched_tbl.
        :param map_identify: identify the achor_out_tensor is in_map or out_map,
            this param is special for the same tensor in pb will map to two or
            more (output)tensor of the same layer in acuity.(refine it)
        :return:
            0:the matched acuity layer
            1:the matched acuity layer's port which achor_out_tensor connected.
        """
        for src_match_map in self.ruler_matched_tbl:
            src_ruler = src_match_map.ruler
            for tensor in src_ruler.src_out_tensor:
                src_alias_node, src_alias_dir, src_alias_port = tlo().tensor_label_split(tensor)
                real_out_tensor = tlo().tensor_label(src_match_map.alias_node_map[src_alias_node], src_alias_dir,
                                                      src_alias_port)
                # src_match_output_tensors.append(real_out_tensors)
                if achor_out_tensor == real_out_tensor:
                    num_outmap_matched = 0
                    out_map_lists = list()
                    for out_map in src_ruler.src_acu_output_map:
                        if out_map[0] == tensor:
                            num_outmap_matched += 1
                            out_map_lists.append(out_map)
                    if num_outmap_matched < 1:
                        al.e('Not matched {} in ruler:{}'.format(tensor, src_ruler.name))
                    elif num_outmap_matched == 1:
                        out_map_list = out_map_lists[0]
                        acu_out_ly, acu_out_dir, acu_ly_outport = tlo().tensor_label_split(out_map_list[1])
                        return src_match_map.acu_ly_alias_map[acu_out_ly], acu_ly_outport
                    elif num_outmap_matched == 2:
                        if map_identify == 'in_map':
                            out_map_list = out_map_lists[0]
                            acu_out_ly, acu_out_dir, acu_ly_outport = tlo().tensor_label_split(
                                out_map_list[1])
                            return src_match_map.acu_ly_alias_map[acu_out_ly], acu_ly_outport
                        elif map_identify == 'out_map':
                            out_map_list = out_map_lists[1]
                            acu_out_ly, acu_out_dir, acu_ly_outport = tlo().tensor_label_split(
                                out_map_list[1])
                            return src_match_map.acu_ly_alias_map[acu_out_ly], acu_ly_outport
                        else:
                            al.e('The parameter:{} is wrong in {}'.format(
                                map_identify, sys._getframe().f_code.co_name))
                    else:
                        out_map_list = out_map_lists[0]
                        al.w('The {} may not matched with {}'.format(tensor, out_map_list[1]))
                        acu_out_ly, acu_out_dir, acu_ly_outport = tlo().tensor_label_split(out_map_list[1])
                        return src_match_map.acu_ly_alias_map[acu_out_ly], acu_ly_outport

                    break
        al.w('Not get any connection of {}'.format(achor_out_tensor))
        return None, None

    def graph_connection_build(self):
        for match_map in self.ruler_matched_tbl:
            ruler = match_map.ruler
            for in_map in ruler.src_acu_input_map:
                anchor_tensor = in_map[0]
                anchors = [anchor[1] for anchor in ruler.src_in_anchor if anchor[0] == anchor_tensor]
                acu_dst_layer_alias, acu_dir, acu_dst_port = tlo().tensor_label_split(in_map[1])
                acu_dst_ly = match_map.acu_ly_alias_map[acu_dst_layer_alias]
                alias_node, alias_dir, alias_port = tlo().tensor_label_split(anchors[0])
                attach_tensor = tlo().tensor_label(match_map.alias_node_map[alias_node], alias_dir, alias_port)
                src_tensor = self.tensor_model.flow_from(attach_tensor)

                if attach_tensor in self.input_tensor_list:
                    acu_src_ly, acu_src_port = self.acu_layer_tbl[attach_tensor], 0
                elif src_tensor in self.input_tensor_list:
                    acu_src_ly, acu_src_port = self.acu_layer_tbl[src_tensor], 0
                else:
                    acu_src_ly, acu_src_port = self.__query_src_acu_ly(src_tensor, 'in_map')

                al.d('connect {} {}  ~ {} {}'.format(acu_src_ly.lid, acu_src_port, acu_dst_ly.lid, acu_dst_port))

                self.anet.connect(acu_src_ly,
                                  acu_src_port,
                                  acu_dst_ly,
                                  acu_dst_port )

        # Connect Output layer
        for out_tensor in self.output_tensor_list:
            acu_output_layer = self.acu_layer_tbl[out_tensor]
            acu_src_ly, acu_src_port = self.__query_src_acu_ly(out_tensor, 'out_map')

            al.d('connect {} {}  ~ {} {}'.format(acu_src_ly.lid, acu_src_port, acu_output_layer.lid,
                                                 0))
            self.anet.connect(acu_src_ly,
                              acu_src_port,
                              acu_output_layer,
                              0)

    def generate_ruler(self, current_node):
        quantable_single_ops = self.sub_match_map.single_ops_dict.keys()
        if current_node.op in ['BiasAdd', 'Add', 'Conv2D', 'FakeQuantWithMinMaxVars', 'DepthwiseConv2dNative',
                               'Conv2DBackpropInput', 'BatchToSpaceND', 'Squeeze', 'MatMul',
                               'FakeQuantWithMinMaxVarsPerChannel']:
            conv_ops_list = list()
            conv_node_list = list()
            conv1d_ops_list = list()
            conv1d_node_list = list()
            matmul_ops_list = list()
            matmul_node_list = list()

            flags_conv = {'quant_w': False, 'quant_o': False, 'quant_b': False}
            flags_conv1d = {'quant_w': False, 'quant_o': False, 'quant_b': False}
            flags_matmul = {'quant_w': False, 'quant_o': False, 'quant_b': False}
            conv_ruler = None
            conv1d_ruler = None
            if current_node.op in ['Add', 'AddN', 'BiasAdd']:
                const_count = 0
                for input in self.tfg_preprocess.get_node_input_node(current_node.name):
                    if input.op in ['Const']:
                        const_count += 1
                if const_count != 0:
                    self.tfg_preprocess.find_conv_ops_list(current_node, conv_ops_list, conv_node_list, flags_conv)
                    self.tfg_preprocess.find_conv1d_ops_list(current_node, conv1d_ops_list,
                                                             conv1d_node_list, flags_conv1d)
                    self.tfg_preprocess.find_matmul_ops_list(current_node, matmul_ops_list,
                                                             matmul_node_list, flags_matmul)
            else:
                self.tfg_preprocess.find_conv_ops_list(current_node, conv_ops_list, conv_node_list, flags_conv)
                self.tfg_preprocess.find_conv1d_ops_list(current_node, conv1d_ops_list, conv1d_node_list, flags_conv1d)
                self.tfg_preprocess.find_matmul_ops_list(current_node, matmul_ops_list, matmul_node_list, flags_matmul)
            if 'Conv2D' in conv_ops_list or 'DepthwiseConv2dNative' in conv_ops_list \
                    or 'Conv2DBackpropInput' in conv_ops_list and 'Squeeze' not in conv_ops_list:
                conv_ruler = self.generate_conv_ruler(conv_node_list, conv_ops_list, flags_conv)
            if conv_ruler is not None:
                return conv_ruler

            if set(['Conv2D', 'Squeeze', 'ExpandDims']) <= set(conv1d_ops_list):
                conv1d_ruler = self.generate_conv_ruler(conv1d_node_list, conv1d_ops_list, flags_conv1d)
            if conv1d_ruler is not None:
                return conv1d_ruler
            if 'MatMul' in matmul_ops_list:
                matmul_ruler = self.generate_matmul_ruler(matmul_node_list, matmul_ops_list, flags_matmul)
                return matmul_ruler
        # TODO: multi ops ruler generate
        # if current_node.op in quantable_single_ops:
        #     multi_ops_ruler = self.generate_multi_ops_ruler(current_node)
        #     if multi_ops_ruler is not None:
        #         self.ruler_db.ruler_list.insert(0, multi_ops_ruler)
        #         return True

        # generate single op ruler (with FakeqQuant in output or const input or not)
        if current_node.op in ['FakeQuantWithMinMaxVars', 'FakeQuantWithMinMaxVarsPerChannel']:
            ruler_dict = dict()
            input_node_list = [current_node]
            output_node_list = [current_node]
            flags = dict()
            const_count = 0
            has_ph = False
            node_list = [current_node]
            for input in self.tfg_preprocess.get_node_input_node(current_node.name):
                if input.op in ['Const']:
                    const_count += 1
                if input.op in ['Placeholder']:
                    has_ph = True
            if const_count == 3 or has_ph:
                self.ruler_generate(input_node_list, output_node_list, node_list, 'quantize')
                self.sub_match_map.ruler.name = 'quantize'
                self.sub_match_map.single_op_param_map('FakeQuantWithMinMaxVars', flags)
                return self.sub_match_map.ruler

        if current_node.op in ['FakeQuantWithMinMaxVars',
                               'FakeQuantWithMinMaxVarsPerChannel'] or current_node.op in quantable_single_ops:
            node_list = list()
            flags = dict()
            self.tfg_preprocess.find_single_ops_list(current_node, node_list, quantable_single_ops, flags)
            single_ruler = self.generate_single_op_ruler(node_list, flags)
            #self.ruler_db.ruler_list.append(single_ruler)
            return single_ruler

        return None

    def generate_conv_ruler(self, node_list, op_list, flags):
        flags['dilated_conv'] = False
        flags['dilated_dwconv'] = False
        flags['dilated_conv1d'] = False
        acu_lys_alias = 'convolution'
        for node in node_list:
            if node.op == 'Conv2D':
                conv_node = node
                const_count = 0
                for in_node in self.tfg_preprocess.get_node_input_node(node.name):
                    if in_node.op in ['Const']:
                        const_count += 1
                    if in_node.op in ['FakeQuantWithMinMaxVars', 'FakeQuantWithMinMaxVarsPerChannel']:
                        q_const = 0
                        quant_node = in_node
                        for q_in in self.tfg_preprocess.get_node_input_node(quant_node.name):
                            if q_in.op in ['Const']:
                                q_const += 1
                        if q_const == 3:
                            const_count += 1
                if const_count == 0:
                    acu_lys_alias = 'conv2d_op'

            elif node.op in ['DepthwiseConv2dNative', 'Conv2DBackpropInput']:
                conv_node = node
            elif node.op in ['Pad']:
                pad_node = node

        if set(['Squeeze', 'ExpandDims', 'Conv2D']) <= set(op_list):
            acu_lys_alias = 'conv1d'

        if 'Conv2DBackpropInput' in op_list:
            acu_lys_alias = 'deconvolution'

        if set(['BatchToSpaceND', 'SpaceToBatchND', 'Conv2D']) <= set(op_list):
            if acu_lys_alias in ['conv1d']:
                flags['dilated_conv1d'] = True
            else:
                flags['dilated_conv'] = True

        if set(['BatchToSpaceND', 'SpaceToBatchND', 'DepthwiseConv2dNative']) <= set(op_list):
            flags['dilated_dwconv'] = True

        if acu_lys_alias == 'convolution':
            if 'Squeeze' in op_list or 'ExpandDims' in op_list:
                return None

        if not flags['dilated_conv'] and not flags['dilated_dwconv'] \
                and not flags['dilated_conv1d']:
            if 'BatchToSpaceND' in op_list or 'SpaceToBatchND' in op_list:
                return None

        input_node_list = node_list
        output_node_list = [node_list[0]]
        self.ruler_generate(input_node_list, output_node_list, node_list, acu_lys_alias)
        name_list = self.sub_match_map.generate_conv_ruler_name(acu_lys_alias, op_list, flags)
        ruler_name = '_'.join(name_list)
        self.sub_match_map.ruler.name = ruler_name
        self.sub_match_map.generate_conv_map(op_list, node_list, flags)
        return self.sub_match_map.ruler

    def generate_matmul_ruler(self, node_list, op_list, flags):
        acu_lys_alias = 'fullconnect'
        for node in node_list:
            if node.op == 'MatMul':
                const_count = 0
                for in_node in self.tfg_preprocess.get_node_input_node(node.name):
                    if in_node.op in ['Const']:
                        const_count += 1
                    if in_node.op in ['FakeQuantWithMinMaxVars', 'FakeQuantWithMinMaxVarsPerChannel']:
                        q_const = 0
                        quant_node = in_node
                        for q_in in self.tfg_preprocess.get_node_input_node(quant_node.name):
                            if q_in.op in ['Const']:
                                q_const += 1
                        if q_const == 3:
                            const_count += 1
                if const_count == 0:
                    acu_lys_alias = 'fullconnect_op'

        input_node_list = node_list
        output_node_list = [node_list[0]]
        self.ruler_generate(input_node_list, output_node_list, node_list, acu_lys_alias)
        name_list = self.sub_match_map.generate_matmul_ruler_name(flags)
        ruler_name = '_'.join(name_list)
        self.sub_match_map.ruler.name = ruler_name
        self.sub_match_map.generate_matmul_map(op_list, node_list, flags)
        return self.sub_match_map.ruler

    def generate_single_op_ruler(self, node_list, flags):
        name_list = []
        acu_lys_alias = 'noop'
        op_node = None

        fakequant_node = None
        # create a dict which contains single op acu_lys_alias and ruler name
        for node in node_list:
            if node.op not in ['FakeQuantWithMinMaxVars', 'Const', 'FakeQuantWithMinMaxVarsPerChannel']:
                name_list.append(self.sub_match_map.single_ops_dict[node.op]['ruler_name'])
                acu_lys_alias = self.sub_match_map.single_ops_dict[node.op]['acu_lys_alias']
                op_node = node
            if node.op in ['FakeQuantWithMinMaxVars', 'FakeQuantWithMinMaxVarsPerChannel']:
                fakequant_node = node
        for key in flags.keys():
            name_list.append(key)
        ruler_name = '_'.join(name_list)

        if len(node_list) == 1:
            if node_list[0].op in ['FakeQuantWithMinMaxVars', 'FakeQuantWithMinMaxVarsPerChannel']:
                op_node = node_list[0]
                acu_lys_alias = 'quantize'
            input_node_list = [node_list[0]]
            output_node_list = [node_list[0]]
        else:
            if op_node.op in self.sub_match_map.multi_input_ops:
                input_node_list = [op_node]
            else:
                input_node_list = node_list
                if fakequant_node is not None:
                    input_node_list.remove(fakequant_node)

            if fakequant_node is None:
                output_node_list = [op_node]
            else:
                output_node_list = [fakequant_node]

        self.ruler_generate(input_node_list, output_node_list, node_list, acu_lys_alias, True)
        self.sub_match_map.ruler.name = ruler_name
        if self.sub_match_map.single_ops_dict[op_node.op]['map_func'] is not None:
            self.sub_match_map.single_op_param_map(op_node.op, flags)
        return self.sub_match_map.ruler

    def generate_multi_ops_ruler(self, current_node):
        node_list, rulertuple = self.sub_match_map.match_multi_ops(current_node)
        if rulertuple is None:
            return None
        input_node_list = self.tfg_preprocess.find_input_node_in_subgraph(node_list)
        output_node_list = [current_node]
        self.ruler_generate(input_node_list, output_node_list, node_list, rulertuple.acu_lys_alias)
        self.sub_match_map.ruler.name = rulertuple.ruler_name
        self.sub_match_map.multi_ops_param_map()
        return self.sub_match_map.ruler

    def ruler_generate(self, input_node_list, output_node_list, node_list, acu_lys_alias, single_op_ruler=False):
        nx_and_acu = NX_ACU('tensorflow')
        ruler_dict = dict()
        ruler_dict['ruler_name'] = 'name'

        # Find disconnect output ports
        output_tensor_list = list()
        for node in output_node_list:
            num = None
            if node.op in self.sub_match_map.multi_output_ops.keys():
                if 'num_split' in node.params.keys():
                    num = node.params['num_split']
                elif 'num' in node.params.keys():
                    num = node.params['num']
            if num is not None:
                for i in range(num):
                    output_tensor_list.append(node.name + ':out' + str(i))
            else:
                output_tensor_list.append(node.name + ':out0')

        # Find disconnected input ports
        # For common ops
        discnt_port = dict()
        input_tensor_list = list()
        discnt_node = set()
        for node in input_node_list:
            port = []
            input_tensors = self.tensor_model.consume_from(node.name)
            if node.op in ['FakeQuantWithMinMaxVars', 'FakeQuantWithMinMaxVarsPerChannel'] and len(node_list) == 1:
                port.append(0)
                src_node = self.tfg_preprocess.query_node(node.nodeObj.input[0])
                discnt_node.add(src_node.name)
            else:
                for dst_tensor in input_tensors:
                    src_tensor = self.tensor_model.flow_from(dst_tensor)
                    src_node = self.tfg_preprocess.query_node(self.tensor_model.product_by(src_tensor))
                    if src_node in node_list:
                        continue
                    if src_node.op not in ['Const'] or \
                            src_node.op in ['Const'] and single_op_ruler:
                        port.append(tlo().tensor_label_split(dst_tensor)[2])
                        if src_node.name not in discnt_node:
                            discnt_node.add(src_node.name)
                    elif node.op in self.sub_match_map.multi_input_ops and len(node_list) == 1:
                        port.append(tlo().tensor_label_split(dst_tensor)[2])
                        if src_node.name not in discnt_node:
                            discnt_node.add(src_node.name)

            if len(port) > 0:
                discnt_port[node.name] = port

        # For those ops who have more than one input
        if len(node_list) == 1 and node_list[0].op in self.sub_match_map.multi_input_ops:
            discnt_port = {}
            discnt_node = list()
            node = node_list[0]
            port = []
            for i, input in enumerate(self.tfg_preprocess.get_node_input_node(node.name)):
                port.append(i)
                discnt_node.append(self.tfg_preprocess.query_node(input.name).name)
            discnt_port[node.name] = port

        # For those ops who take each const as different inputs
        if len(node_list) == 1 and node_list[0].op in self.sub_match_map.special_input_ops:
            discnt_port = {}
            discnt_node = list()
            node = node_list[0]
            port = []
            node_input = self.tfg_preprocess.get_node_input_node(node.name)
            for i, input in enumerate(node_input):
                port.append(i)
                discnt_node.append(input.name)
            if node_input[-1].op in ['Const']:
                discnt_node.pop(-1)
                port.pop(-1)
            discnt_node = set(discnt_node)
            discnt_port[node.name] = port

        # Cut subgraph
        for name, ports in discnt_port.items():
            for port in ports:
                input_tensor_list.append(name+':in'+str(port))

        subgraph = self.tfg_preprocess.cut_subgraph_def(input_tensor_list, output_tensor_list)

        # Add placeholder in subgraph
        placeholder = {}
        for i, name in enumerate(discnt_node):
            new_node = self.tfg_preprocess.tp_util.create_node('Placeholder', 'input_' + str(i), [])
            placeholder[name] = new_node.name
            subgraph.node.extend([new_node])

        for input_node, ports in discnt_port.items():
            in_node = self.tfg_preprocess.query_node(input_node)
            for node in self.tfg_preprocess.graph.node:
                if node.name == input_node:
                    for i, input in enumerate(self.tfg_preprocess.get_node_input_node(in_node.name)):
                        if input.name in discnt_node:
                            new = placeholder[input.name]
                        else:
                            continue
                        for n in subgraph.node:
                            if n.name == node.name:
                                n.input.insert(i, new)
                                break

        # generate ruler
        nodes_in_scan_order = \
            nx_and_acu.build_unique_graph(subgraph, input_tensor_list, output_tensor_list, use_alias=True)

        ruler_dict['src_ops_alias'] = nodes_in_scan_order
        ruler_dict['src_inter_flow'] = nx_and_acu.flows(nodes_in_scan_order, use_alias=True)
        ruler_dict['src_in_anchor'] = nx_and_acu.in_flows(use_alias=True)
        ruler_dict['src_out_tensor'] = nx_and_acu.out_tensors(use_alias=True)
        ruler_dict['acu_lys_alias'] = [ly for ly in acu_lys_alias] if isinstance(acu_lys_alias, list) \
            else [acu_lys_alias]
        ruler_dict['src_acu_in_tensor_map'] = self.tfg_preprocess.gen_input_map(ruler_dict['src_in_anchor'],
                                                            [acu_ly for acu_ly in acu_lys_alias]
                                                            if isinstance(acu_lys_alias, list) else [acu_lys_alias])
        ruler_dict['src_acu_out_tensor_map'] = self.tfg_preprocess.gen_output_map(ruler_dict['src_out_tensor'],
                                                              [acu_ly for acu_ly in acu_lys_alias]
                                                              if isinstance(acu_lys_alias, list) else [acu_lys_alias])
        ruler_dict['acu_inter_flow'] = self.tfg_preprocess.gen_internal_flow([acu_lys_alias])
        ruler_dict['param_map'] = {acu_ly: dict() for acu_ly in acu_lys_alias} if isinstance(acu_lys_alias, list) \
            else {acu_lys_alias: dict()}
        ruler_dict['blob_map'] = {acu_ly: dict() for acu_ly in acu_lys_alias} if isinstance(acu_lys_alias, list) \
            else {acu_lys_alias: dict()}
        ruler_dict['priority_tip'] = 0
        ruler_dict['pre_condition'] = None
        self.sub_match_map.ruler = Ruler(ruler_dict)
        self.sub_match_map.alias_map = nx_and_acu.get_alias_map()
        self.sub_match_map.name_map = nx_and_acu.get_name_map()

if __name__ == '__main__':
    print('TEST')
