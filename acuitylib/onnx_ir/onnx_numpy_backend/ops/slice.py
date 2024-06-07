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

def Slice(inputs, outputs, attr=None, op_version=11):  # type: (np.ndarray, list, dict, int) -> np.ndarray
    in_count = len(inputs)
    data = inputs[0]
    if isinstance(data, np.ndarray) == False:
        data = np.array(data)
    if op_version < 10:
        axes = attr.get('axes', None)
        ends = attr['ends']
        starts = attr['starts']
        steps = None
    else:
        starts = inputs[1]
        ends = inputs[2]
        axes = list(inputs[3]) if in_count >= 4 else None
        steps = list(inputs[4]) if in_count >= 5 else None

    rank = data.ndim
    if axes == None:
        axes = [i for i in range(len(starts))]
    axes = [a if a >= 0 else rank + a for a in axes]
    slice_arg = list()
    for r in range(rank):
        if isinstance(axes, list) and r in axes:
            index = axes.index(r)
            if isinstance(steps, list):
                slice_arg.append(str(int(starts[index])) + ':' + str(int(ends[index])) + ':' + str(int(steps[index])))
            else:
                slice_arg.append(str(int(starts[index])) + ':' + str(int(ends[index])))
        elif isinstance(axes, list) and r not in axes:
            slice_arg.append(':')
        elif axes == None:
            slice_arg.append(str(int(starts[r])) + ':' + str(int(ends[r])))
    slice_arg_str = ','.join(slice_arg)
    res = eval('data[{}]'.format(slice_arg_str))
    return res
