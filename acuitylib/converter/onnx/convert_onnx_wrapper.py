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

#
# Synaptics wrapper class for OnnxRulerMatcher
#

class OnnxRulerMatcherWrapper:
    def  __init__(self, base_obj):
        self.base_obj = base_obj
        base_obj.wrapper = self

    def eval_ruler_pre_condition(self, pre_condition, node, tensor):
        return eval(pre_condition)

    def __getattr__(self, name):
        return getattr(self.base_obj, name)
