/****************************************************************************
*   Generated by ACUITY #ACUITY_VERSION#
*   Match ovxlib #OVXLIB_VERSION#
*
*   Neural Network appliction network definition header file
****************************************************************************/

#ifndef _VNN_#NETWORK_NAME_UPPER#_H
#define _VNN_#NETWORK_NAME_UPPER#_H

#include "vsi_nn_pub.h"

#define VNN_APP_DEBUG (FALSE)
#define VNN_VERSION_MAJOR #OVXLIB_VERSION_MAJOR#
#define VNN_VERSION_MINOR #OVXLIB_VERSION_MINOR#
#define VNN_VERSION_PATCH #OVXLIB_VERSION_PATCH#
#define VNN_RUNTIME_VERSION \
    (VNN_VERSION_MAJOR * 10000 + VNN_VERSION_MINOR * 100 + VNN_VERSION_PATCH)

_version_assert(VNN_RUNTIME_VERSION <= VSI_NN_VERSION,
                CASE_VERSION_is_higher_than_OVXLIB_VERSION)

void vnn_Release#NETWORK_NAME#
    (
    vsi_nn_graph_t * graph,
    vsi_bool release_ctx
    );

vsi_nn_graph_t * vnn_Create#NETWORK_NAME#
    (
    const char * data_file_name,
    vsi_nn_context_t in_ctx,
    const vsi_nn_preprocess_map_element_t * pre_process_map,
    uint32_t pre_process_map_count,
    const vsi_nn_postprocess_map_element_t * post_process_map,
    uint32_t post_process_map_count
    );

#endif
