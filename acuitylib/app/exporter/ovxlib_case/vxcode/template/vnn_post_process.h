/****************************************************************************
*   Generated by ACUITY #ACUITY_VERSION#
*   Match ovxlib #OVXLIB_VERSION#
*
*   Neural Network appliction post-process header file
****************************************************************************/
#ifndef _VNN_POST_PROCESS_H_
#define _VNN_POST_PROCESS_H_

vsi_status vnn_PostProcess#NETWORK_NAME#(vsi_nn_graph_t *graph);

const vsi_nn_postprocess_map_element_t * vnn_GetPostProcessMap();

uint32_t vnn_GetPostProcessMapCount();

#endif