/****************************************************************************
*   Generated by ACUITY #ACUITY_VERSION#
*
*   Neural Network global header file
****************************************************************************/
#ifndef _VNN_GLOBAL_H_
#define _VNN_GLOBAL_H_

#include "vip_lite.h"

#ifdef _MSC_VER
#define snprintf _snprintf
#endif

#define _CHECK_PTR( ptr, lbl )      do {\
    if( NULL == ptr ) {\
        printf("Error: %s: %s at %d\n", __FILE__, __FUNCTION__, __LINE__);\
        goto lbl;\
    }\
} while(0)

#define _CHECK_STATUS( stat, lbl )  do {\
    if( VIP_SUCCESS != stat ) {\
        printf("Error: %s: %s at %d\n", __FILE__, __FUNCTION__, __LINE__);\
        goto lbl;\
    }\
} while(0)

typedef struct _vip_network_items {
    /* argv information. */
    char           *nbg_name;
    int             input_count;
    char          **input_names;
    int             output_count;
    char          **output_names;

    /* VIP lite buffer objects. */
    vip_network     network;
    vip_buffer     *input_buffers;
    vip_buffer     *output_buffers;
} vip_network_items;

#endif
