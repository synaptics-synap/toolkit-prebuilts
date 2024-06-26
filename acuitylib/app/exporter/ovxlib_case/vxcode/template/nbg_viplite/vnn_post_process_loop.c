/****************************************************************************
*   Generated by ACUITY #ACUITY_VERSION#
*
*   Neural Network appliction post-process source file
****************************************************************************/
/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vnn_global.h"
#include "vnn_post_process.h"

#define _BASETSD_H

/*-------------------------------------------
                  Functions
-------------------------------------------*/
static float fp16_to_fp32
    (
    vip_int16_t in
    )
{
    const _fp32_t magic = { (254 - 15) << 23 };
    const _fp32_t infnan = { (127 + 16) << 23 };
    _fp32_t o;
    // Non-sign bits
    o.u = ( in & 0x7fff ) << 13;
    o.f *= magic.f;
    if(o.f  >= infnan.f)
    {
        o.u |= 255 << 23;
    }
    //Sign bit
    o.u |= ( in & 0x8000 ) << 16;
    return o.f;
} /* fp16_to_fp32() */

static float bfp16_to_fp32
    (
    vip_int16_t in
    )
{
    vip_int32_t t1, t2, t3;
    float out;

    t1 = in & 0x00FF;                       // Mantissa
    t2 = in & 0xFF00;                       // Sign bit + Exponent
    t3 = in & 0x7F00;                       // Exponent

    t1 <<= 16;
    t2 <<= 16;                              // Shift (sign + Exponent) bit into position
    t1 |= t2;                               // Re-insert (sign + Exponent) bit

    *((vip_uint32_t*)&out) = t1;

    return t3 == 0 ? 0 : out;
} /* bfp16_to_fp32() */

float vsi_nn_Fp16ToFp32
    (
    vip_int16_t in
    )
{
    return fp16_to_fp32(in);
} /* vsi_nn_Fp16ToFp32() */

float vsi_nn_BFp16ToFp32
    (
    vip_int16_t in
    )
{
    return bfp16_to_fp32(in);
} /* vsi_nn_Fp16ToFp32() */

float vsi_nn_DataAsFloat32
    (
    vip_uint8_t    * data,
    const vip_enum type
    )
{
    float val;
    vip_uint32_t *p = (vip_uint32_t*)(&val);
    vip_int16_t fp16;

    *p = 0xFFFFFFFF;
    switch( type )
    {
    case VIP_BUFFER_FORMAT_INT8:
        val = (float)((vip_int8_t*)data)[0];
        break;
    case VIP_BUFFER_FORMAT_UINT8:
        val = (float)data[0];
        break;
    case VIP_BUFFER_FORMAT_INT16:
        val = (float)( (vip_int16_t *)data )[0];
        break;
    case VIP_BUFFER_FORMAT_UINT16:
        val = (float)( (vip_uint16_t *)data )[0];
        break;
    case VIP_BUFFER_FORMAT_FP16:
        fp16 = ( (vip_int16_t *)data )[0];
        val = vsi_nn_Fp16ToFp32( fp16 );
        break;
    case VIP_BUFFER_FORMAT_BFP16:
        fp16 = ( (vip_int16_t *)data )[0];
        val = vsi_nn_BFp16ToFp32( fp16 );
        break;
    case VIP_BUFFER_FORMAT_INT32:
        val = (float)( (vip_int32_t *)data )[0];
        break;
    case VIP_BUFFER_FORMAT_UINT32:
        val = (float)( (vip_uint32_t *)data )[0];
        break;
    case VIP_BUFFER_FORMAT_FP32:
        val = ( (float *)data )[0];
        break;
    case VIP_BUFFER_FORMAT_INT64:
    case VIP_BUFFER_FORMAT_UINT64:
    case VIP_BUFFER_FORMAT_FP64:
    default:
        printf( "Unsupport type %d", type );
        break;
    }
    return val;
} /* vsi_nn_DataAsFloat32() */

vip_uint32_t shape_to_string(
    vip_uint32_t   *shape,
    vip_uint32_t   dim_num,
    char          *buf,
    vip_uint32_t   buf_sz,
    vip_bool_e     for_print)
{
#define _PRINT_FMT     (0)
#define _NOT_PRINT_FMT (1)
    vip_uint32_t s;
    vip_uint32_t count;
    const char * all_fmt[] = {" %d,", "%d_" };
    const char * fmt;
    if( NULL == shape || NULL == buf
        || dim_num == 0 || buf_sz == 0 )
    {
        return 0;
    }
    if( vip_false_e == for_print )
    {
        fmt = all_fmt[_NOT_PRINT_FMT];
    }
    else
    {
        fmt = all_fmt[_PRINT_FMT];
    }
    count = 0;
    for( s = 0; s < dim_num; s++ )
    {
        if( count >= buf_sz )
        {
            break;
        }
        count += snprintf( &buf[count], buf_sz - count,
            fmt, shape[s] );
    }
    buf[count - 1] = 0;
    return count;
}

unsigned int save_file(
    const char *name,
    void *data,
    unsigned int size)
{
    FILE *fp = fopen(name, "wb+");
    unsigned int saved = 0;

    if (fp != NULL) {
        saved = fwrite(data, size, 1, fp);
        fclose(fp);
    } else {
        printf("Saving file %s failed.\n", name);
    }

    return saved;
}

void save_data_to_fp32(
    const char *name,
    void *data,
    unsigned int size,
    const vip_buffer_create_params_t *buf_param)
{
    FILE *fp = NULL;
    vip_uint32_t i = 0, stride = 0;
    vip_uint32_t element_size = 0;
    vip_uint8_t * data_ptr = (vip_uint8_t *)data;

    stride = type_get_bytes(buf_param->data_format);
    element_size = size / stride;
    fp = fopen(name, "w");
    if (!fp)
    {
        printf("Open file %s failed.", name);
    }
    for (i = 0; i < element_size; i++)
    {
        float val = vsi_nn_DataAsFloat32(&data_ptr[stride * i], buf_param->data_format);
        fprintf(fp, "%f\n", val);
    }
}

vip_status_e save_output_data(
    vip_network_items *network_items)
{
    vip_status_e status = VIP_SUCCESS;
    int i = 0;
#define _DUMP_FILE_LENGTH 1028
#define _DUMP_SHAPE_LENGTH 128
    char filename[_DUMP_FILE_LENGTH] = {0}, shape[_DUMP_SHAPE_LENGTH] = {0};
    int buff_size = 0;
    void *out_data = NULL;
    vip_buffer_create_params_t buf_param;

    for (i = 0; i < network_items->output_count; i++)
    {
        buff_size = vip_get_buffer_size(network_items->output_buffers[i]);
        if (buff_size <= 0)
        {
            status = VIP_ERROR_IO;
            return status;
        }
        memset(&buf_param, 0, sizeof(buf_param));
        status = vip_query_output(network_items->network, i,
            VIP_BUFFER_PROP_DATA_FORMAT, &buf_param.data_format);
        _CHECK_STATUS(status, final);
        status = vip_query_output(network_items->network, i,
            VIP_BUFFER_PROP_NUM_OF_DIMENSION, &buf_param.num_of_dims);
        _CHECK_STATUS(status, final);
        status = vip_query_output(network_items->network, i,
            VIP_BUFFER_PROP_SIZES_OF_DIMENSION, buf_param.sizes);
        _CHECK_STATUS(status, final);

        shape_to_string( buf_param.sizes, buf_param.num_of_dims,
            shape, _DUMP_SHAPE_LENGTH, vip_false_e );
        snprintf(filename, _DUMP_FILE_LENGTH, "output%u_%s.tensor", i, shape);

        out_data = vip_map_buffer(network_items->output_buffers[i]);
        /* TODO: use save_file(filename, out_data, buff_size) to save binary file. */
        save_data_to_fp32(filename, out_data, buff_size, &buf_param);
    }

final:
    return status;
}

vip_status_e destroy_network(
    vip_network_items *network_items)
{
    vip_status_e status = VIP_SUCCESS;
    int i = 0;

    status = vip_finish_network(network_items->network);
    _CHECK_STATUS(status, final);
    status = vip_destroy_network(network_items->network);
    _CHECK_STATUS(status, final);
    /* TODO: if conn_cnt + input_count not equal input_num, refine this */
    for (i = 0; i < (network_items->input_count + network_items->rnn_conn.conn_cnt); i++)
    {
        status = vip_destroy_buffer(network_items->input_buffers[i]);
        _CHECK_STATUS(status, final);
    }
    if (network_items->input_buffers) {
        free(network_items->input_buffers);
        network_items->input_buffers = VIP_NULL;
    }
    for (i = 0; i < network_items->output_count; i++)
    {
        status = vip_destroy_buffer(network_items->output_buffers[i]);
        _CHECK_STATUS(status, final);
    }
    if (network_items->output_buffers) {
        free(network_items->output_buffers);
        network_items->output_buffers = VIP_NULL;
    }

final:
    return status;
}

void destroy_network_items(
    vip_network_items *network_items)
{
    if (network_items->nbg_name) {
        free(network_items->nbg_name);
        network_items->nbg_name = VIP_NULL;
    }
    if (network_items->input_names) {
        free(network_items->input_names);
        network_items->input_names = VIP_NULL;
    }
    if (network_items->rnn_conn.connections) {
        free(network_items->rnn_conn.connections);
        network_items->rnn_conn.connections = VIP_NULL;
    }
    if (network_items) {
        free(network_items);
        network_items = VIP_NULL;
    }
}