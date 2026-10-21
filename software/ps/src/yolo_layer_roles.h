/* ===========================================================================
 * yolo_layer_roles.h -- who executes which layer.
 *
 * This header is included by BOTH the HLS kernel and the PS application, on
 * purpose.  The split between PL and PS used to live in two places (a shape test
 * inside the kernel's dispatch and a hand-written boundary list in the app), and
 * the two had to be kept in step by hand.  Now there is one predicate.
 *
 * THE RULE
 *   * OP_ATTENTION                    -> PL  (YOLO26's C2PSA was replaced by a
 *                                             CBAM block -- pooling, convolution
 *                                             and sigmoid, all of which the
 *                                             engine expresses.  Implemented in
 *                                             yolo_cbam_op.h, called from the
 *                                             dispatcher.)
 *   * everything else                 -> PL
 *
 * As of the grouped-conv rework NO layer in any table matches a PS rule, so the
 * whole network runs in ONE PL call.  That is the entire point: the two
 * attentions and the six head depthwise convs used to be PS-side and forced the
 * schedule to be cut into 9 range calls with 8 PS operators, which is where the
 * runtime went.
 *
 * ---------------------------------------------------------------------------
 * GROUPED / DEPTHWISE CONVOLUTION IS NOW A PL OPERATOR
 * ---------------------------------------------------------------------------
 * The grouped conv used to be the second PS rule:
 *
 *     if (L->op == OP_CONV && L->ic == 1 && L->oc > 1 && L->nin == 1) return 1;
 *
 * and it is gone.  It was only ever there because the engine's layer model was
 * "every output channel consumes all `ic` input channels" -- a depthwise conv
 * (groups == oc, one input channel per output channel) violates it -- and the
 * one attempt to fix that with a DEDICATED datapath cost ~12,773 LUT and failed
 * place-and-route (108% of the xc7z020, WNS -0.473 ns).
 *
 * It does not need a dedicated datapath.  See `op_conv_i8` in
 * yolo_hls_i8_26.cpp: the grouped case reuses the identical MAC array, row
 * cache, weight tile and requant chain, and differs only in three things --
 *
 *   * the serial input-channel loop runs once per LANE (lane t consumes input
 *     channel ocb+t, its group's only input channel) instead of once per layer,
 *   * the input channel index therefore equals the output channel index, and
 *     the per-channel zero point is read with the lane's own channel index,
 *   * the weight index collapses to `ky*k + kx` (the layer's `ic` is already 1,
 *     so `ic*k2` in the weight stride is already the per-output-channel count).
 *
 * `yolo_layer_is_grouped_conv()` below is that predicate, and it is the SAME
 * function the kernel's `op_conv_i8` tests.  That is deliberate: the dispatch
 * decision and the roles decision are then one expression, so they cannot drift
 * apart and silently compute a layer with the wrong input channels.
 *
 * ENCODING OF A GROUPED ROW IN THE LAYER TABLE (verified against
 * src/_old115_backup/yolo_layers_i8.h, the 115-layer depthwise-bearing table,
 * rows 100/102/105/107/110/112):
 *
 *   op    = OP_CONV
 *   nin   = 1                    (number of input BRANCHES, not kernel size)
 *   k     = 3  stride = 1  pad = 1
 *   ic    = 1                    = input channels PER GROUP
 *   oc    = 64/128/256           = number of groups (groups == oc here)
 *   cin[0]= 64/128/256           = TOTAL input channels == oc
 *   wq_off stride between output channels = ic*k2 = 9 bytes, i.e. the layer's
 *   weight block is `oc * 9` int8 values and bias_off - wq_off confirms it
 *   (e.g. 2385104 - 2384528 = 576 = 64*9).
 *
 * The `cin[0] == oc` term in the predicate is what makes "grouped" and "the row
 * cache already holds all the channels this layer needs" the same statement;
 * without it a row with ic==1 but a wider input than output would be dispatched
 * into the grouped path and read channels that were never fetched.
 *
 * Requires yolo_layers_i8.h to be included first (for YoloLayerI8 and the OP_*
 * enum).
 * ===========================================================================
 */
#ifndef YOLO_LAYER_ROLES_H
#define YOLO_LAYER_ROLES_H

#ifdef __cplusplus
extern "C" {
#endif

/* 1 when this OP_CONV is a grouped/depthwise convolution, i.e. one input
 * channel per output channel, which the PL engine executes in its grouped mode.
 *
 * This is the single definition.  `op_conv_i8` calls it to decide whether to run
 * the dense or the grouped inner loop, and the PS application calls it (through
 * `yolo_layer_is_ps`) to decide nobody else has to. */
static inline int yolo_layer_is_grouped_conv(const YoloLayerI8 *L)
{
    return L->op == OP_CONV && (int)L->ic == 1 && (int)L->oc > 1 &&
           (int)L->nin == 1 && (int)L->cin[0] == (int)L->oc;
}

/* 1 when the layer is executed outside the accelerator (on the PS).
 *
 * Nothing is any more.  OP_ATTENTION left in the CBAM rework (it is a real PL
 * operator now) and grouped conv left when `op_conv_i8` learned grouped mode.
 * Returning 1 for either would keep the application cutting the schedule into
 * range calls at those layers -- silently undoing the rework while everything
 * still looked correct -- so this is deliberately the constant that removes all
 * PS operators and collapses the network into a single PL call.
 *
 * The predicate is kept as a function (rather than deleted) because the
 * application's schedule builder and the testbenches are written against it. */
static inline int yolo_layer_is_ps(const YoloLayerI8 *L)
{
    (void)L;
    return 0;
}

/* Same question for a table index. */
static inline int yolo_layer_index_is_ps(int idx)
{
    if (idx < 0 || idx >= (int)YOLO_I8_NUM_LAYERS)
        return 0;
    return yolo_layer_is_ps(&yolo_layers_i8[idx]);
}

/* Bytes one layer's output occupies in `work` (0 for head layers, which write
 * `pred` instead). */
static inline unsigned yolo_layer_out_bytes(const YoloLayerI8 *L)
{
    if (L->out_target)
        return 0u;
    return (unsigned)L->oc * (unsigned)L->oh * (unsigned)L->ow;
}

#ifdef __cplusplus
}
#endif

#endif /* YOLO_LAYER_ROLES_H */
