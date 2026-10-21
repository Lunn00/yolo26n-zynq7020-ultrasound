#include <cmath>
#include <cstdio>
#include <stdint.h>

#include "yolo_layers_i8.h"
#include "yolo_fixed_params_i8.h"

#define YOLO_I8_WEIGHT_BASE 8
#define YOLO_I8_SCALE_BYTES 64264
#define YOLO_I8_CONV_MAX_ROW 5440
#define YOLO_I8_CONV_MAX_OW 160
#ifndef YOLO_I8_CONV_TILE
#define YOLO_I8_CONV_TILE 32
#endif
#define YOLO_I8_MAX_IC 512
#define YOLO_I8_MAX_OC 512
#define YOLO_I8_MAX_K2 9
/* YOLOv8n layer table max ic*k2 is 2304 (64x256x3x3). */
#define YOLO_I8_TILE_WORDS 2304

/* Keep the 7x7 SPPF chain (layers 31..33) on chip and feed layer 34
   directly. Four 128x7x7 banks replace three DDR round trips and the
   repeated short-row reads made once per output-channel tile. */
#ifndef YOLO_I8_FUSE_SPPF_ONCHIP
#define YOLO_I8_FUSE_SPPF_ONCHIP 0
#endif
#define YOLO_I8_SPPF_C 128
#define YOLO_I8_SPPF_H 7
#define YOLO_I8_SPPF_W 7
#define YOLO_I8_SPPF_PLANE (YOLO_I8_SPPF_H * YOLO_I8_SPPF_W)
#define YOLO_I8_SPPF_BANK_BYTES (YOLO_I8_SPPF_C * YOLO_I8_SPPF_PLANE)


#define YOLO_I8_ROW_WORDS 32
/* Row cache geometry: 8 byte-banks per k-slot.  See tools/row_bank_size.py for
   the sizing proof (max ic*((iw+7)>>3) over the layer table is 768, plus one
   group of overflow because nwv can be one group larger than NW). */
#define YOLO_I8_ROW_BANKS 8
#define YOLO_I8_ROW_MAXB  1040
#if YOLO_I8_ROW_MAXB < YOLO_I8_ROW_NEED
#error "row cache smaller than generated YOLO26 layer table requires"
#endif
#if YOLO_I8_TILE_WORDS < YOLO_I8_TILE_NEED
#error "weight tile smaller than generated YOLO26 layer table requires"
#endif
#define YOLO_I8_SIG_N 512
#define YOLO_I8_SIG_HALF 256
#define YOLO_I8_SIG_STEP_F 32.0f

static const float yolo_i8_sigmoid_lut[YOLO_I8_SIG_N + 1] = {
    0.0003353501f, 0.0003459916f, 0.0003569706f, 0.0003682979f, 0.0003799845f, 0.0003920418f, 0.0004044815f, 0.0004173158f,
    0.0004305571f, 0.0004442184f, 0.0004583129f, 0.0004728544f, 0.0004878571f, 0.0005033356f, 0.0005193049f, 0.0005357805f,
    0.0005527786f, 0.0005703157f, 0.0005884088f, 0.0006070756f, 0.0006263342f, 0.0006462033f, 0.0006667023f, 0.0006878512f,
    0.0007096704f, 0.0007321813f, 0.0007554056f, 0.0007793661f, 0.0008040859f, 0.0008295892f, 0.0008559006f, 0.0008830458f,
    0.0009110512f, 0.0009399439f, 0.000969752f, 0.001000504f, 0.001032231f, 0.001064963f, 0.001098731f, 0.001133569f,
    0.00116951f, 0.00120659f, 0.001244843f, 0.001284308f, 0.001325022f, 0.001367026f, 0.001410358f, 0.001455063f,
    0.001501182f, 0.001548761f, 0.001597845f, 0.001648483f, 0.001700722f, 0.001754614f, 0.001810211f, 0.001867566f,
    0.001926735f, 0.001987774f, 0.002050744f, 0.002115703f, 0.002182716f, 0.002251847f, 0.002323163f, 0.002396731f,
    0.002472623f, 0.002550912f, 0.002631674f, 0.002714985f, 0.002800927f, 0.002889581f, 0.002981033f, 0.00307537f,
    0.003172683f, 0.003273065f, 0.003376612f, 0.003483424f, 0.003593603f, 0.003707253f, 0.003824484f, 0.003945407f,
    0.004070138f, 0.004198795f, 0.004331502f, 0.004468384f, 0.004609572f, 0.0047552f, 0.004905406f, 0.005060332f,
    0.005220126f, 0.005384938f, 0.005554925f, 0.005730247f, 0.005911069f, 0.006097562f, 0.006289902f, 0.00648827f,
    0.006692851f, 0.006903838f, 0.007121429f, 0.007345827f, 0.007577241f, 0.007815889f, 0.008061992f, 0.008315778f,
    0.008577485f, 0.008847355f, 0.009125637f, 0.00941259f, 0.009708476f, 0.01001357f, 0.01032815f, 0.01065251f,
    0.01098694f, 0.01133175f, 0.01168726f, 0.01205378f, 0.01243165f, 0.01282121f, 0.01322282f, 0.01363684f,
    0.01406363f, 0.01450358f, 0.01495709f, 0.01542455f, 0.01590639f, 0.01640303f, 0.01691491f, 0.01744248f,
    0.01798621f, 0.01854656f, 0.01912404f, 0.01971913f, 0.02033235f, 0.02096424f, 0.02161533f, 0.02228619f,
    0.02297737f, 0.02368947f, 0.02442309f, 0.02517884f, 0.02595736f, 0.02675928f, 0.02758528f, 0.02843603f,
    0.02931223f, 0.03021459f, 0.03114383f, 0.03210071f, 0.03308598f, 0.03410043f, 0.03514485f, 0.03622006f,
    0.03732689f, 0.03846619f, 0.03963884f, 0.04084572f, 0.04208773f, 0.0433658f, 0.04468087f, 0.0460339f,
    0.04742587f, 0.04885778f, 0.05033063f, 0.05184547f, 0.05340333f, 0.05500529f, 0.05665243f, 0.05834584f,
    0.06008665f, 0.06187599f, 0.06371499f, 0.06560484f, 0.06754669f, 0.06954174f, 0.0715912f, 0.07369627f,
    0.07585818f, 0.07807817f, 0.08035747f, 0.08269734f, 0.08509905f, 0.08756384f, 0.09009299f, 0.09268778f,
    0.09534946f, 0.09807932f, 0.1008786f, 0.1037486f, 0.1066906f, 0.1097058f, 0.1127954f, 0.1159607f,
    0.1192029f, 0.1225232f, 0.1259228f, 0.1294027f, 0.1329642f, 0.1366084f, 0.1403362f, 0.1441489f,
    0.1480472f, 0.1520322f, 0.1561049f, 0.160266f, 0.1645165f, 0.168857f, 0.1732882f, 0.1778109f,
    0.1824255f, 0.1871327f, 0.1919328f, 0.1968262f, 0.2018132f, 0.206894f, 0.2120688f, 0.2173375f,
    0.2227001f, 0.2281565f, 0.2337064f, 0.2393493f, 0.245085f, 0.2509128f, 0.256832f, 0.2628418f,
    0.2689414f, 0.2751297f, 0.2814056f, 0.2877678f, 0.294215f, 0.3007456f, 0.307358f, 0.3140505f,
    0.3208213f, 0.3276683f, 0.3345894f, 0.3415825f, 0.3486451f, 0.3557749f, 0.3629692f, 0.3702254f,
    0.3775407f, 0.3849121f, 0.3923368f, 0.3998116f, 0.4073334f, 0.4148988f, 0.4225046f, 0.4301473f,
    0.4378235f, 0.4455295f, 0.4532618f, 0.4610168f, 0.4687906f, 0.4765797f, 0.4843801f, 0.4921881f,
    0.5f, 0.5078119f, 0.5156199f, 0.5234203f, 0.5312094f, 0.5389832f, 0.5467382f, 0.5544705f,
    0.5621765f, 0.5698527f, 0.5774954f, 0.5851012f, 0.5926666f, 0.6001884f, 0.6076632f, 0.6150879f,
    0.6224593f, 0.6297746f, 0.6370308f, 0.6442251f, 0.6513549f, 0.6584175f, 0.6654106f, 0.6723317f,
    0.6791787f, 0.6859495f, 0.692642f, 0.6992544f, 0.705785f, 0.7122322f, 0.7185944f, 0.7248703f,
    0.7310586f, 0.7371582f, 0.743168f, 0.7490872f, 0.754915f, 0.7606507f, 0.7662936f, 0.7718435f,
    0.7772999f, 0.7826625f, 0.7879312f, 0.793106f, 0.7981868f, 0.8031738f, 0.8080672f, 0.8128673f,
    0.8175745f, 0.8221891f, 0.8267118f, 0.831143f, 0.8354835f, 0.839734f, 0.8438951f, 0.8479678f,
    0.8519528f, 0.8558511f, 0.8596638f, 0.8633916f, 0.8670358f, 0.8705973f, 0.8740772f, 0.8774768f,
    0.8807971f, 0.8840393f, 0.8872046f, 0.8902942f, 0.8933094f, 0.8962514f, 0.8991214f, 0.9019207f,
    0.9046505f, 0.9073122f, 0.909907f, 0.9124362f, 0.914901f, 0.9173027f, 0.9196425f, 0.9219218f,
    0.9241418f, 0.9263037f, 0.9284088f, 0.9304583f, 0.9324533f, 0.9343952f, 0.936285f, 0.938124f,
    0.9399133f, 0.9416542f, 0.9433476f, 0.9449947f, 0.9465967f, 0.9481545f, 0.9496694f, 0.9511422f,
    0.9525741f, 0.9539661f, 0.9553191f, 0.9566342f, 0.9579123f, 0.9591543f, 0.9603612f, 0.9615338f,
    0.9626731f, 0.9637799f, 0.9648552f, 0.9658996f, 0.966914f, 0.9678993f, 0.9688562f, 0.9697854f,
    0.9706878f, 0.971564f, 0.9724147f, 0.9732407f, 0.9740426f, 0.9748212f, 0.9755769f, 0.9763105f,
    0.9770226f, 0.9777138f, 0.9783847f, 0.9790358f, 0.9796676f, 0.9802809f, 0.980876f, 0.9814534f,
    0.9820138f, 0.9825575f, 0.9830851f, 0.983597f, 0.9840936f, 0.9845754f, 0.9850429f, 0.9854964f,
    0.9859364f, 0.9863632f, 0.9867772f, 0.9871788f, 0.9875683f, 0.9879462f, 0.9883127f, 0.9886682f,
    0.9890131f, 0.9893475f, 0.9896718f, 0.9899864f, 0.9902915f, 0.9905874f, 0.9908744f, 0.9911526f,
    0.9914225f, 0.9916842f, 0.991938f, 0.9921841f, 0.9924228f, 0.9926542f, 0.9928786f, 0.9930962f,
    0.9933071f, 0.9935117f, 0.9937101f, 0.9939024f, 0.9940889f, 0.9942698f, 0.9944451f, 0.9946151f,
    0.9947799f, 0.9949397f, 0.9950946f, 0.9952448f, 0.9953904f, 0.9955316f, 0.9956685f, 0.9958012f,
    0.9959299f, 0.9960546f, 0.9961755f, 0.9962927f, 0.9964064f, 0.9965166f, 0.9966234f, 0.9967269f,
    0.9968273f, 0.9969246f, 0.997019f, 0.9971104f, 0.9971991f, 0.997285f, 0.9973683f, 0.9974491f,
    0.9975274f, 0.9976033f, 0.9976768f, 0.9977482f, 0.9978173f, 0.9978843f, 0.9979493f, 0.9980122f,
    0.9980733f, 0.9981324f, 0.9981898f, 0.9982454f, 0.9982993f, 0.9983515f, 0.9984022f, 0.9984512f,
    0.9984988f, 0.9985449f, 0.9985896f, 0.998633f, 0.998675f, 0.9987157f, 0.9987552f, 0.9987934f,
    0.9988305f, 0.9988664f, 0.9989013f, 0.998935f, 0.9989678f, 0.9989995f, 0.9990302f, 0.9990601f,
    0.9990889f, 0.999117f, 0.9991441f, 0.9991704f, 0.9991959f, 0.9992206f, 0.9992446f, 0.9992678f,
    0.9992903f, 0.9993121f, 0.9993333f, 0.9993538f, 0.9993737f, 0.9993929f, 0.9994116f, 0.9994297f,
    0.9994472f, 0.9994642f, 0.9994807f, 0.9994967f, 0.9995121f, 0.9995271f, 0.9995417f, 0.9995558f,
    0.9995694f, 0.9995827f, 0.9995955f, 0.999608f, 0.99962f, 0.9996317f, 0.999643f, 0.999654f,
    0.999654f,
};

static float sigmoid_fast(float x)
{
#pragma HLS INLINE
    if (x > 8.0f)
        x = 8.0f;
    if (x < -8.0f)
        x = -8.0f;
    const float xf = x * YOLO_I8_SIG_STEP_F;
    int xi = (int)floorf(xf);
    if (xi < -YOLO_I8_SIG_HALF)
        xi = -YOLO_I8_SIG_HALF;
    if (xi > YOLO_I8_SIG_HALF - 1)
        xi = YOLO_I8_SIG_HALF - 1;
    const float frac = xf - (float)xi;
    const int idx = xi + YOLO_I8_SIG_HALF;
    const float v0 = yolo_i8_sigmoid_lut[idx];
    const float v1 = yolo_i8_sigmoid_lut[idx + 1];
    return v0 + (v1 - v0) * frac;
}

static float silu_i8(float x)
{
#pragma HLS INLINE
    return x * sigmoid_fast(x);
}

static int32_t requant_q31(int32_t x, int32_t multiplier, int32_t rshift)
{
#pragma HLS INLINE
    const int64_t prod = (int64_t)x * (int64_t)multiplier;
    if (rshift <= 0) {
        const int lshift = -rshift;
        if (lshift >= 31)
            return x < 0 ? (int32_t)0x80000000u : 0x7fffffff;
        const int64_t v = prod << lshift;
        if (v > 0x7fffffffLL)
            return 0x7fffffff;
        if (v < -2147483648LL)
            return (int32_t)0x80000000u;
        return (int32_t)v;
    }
    if (rshift >= 63)
        return 0;
    const int64_t round = (int64_t)1 << (rshift - 1);
    if (prod >= 0)
        return (int32_t)((prod + round) >> rshift);
    return (int32_t)(-(((-prod) + round) >> rshift));
}

/* Integer SiLU. Input/output are signed Q10.  Seventeen sigmoid knots at
   integer x=-8..8 are linearly interpolated; no floating-point operator is
   present in the synthesized path. */
static int32_t silu_q10(int32_t x)
{
#pragma HLS INLINE
    static const uint16_t sig_q15[17] = {
        11, 30, 81, 221, 589, 1554, 3906, 8813, 16384,
        23955, 28862, 31214, 32179, 32547, 32687, 32738, 32757
    };
    if (x <= -(8 << YOLO_I8_HEAD_FRAC_BITS))
        return (int32_t)(((int64_t)x * sig_q15[0]) >> 15);
    if (x >= (8 << YOLO_I8_HEAD_FRAC_BITS))
        return (int32_t)(((int64_t)x * sig_q15[16]) >> 15);
    int integer = x >> YOLO_I8_HEAD_FRAC_BITS;
    int frac = x - (integer << YOLO_I8_HEAD_FRAC_BITS);
    const int idx = integer + 8;
    const int32_t y0 = sig_q15[idx];
    const int32_t dy = (int32_t)sig_q15[idx + 1] - y0;
    const int32_t sigmoid = y0 +
        ((dy * frac + (1 << (YOLO_I8_HEAD_FRAC_BITS - 1))) >> YOLO_I8_HEAD_FRAC_BITS);
    const int64_t prod = (int64_t)x * sigmoid;
    return prod >= 0 ? (int32_t)((prod + (1 << 14)) >> 15)
                     : (int32_t)(-(((-prod) + (1 << 14)) >> 15));
}

static void op_copy_i8(const uint8_t *input, uint8_t *work, const YoloLayerI8 &L)
{
#pragma HLS INLINE
    const int nwords = YOLO_I8_INPUT_BYTES >> 3;
    const uint64_t *src64 = (const uint64_t *)input;
    uint64_t *dst64 = (uint64_t *)(work + L.out_off);
    for (int i = 0; i < nwords; i++) {
#pragma HLS PIPELINE II=1
        dst64[i] = src64[i];
    }
}

static void load_weight_tile(const int8_t *wbase,
                             int8_t tile[YOLO_I8_CONV_TILE][YOLO_I8_TILE_WORDS],
                             int tmax, int ocb, int ic, int k2)
{
#pragma HLS INLINE
    static uint64_t wcache_word[YOLO_I8_TILE_WORDS / 8];
    for (int t = 0; t < tmax; t++) {
        const int woff = (ocb + t) * ic * k2;
        const int nwords = ic * k2;
        if ((nwords & 7) == 0) {
            const int nw = nwords >> 3;
            const uint64_t *w64 = (const uint64_t *)(wbase + woff);
            for (int w = 0; w < nw; w++) {
#pragma HLS PIPELINE
                wcache_word[w] = w64[w];
            }
            for (int p = 0; p < nwords; p++) {
#pragma HLS PIPELINE
                const uint64_t v = wcache_word[p >> 3];
                tile[t][p] = (int8_t)((v >> ((p & 7) << 3)) & 0xFFu);
            }
        } else {
            for (int p = 0; p < nwords; p++) {
#pragma HLS PIPELINE
                tile[t][p] = wbase[woff + p];
            }
        }
    }
}

static void op_conv_i8(const int8_t *weights, const uint8_t *fixed_params,
                       uint8_t *work, int16_t *pred, const YoloLayerI8 &L,
                       const YoloFixedParamDesc &Q)
{
#pragma HLS INLINE
    static int8_t wcache[YOLO_I8_CONV_TILE][YOLO_I8_TILE_WORDS];
    /* Row cache: YOLO_I8_ROW_BANKS byte-banks per k-slot.  Byte position n of
       the padded row lives in bank (n & 7) at slot (n >> 3); because the padded
       per-channel stride is a multiple of 8 every channel starts at bank 0, so a
       64-bit DDR word maps onto all 8 banks at one slot.  That lets the fetch and
       the byte-unpack be a single pipelined loop (halving the loop entries, the
       dominant cost per the on-board profile) and removes rowbufB entirely. */
    static uint8_t rowA[3][YOLO_I8_ROW_BANKS][YOLO_I8_ROW_MAXB];
    static int16_t in_zp_buf[YOLO_I8_MAX_IC];
    static int32_t bias_buf[YOLO_I8_MAX_OC];
    static int32_t mul0_buf[YOLO_I8_MAX_OC];
    static int32_t shift0_buf[YOLO_I8_MAX_OC];
    static int32_t mul1_buf[YOLO_I8_MAX_OC];
    static int32_t shift1_buf[YOLO_I8_MAX_OC];
    static int16_t out_zp_buf[YOLO_I8_MAX_OC];
    static uint8_t out_row0[YOLO_I8_CONV_TILE][YOLO_I8_CONV_MAX_OW / 2];
    static uint8_t out_row1[YOLO_I8_CONV_TILE][YOLO_I8_CONV_MAX_OW / 2];
    static uint64_t row_word[YOLO_I8_ROW_WORDS];
    static uint8_t resid8[YOLO_I8_CONV_MAX_OW];
    static uint8_t resq8[YOLO_I8_CONV_MAX_OW];
    int32_t acc0[YOLO_I8_CONV_TILE];
    int32_t acc1[YOLO_I8_CONV_TILE];
#pragma HLS ARRAY_PARTITION variable=wcache dim=1 complete
#pragma HLS ARRAY_PARTITION variable=rowA dim=1 complete
#pragma HLS ARRAY_PARTITION variable=rowA dim=2 complete
#pragma HLS RESOURCE variable=out_row0 core=RAM_T2P
#pragma HLS RESOURCE variable=out_row1 core=RAM_T2P
#pragma HLS ARRAY_PARTITION variable=acc0 complete
#pragma HLS ARRAY_PARTITION variable=acc1 complete

    const int k2 = L.k * L.k;
    const int oc = (int)L.oc;
    const int ic = (int)L.ic;
    const int k = (int)L.k;
    const int ih = (int)L.ih;
    const int iw = (int)L.iw;
    const int oh = (int)L.oh;
    const int ow = (int)L.ow;
    const int NW = (iw + 7) >> 3;      /* padded groups per channel row */
    const uint64_t *src64w = (const uint64_t *)work;
#if YOLO_I8_FUSE_SPPF_ONCHIP
    const bool use_sppf_local =
        L.nin == 4 && L.ic == 512 && L.ih == YOLO_I8_SPPF_H &&
        L.iw == YOLO_I8_SPPF_W && L.k == 1;
#endif

    const int8_t *wbase = weights + YOLO_I8_WEIGHT_BASE + (int)L.wq_off;
    const int16_t *in_zp = (const int16_t *)(fixed_params + Q.in0_zp_off);
    const int16_t *out_zp_all = (const int16_t *)(fixed_params + Q.out_zp_off);
    const int32_t *bias = (const int32_t *)(fixed_params + Q.bias_off);
    const int32_t *mul0 = (const int32_t *)(fixed_params + Q.mul0_off);
    const int32_t *shift0 = (const int32_t *)(fixed_params + Q.shift0_off);
    const int32_t *mul1 = (const int32_t *)(fixed_params + Q.mul1_off);
    const int32_t *shift1 = (const int32_t *)(fixed_params + Q.shift1_off);
    const int16_t *add_in_zp =
        (const int16_t *)(fixed_params + Q.in1_zp_off);
    const int16_t *add_out_zp =
        (const int16_t *)(fixed_params + Q.add_out_zp_off);
    const int32_t *add_mul0 =
        (const int32_t *)(fixed_params + Q.add_mul0_off);
    const int32_t *add_shift0 =
        (const int32_t *)(fixed_params + Q.add_shift0_off);
    const int32_t *add_mul1 =
        (const int32_t *)(fixed_params + Q.add_mul1_off);
    const int32_t *add_shift1 =
        (const int32_t *)(fixed_params + Q.add_shift1_off);

    /* Hoist the per-channel quant constants into BRAM once per layer so the
       row-load and quantize inner loops never read scales from DDR. */
    for (int c = 0; c < ic; c++) {
#pragma HLS PIPELINE
        in_zp_buf[c] = in_zp[c];
    }
    for (int o = 0; o < oc; o++) {
#pragma HLS PIPELINE
        bias_buf[o] = bias[o];
        mul0_buf[o] = mul0[o];
        shift0_buf[o] = shift0[o];
        if (L.silu && !L.out_target) {
            mul1_buf[o] = mul1[o];
            shift1_buf[o] = shift1[o];
        }
        if (!L.out_target)
            out_zp_buf[o] = out_zp_all[o];
    }

#if YOLO_I8_FUSE_SPPF_ONCHIP
    /* Layer 34 only needs 512 weight bytes per output channel.  Reuse the
       otherwise idle tail of sixteen existing wcache banks as four 128x7x7
       SPPF feature banks.  This removes the three intermediate DDR writes
       and their later reads without allocating another full-frame BRAM. */
    if (use_sppf_local) {
        const int src_off = (int)L.in_off[0];
        for (int i = 0; i < YOLO_I8_SPPF_BANK_BYTES; i++) {
#pragma HLS PIPELINE II=1
            const int wb = i & 3;
            const int wa = 512 + (i >> 2);
            wcache[wb][wa] = (int8_t)work[src_off + i];
        }
        for (int stage = 1; stage < 4; stage++) {
            for (int c = 0; c < YOLO_I8_SPPF_C; c++) {
                const int plane_off = c * YOLO_I8_SPPF_PLANE;
                for (int y = 0; y < YOLO_I8_SPPF_H; y++) {
                    for (int x = 0; x < YOLO_I8_SPPF_W; x++) {
                        uint8_t m = 0;
                        for (int ky = 0; ky < 5; ky++) {
                            const int iy = y + ky - 2;
                            if (iy < 0 || iy >= YOLO_I8_SPPF_H)
                                continue;
                            for (int kx = 0; kx < 5; kx++) {
#pragma HLS PIPELINE II=1
                                const int ix = x + kx - 2;
                                if (ix < 0 || ix >= YOLO_I8_SPPF_W)
                                    continue;
                                const int ri = plane_off +
                                    iy * YOLO_I8_SPPF_W + ix;
                                const int rb = (stage - 1) * 4 + (ri & 3);
                                const int ra = 512 + (ri >> 2);
                                const uint8_t v = (uint8_t)wcache[rb][ra];
                                if (v > m)
                                    m = v;
                            }
                        }
                        const int wi = plane_off + y * YOLO_I8_SPPF_W + x;
                        const int wb = stage * 4 + (wi & 3);
                        const int wa = 512 + (wi >> 2);
                        wcache[wb][wa] = (int8_t)m;
                    }
                }
            }
        }
    }
#endif

    for (int ocb = 0; ocb < oc; ocb += YOLO_I8_CONV_TILE) {
        int tmax = oc - ocb;
        if (tmax > YOLO_I8_CONV_TILE)
            tmax = YOLO_I8_CONV_TILE;
        /* Weight tile is loaded once per output-channel block and reused for
           every output row, instead of being re-read from DDR each row. */
        load_weight_tile(wbase, wcache, tmax, ocb, ic, k2);

        /* Line buffer: reuse the k-stride overlapped rows from the previous
           output row; only load the new row(s) sliding in. */
        int slot_iy[3] = {-1, -1, -1};
        int slot_src_iy[3][4] = {
            {-1, -1, -1, -1},
            {-1, -1, -1, -1},
            {-1, -1, -1, -1}
        };
        int base = 0;
        for (int oy = 0; oy < oh; oy++) {
            int r = base;
            for (int ky = 0; ky < k; ky++) {
                const int iy = oy * L.stride - L.pad + ky;
                if (iy < 0 || iy >= ih) {
                    slot_iy[r] = -1;
                    for (int j = 0; j < 4; j++)
                        slot_src_iy[r][j] = -1;
                } else if (slot_iy[r] != iy) {
                    slot_iy[r] = iy;
                    int cbase = 0;
                    for (int j = 0; j < 4; j++) {
                        if (j < (int)L.nin) {
                            const int channels = (int)L.cin[j];
                            const int up2x = (int)L.in_up2x[j];
                            const int src_h = up2x ? (ih >> 1) : ih;
                            const int src_w = up2x ? (iw >> 1) : iw;
                            const int src_y = up2x ? (iy >> 1) : iy;
                            if (slot_src_iy[r][j] != src_y) {
                                slot_src_iy[r][j] = src_y;
                                if (up2x) {
                                    /* Half-width source row, each byte feeding two
                                       row positions: only two layers use this, so the
                                       per-channel form is kept. */
                                    for (int lc = 0; lc < channels; lc++) {
                                        const int src = (int)L.in_off[j] +
                                            lc * src_h * src_w + src_y * src_w;
                                        const int wb = src >> 3;
                                        const int off = src & 7;
                                        const int nwv = (off + src_w + 7) >> 3;
                                        for (int w = 0; w < YOLO_I8_ROW_WORDS; w++) {
#pragma HLS PIPELINE II=1
                                            if (w < nwv)
                                                row_word[w] = src64w[wb + w];
                                        }
                                        const int c = cbase + lc;
                                        for (int x2 = 0; x2 < iw; x2++) {
#pragma HLS PIPELINE II=1
                                            const int idx = off + (x2 >> 1);
                                            const uint8_t b = (uint8_t)
                                                (row_word[idx >> 3] >> ((idx & 7) << 3));
                                            rowA[r][x2 & 7][c * NW + (x2 >> 3)] = b;
                                        }
                                    }
                                } else {
                                    /* Channel-batched row fetch: outer over the word
                                       index, inner over the input channels.  The two
                                       adjacent reads per iteration are independent
                                       across iterations so the memory pipeline keeps
                                       several requests in flight, and the inner loop
                                       runs `channels` times instead of ~3, which is
                                       what hides the DDR latency.  Each iteration
                                       writes exactly one 8-byte group of one channel:
                                       bank j gets logical byte (8*w + j). */
                                    const int srcbase = (int)L.in_off[j] + src_y * src_w;
                                    const int cstride = src_h * src_w;
                                    for (int w = 0; w < NW; w++) {
                                        for (int lc = 0; lc < channels; lc++) {
#pragma HLS PIPELINE II=1
                                            const int src = srcbase + lc * cstride;
                                            const int wb = src >> 3;
                                            const int off = src & 7;
                                            const uint64_t lo = src64w[wb + w];
                                            const uint64_t hi = src64w[wb + w + 1];
                                            const uint64_t o = (off == 0) ? lo
                                                : ((lo >> (off << 3)) |
                                                   (hi << ((8 - off) << 3)));
                                            const int g = (cbase + lc) * NW + w;
                                            rowA[r][0][g] = (uint8_t)(o);
                                            rowA[r][1][g] = (uint8_t)(o >> 8);
                                            rowA[r][2][g] = (uint8_t)(o >> 16);
                                            rowA[r][3][g] = (uint8_t)(o >> 24);
                                            rowA[r][4][g] = (uint8_t)(o >> 32);
                                            rowA[r][5][g] = (uint8_t)(o >> 40);
                                            rowA[r][6][g] = (uint8_t)(o >> 48);
                                            rowA[r][7][g] = (uint8_t)(o >> 56);
                                        }
                                    }
                                }
                            }
                            cbase += channels;
                        }
                    }
                }
                r++;
                if (r >= k)
                    r = 0;
            }

            for (int ox = 0; ox < ow; ox += 2) {
                /* Fully unrolled so the 128 register clears cost one cycle; the
                   accumulators must NOT be initialised from inside the MAC loop
                   because that puts a wide mux in front of the DSP accumulator
                   and collapses the pipeline (that cost -1.0 ns of WNS). */
                for (int t = 0; t < YOLO_I8_CONV_TILE; t++) {
#pragma HLS UNROLL
                    acc0[t] = 0;
                    acc1[t] = 0;
                }

                /* 32 output channels x 2 pixels per cycle. The row buffer is
                   a dual-port RAM, so both pixels can be read in the same
                   pipelined input-channel iteration. */
                int r = base;
                for (int ky = 0; ky < k; ky++) {
                    if (slot_iy[r] < 0) {
                        r++;
                        if (r >= k)
                            r = 0;
                        continue;
                    }
                    for (int kx = 0; kx < k; kx++) {
                        const int ix0 = ox * L.stride - L.pad + kx;
                        const int ix1 = (ox + 1) * L.stride - L.pad + kx;
                        const int ok0 = (ix0 >= 0 && ix0 < iw) ? 1 : 0;
                        const int ok1 = (ix1 >= 0 && ix1 < iw) ? 1 : 0;
                        if (!ok0 && !ok1)
                            continue;
                        const int ixc0 = ok0 ? ix0 : 0;
                        const int ixc1 = ok1 ? ix1 : 0;
                        for (int c = 0; c < ic; c++) {
#pragma HLS PIPELINE II=1
                            const int z = (int)in_zp_buf[c];
                            const int v0 = ok0 ?
                                ((int)rowA[r][ixc0 & 7][c * NW + (ixc0 >> 3)] - z) : 0;
                            const int v1 = ok1 ?
                                ((int)rowA[r][ixc1 & 7][c * NW + (ixc1 >> 3)] - z) : 0;
                            const int p = c * k2 + ky * k + kx;
                            for (int t = 0; t < YOLO_I8_CONV_TILE; t++) {
#pragma HLS UNROLL
                                if (t < tmax) {
                                    const int8_t w = wcache[t][p];
                                    int32_t prod0 = (int32_t)w * v0;
#pragma HLS RESOURCE variable=prod0 core=Mul_DSP
                                    acc0[t] += prod0;
                                    int32_t prod1 = (int32_t)w * v1;
#pragma HLS RESOURCE variable=prod1 core=Mul_DSP
                                    acc1[t] += prod1;
                                }
                            }
                        }
                    }
                    r++;
                    if (r >= k)
                        r = 0;
                }

                if (L.out_target) {
                    for (int t = 0; t < tmax; t++) {
#pragma HLS PIPELINE II=2
                        const int o = ocb + t;
                        const int32_t biased0 = acc0[t] + bias_buf[o];
                        const int32_t biased1 = acc1[t] + bias_buf[o];
                        int32_t q0 = requant_q31(biased0, mul0_buf[o], shift0_buf[o]);
                        int32_t q1 = requant_q31(biased1, mul0_buf[o], shift0_buf[o]);
                        if (q0 < -32768) q0 = -32768;
                        if (q0 > 32767) q0 = 32767;
                        if (q1 < -32768) q1 = -32768;
                        if (q1 > 32767) q1 = 32767;
                        pred[L.out_off + o * L.out_stride + oy * ow + ox] = (int16_t)q0;
                        if (ox + 1 < ow)
                            pred[L.out_off + o * L.out_stride + oy * ow + ox + 1] = (int16_t)q1;
                    }
                } else {
                    for (int t = 0; t < tmax; t++) {
#pragma HLS PIPELINE II=1
                        const int o = ocb + t;
                        const int32_t biased0 = acc0[t] + bias_buf[o];
                        const int32_t biased1 = acc1[t] + bias_buf[o];
                        int32_t q0 = requant_q31(biased0, mul0_buf[o], shift0_buf[o]);
                        int32_t q1 = requant_q31(biased1, mul0_buf[o], shift0_buf[o]);
                        if (L.silu) {
                            q0 = requant_q31(silu_q10(q0), mul1_buf[o], shift1_buf[o]);
                            q1 = requant_q31(silu_q10(q1), mul1_buf[o], shift1_buf[o]);
                        }
                        q0 += (int)out_zp_buf[o];
                        q1 += (int)out_zp_buf[o];
                        if (q0 < 0) q0 = 0;
                        if (q0 > 255) q0 = 255;
                        if (q1 < 0) q1 = 0;
                        if (q1 > 255) q1 = 255;
                        out_row0[t][ox >> 1] = (uint8_t)q0;
                        out_row1[t][ox >> 1] = (uint8_t)q1;
                    }
                }
            }
            /* Flush the buffered row plane by plane as 32-bit words so each
               DDR write is a burst instead of one byte transaction per pixel. */
            if (!L.out_target) {
                const int ow4 = ow >> 2;
                for (int t = 0; t < tmax; t++) {
                    const int o = ocb + t;
                    const int doff =
                        (int)(L.out_off + o * L.out_stride) + oy * ow;
                    uint32_t *d32 = (uint32_t *)(work + doff);
                    if (L.fuse_add) {
                        const int aoff = (int)L.add_off + o * oh * ow + oy * ow;
                        const uint8_t *a8 = work + aoff;
                        uint8_t *d8 = work + doff;
                        const int zc = (int)out_zp_buf[o];
                        const int zr = (int)add_in_zp[o];
                        const int zo = (int)add_out_zp[o];
                        const int32_t mc = add_mul0[o];
                        const int32_t sc = add_shift0[o];
                        const int32_t mr = add_mul1[o];
                        const int32_t sr = add_shift1[o];
                        /* Stage the residual row on chip, then requantise one
                           byte per cycle.  Reading a8[x] and writing d8[x]
                           inside a single pipelined loop made HLS assume
                           aliasing and forced II=24; parallelising all four
                           bytes to reach II=1 cost ~9.4k LUT.  One byte per
                           cycle keeps 2 requant chains and still runs 2.7x
                           faster than the II=24 version. */
                        for (int x = 0; x < ow; x++) {
#pragma HLS PIPELINE II=1
                            resid8[x] = a8[x];
                        }
                        for (int x = 0; x < ow; x++) {
#pragma HLS PIPELINE II=1
                            const int idx = x >> 1;
                            const int cv = (x & 1) ?
                                (int)out_row1[t][idx] :
                                (int)out_row0[t][idx];
                            const int rv = (int)resid8[x];
                            int q = requant_q31(cv - zc, mc, sc) +
                                    requant_q31(rv - zr, mr, sr) + zo;
                            if (q < 0)
                                q = 0;
                            if (q > 255)
                                q = 255;
                            resq8[x] = (uint8_t)q;
                        }
                        if ((doff & 3) == 0) {
                            for (int w = 0; w < ow4; w++) {
#pragma HLS PIPELINE II=1
                                d32[w] = (uint32_t)resq8[w << 2] |
                                         ((uint32_t)resq8[(w << 2) + 1] << 8) |
                                         ((uint32_t)resq8[(w << 2) + 2] << 16) |
                                         ((uint32_t)resq8[(w << 2) + 3] << 24);
                            }
                            for (int x = ow4 << 2; x < ow; x++)
                                d8[x] = resq8[x];
                        } else {
                            for (int x = 0; x < ow; x++) {
#pragma HLS PIPELINE II=1
                                d8[x] = resq8[x];
                            }
                        }
                    } else {
                        uint8_t *d8 = work + doff;
                        if ((doff & 3) == 0) {
                            for (int w = 0; w < ow4; w++) {
#pragma HLS PIPELINE
                                uint32_t word =
                                    (uint32_t)out_row0[t][w << 1] |
                                    ((uint32_t)out_row1[t][w << 1] << 8) |
                                    ((uint32_t)out_row0[t][(w << 1) + 1] << 16) |
                                    ((uint32_t)out_row1[t][(w << 1) + 1] << 24);
                                d32[w] = word;
                            }
                            for (int x = ow4 << 2; x < ow; x++) {
                                const int idx = x >> 1;
                                d8[x] = (x & 1) ? out_row1[t][idx]
                                                : out_row0[t][idx];
                            }
                        } else {
                            for (int x = 0; x < ow; x++) {
#pragma HLS PIPELINE II=1
                                const int idx = x >> 1;
                                d8[x] = (x & 1) ? out_row1[t][idx]
                                                : out_row0[t][idx];
                            }
                        }
                    }
                }
            }
            base += L.stride;
            if (base >= k)
                base -= k;
        }
    }
}

static void op_split_i8(uint8_t *work, const YoloLayerI8 &L)
{
#pragma HLS INLINE
    /* The generated allocator makes both outputs channel-contiguous views of
       the input and extends the backing allocation's lifetime. */
    (void)work;
    (void)L;
}

static void op_add_i8(uint8_t *work, const uint8_t *fixed_params,
                      const YoloLayerI8 &L, const YoloFixedParamDesc &Q)
{
#pragma HLS INLINE
    const int plane = L.oh * L.ow;
    const int oc = (int)L.oc;
    const int16_t *za = (const int16_t *)(fixed_params + Q.in0_zp_off);
    const int16_t *zb = (const int16_t *)(fixed_params + Q.in1_zp_off);
    const int16_t *zo = (const int16_t *)(fixed_params + Q.out_zp_off);
    const int32_t *ma = (const int32_t *)(fixed_params + Q.mul0_off);
    const int32_t *sa = (const int32_t *)(fixed_params + Q.shift0_off);
    const int32_t *mb = (const int32_t *)(fixed_params + Q.mul1_off);
    const int32_t *sb = (const int32_t *)(fixed_params + Q.shift1_off);
    static uint64_t tmpa64[512];
    static uint64_t tmpb64[512];
    static uint64_t tmpout64[512];

    for (int o = 0; o < oc; o++) {
        const int aoff = (int)L.in_off[0] + o * plane;
        const int boff = (int)L.in_off[1] + o * plane;
        const int doff = (int)L.out_off + o * plane;
        const int nw = (int)plane >> 3;
        const uint64_t *a64 = (const uint64_t *)(work + aoff);
        const uint64_t *b64 = (const uint64_t *)(work + boff);
        uint64_t *d64 = (uint64_t *)(work + doff);
        for (int base = 0; base < nw; base += 512) {
            int n = nw - base;
            if (n > 512)
                n = 512;
            for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE
                tmpa64[i] = a64[base + i];
            }
            for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE
                tmpb64[i] = b64[base + i];
            }
            for (int i = 0; i < n; i++) {
                uint64_t word = 0;
                for (int j = 0; j < 8; j++) {
#pragma HLS PIPELINE
                    const int av = (int)((tmpa64[i] >> (j << 3)) & 0xFFu) - za[o];
                    const int bv = (int)((tmpb64[i] >> (j << 3)) & 0xFFu) - zb[o];
                    int q = requant_q31(av, ma[o], sa[o]) +
                            requant_q31(bv, mb[o], sb[o]) + zo[o];
                    if (q < 0)
                        q = 0;
                    if (q > 255)
                        q = 255;
                    word |= (uint64_t)q << (j << 3);
                }
                tmpout64[i] = word;
            }
            for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE
                d64[base + i] = tmpout64[i];
            }
        }
    }
}

static void op_concat_i8(uint8_t *work, const YoloLayerI8 &L)
{
#pragma HLS INLINE
    const int plane = L.oh * L.ow;
    uint8_t *dst = work + L.out_off;
    static uint64_t tmp64[512];
    int cum = 0;
    for (int j = 0; j < (int)L.nin; j++) {
        const int c = (int)L.cin[j];
        const uint8_t *src = work + L.in_off[j];
        const int nw_plane = (int)plane >> 3;
        for (int oi = 0; oi < c; oi++) {
            const uint64_t *s64 = (const uint64_t *)(src + oi * plane);
            uint64_t *d64 = (uint64_t *)(dst + (cum + oi) * plane);
            for (int base = 0; base < nw_plane; base += 512) {
                int n = nw_plane - base;
                if (n > 512)
                    n = 512;
                for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE
                    tmp64[i] = s64[base + i];
                }
                for (int i = 0; i < n; i++) {
#pragma HLS PIPELINE
                    d64[base + i] = tmp64[i];
                }
            }
        }
        cum += c;
    }
}

static void op_up2x_i8(uint8_t *work, const YoloLayerI8 &L)
{
#pragma HLS INLINE
    for (int c = 0; c < (int)L.oc; c++) {
        for (int y = 0; y < (int)L.oh; y++) {
            for (int x = 0; x < (int)L.ow; x++) {
#pragma HLS PIPELINE
                work[L.out_off + (c * L.oh + y) * L.ow + x] =
                    work[L.in_off[0] + (c * L.ih + y / 2) * L.iw + x / 2];
            }
        }
    }
}

static void op_maxpool_i8(uint8_t *work, const YoloLayerI8 &L)
{
#pragma HLS INLINE
    /* All deployed maxpool layers use a 5x5 window on a 7x7 map.  Keep
       the five source rows local, reduce them vertically once per column,
       then reduce the five horizontal neighbours.  Max is associative and
       the same zero padding is applied at both edges. */
    static uint8_t poolbuf[5][7];
#pragma HLS ARRAY_PARTITION variable=poolbuf complete dim=1
    static uint8_t vertical[7];
    /* The three SPPF pooling entries in the fixed 111-layer table are all
       128 x 7 x 7 with stride 1 and pad 2.  Keeping their geometry constant
       prevents a dynamic 64-bit address multiply in the pipelined store. */
    const int iw = 7;
    const int ih = 7;
    const int oh = 7;
    const int ow = 7;
    for (int c = 0; c < (int)L.oc; c++) {
        const int in_base = (int)L.in_off[0] + c * ih * iw;
        for (int y = 0; y < oh; y++) {
            for (int ky = 0; ky < 5; ky++) {
                const int iy = y + ky - 2;
                if (iy < 0 || iy >= ih)
                    continue;
                const uint8_t *s8 = work + in_base + iy * iw;
                for (int x = 0; x < iw; x++) {
#pragma HLS PIPELINE
                    poolbuf[ky][x] = s8[x];
                }
            }
            for (int x = 0; x < iw; x++) {
#pragma HLS PIPELINE II=1
                uint8_t m = 0;
                for (int ky = 0; ky < 5; ky++) {
#pragma HLS UNROLL
                    const int iy = y + ky - 2;
                    if (iy >= 0 && iy < ih) {
                        const uint8_t v = poolbuf[ky][x];
                        if (v > m)
                            m = v;
                    }
                }
                vertical[x] = m;
            }
            const int out_row = (int)L.out_off + (c * oh + y) * ow;
            uint8_t *out8 = work + out_row;
            for (int x = 0; x < ow; x++) {
#pragma HLS PIPELINE II=1
                uint8_t m = 0;
                for (int kx = 0; kx < 5; kx++) {
                    const int ix = x + kx - 2;
                    if (ix >= 0 && ix < iw) {
                        const uint8_t v = vertical[ix];
                        if (v > m)
                            m = v;
                    }
                }
                out8[x] = m;
            }
        }
    }
}

void yolo8n_hls_i8(const uint8_t *input, const int8_t *weights,
                   const uint8_t *fixed_params, uint8_t *work, int16_t *pred)
{
#pragma HLS INTERFACE m_axi port=input offset=slave bundle=gmem0 depth=150528 num_read_outstanding=8 max_read_burst_length=256 max_widen_bitwidth=64
#pragma HLS INTERFACE m_axi port=weights offset=slave bundle=gmem0 depth=3238768 num_read_outstanding=8 max_read_burst_length=256 max_widen_bitwidth=64
#pragma HLS INTERFACE m_axi port=fixed_params offset=slave bundle=gmem0 depth=208676 num_read_outstanding=8 max_read_burst_length=256 max_widen_bitwidth=64
#pragma HLS INTERFACE m_axi port=work offset=slave bundle=gmem1 depth=1122688 num_read_outstanding=8 num_write_outstanding=8 max_read_burst_length=256 max_write_burst_length=256 max_widen_bitwidth=64
#pragma HLS INTERFACE m_axi port=pred offset=slave bundle=gmem1 depth=5145 num_write_outstanding=8 max_write_burst_length=256 max_widen_bitwidth=64
#pragma HLS INTERFACE s_axilite port=return
#pragma HLS INTERFACE ap_ctrl_hs port=return

    for (int li = 0; li < YOLO_I8_NUM_LAYERS; li++) {
        const YoloLayerI8 &L = yolo_layers_i8[li];
        const YoloFixedParamDesc &Q = yolo_fixed_desc[li];
#if YOLO_I8_FUSE_SPPF_ONCHIP
        if (li >= 31 && li <= 33)
            continue;
#endif
        switch (L.op) {
        case OP_COPY:
            op_copy_i8(input, work, L);
            break;
        case OP_CONV:
            op_conv_i8(weights, fixed_params, work, pred, L, Q);
            break;
        case OP_SPLIT:
            op_split_i8(work, L);
            break;
        case OP_MAXPOOL:
#if YOLO_I8_FUSE_SPPF_ONCHIP
            /* All max-pool layers belong to the fused SPPF block above. */
            break;
#else
            op_maxpool_i8(work, L);
            break;
#endif
        default:
            break;
        }
#ifdef I8_DEBUG_DUMP
        if (li <= 20) {
            char fn[128];
            snprintf(fn, sizeof(fn), "c_layer_%03d.bin", li);
            FILE *f = fopen(fn, "wb");
            if (f) {
                fwrite(work, 1, YOLO_I8_WORK_BYTES, f);
                fclose(f);
            }
        }
#endif
    }
}

