#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "yolo_layers_i8.h"
#include "yolo_fixed_params_i8.h"

void yolo8n_hls_i8(const uint8_t *, const int8_t *, const uint8_t *,
                   uint8_t *, int16_t *);

#if YOLO_I8_WEIGHT_BYTES != 3238768u || YOLO_I8_FIXED_PARAM_BYTES != 208676u
#error "YOLO26 Conv1x1 parameters differ from the verified blob snapshot"
#endif

static bool read_exact(const char *path, void *dst, size_t n)
{
    FILE *f = fopen(path, "rb");
    if (!f) { printf("FAIL cannot open %s\n", path); return false; }
    const size_t got = fread(dst, 1, n, f);
    const bool no_extra = fgetc(f) == EOF;
    fclose(f);
    if (got != n || !no_extra) {
        printf("FAIL blob size %s: got %zu, expected %zu\n", path, got, n);
        return false;
    }
    return true;
}

int main(int argc, char **argv)
{
    const char *base = argc > 1 ? argv[1] : "../data/";
    char path[1024];
    std::vector<uint8_t> input(YOLO_I8_INPUT_BYTES), fixed(YOLO_I8_FIXED_PARAM_BYTES);
    std::vector<int8_t> weights(YOLO_I8_WEIGHT_BYTES);
    std::vector<uint8_t> work(YOLO_I8_WORK_BYTES, 0);
    std::vector<int16_t> pred(YOLO_I8_PRED_ELEMS, 0);
    std::vector<float> ref(YOLO_I8_PRED_ELEMS);
    for (int li = 0; li < YOLO_I8_NUM_LAYERS; li++) {
        const int op = yolo_layers_i8[li].op;
        if (op != OP_COPY && op != OP_CONV && op != OP_SPLIT && op != OP_MAXPOOL) {
            printf("FAIL unsupported op %d at layer %d\n", op, li);
            return 2;
        }
        if (op == OP_MAXPOOL &&
            (yolo_layers_i8[li].iw != 7 || yolo_layers_i8[li].ih != 7 ||
             yolo_layers_i8[li].ow != 7 || yolo_layers_i8[li].oh != 7)) {
            printf("FAIL unexpected maxpool geometry at layer %d\n", li);
            return 2;
        }
    }
    snprintf(path, sizeof(path), "%sverify_input_i8.bin", base);
    if (!read_exact(path, input.data(), input.size())) return 2;
    snprintf(path, sizeof(path), "%sverify_pred_i8.bin", base);
    if (!read_exact(path, ref.data(), ref.size() * sizeof(float))) return 2;
    snprintf(path, sizeof(path), "%syolo26n_weights_i8.bin", base);
    if (!read_exact(path, weights.data(), weights.size())) return 2;
    snprintf(path, sizeof(path), "%syolo_fixed_params_i8.bin", base);
    if (!read_exact(path, fixed.data(), fixed.size())) return 2;

    printf("table: %u layers, %u weight bytes, %u fixed bytes, %u work bytes\n",
           (unsigned)YOLO_I8_NUM_LAYERS, (unsigned)YOLO_I8_WEIGHT_BYTES,
           (unsigned)YOLO_I8_FIXED_PARAM_BYTES, (unsigned)YOLO_I8_WORK_BYTES);
    yolo8n_hls_i8(input.data(), weights.data(), fixed.data(), work.data(), pred.data());
    const float q = 1.0f / 1024.0f;
    const int n = YOLO_I8_ANCHORS;
    const int box = YOLO_I8_BOX_CHANNELS;
    int active = 0;
    float worst = 0.0f;
    for (int a = 0; a < n; a++) {
        const int ci = box * n + a;
        if (ref[ci] <= -1.0986123f) continue;
        active++;
        const float ce = fabsf(pred[ci] * q - ref[ci]);
        if (ce > worst) worst = ce;
        for (int c = 0; c < box; c++) {
            const int i = c * n + a;
            const float e = fabsf(pred[i] * q - ref[i]);
            if (e > worst) worst = e;
        }
    }
    printf("active anchors = %d, active max abs err = %.6e (bar 0.50)\n",
           active, (double)worst);
    if (argc > 2) {
        FILE *f = fopen(argv[2], "wb");
        if (!f || fwrite(pred.data(), sizeof(int16_t), pred.size(), f) != pred.size()) {
            printf("FAIL writing prediction dump\n");
            if (f) fclose(f);
            return 2;
        }
        fclose(f);
    }
    if (argc > 3) {
        FILE *f = fopen(argv[3], "wb");
        if (!f || fwrite(work.data(), 1, work.size(), f) != work.size()) {
            printf("FAIL writing work dump\n");
            if (f) fclose(f);
            return 2;
        }
        fclose(f);
    }
    const bool pass = active > 0 && worst < 0.50f;
    printf(pass ? "PASS\n" : "FAIL\n");
    return pass ? 0 : 1;
}
