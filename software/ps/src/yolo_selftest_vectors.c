/* Optional boot self-test vectors are omitted from the public source package.
 * The original input was derived from an ultrasound image. The delivered
 * application builds with YOLO_BOOT_SELFTEST=0; supply an authorized vector
 * pair before enabling this diagnostic mode.
 */
#ifndef YOLO_BOOT_SELFTEST
#define YOLO_BOOT_SELFTEST 0
#endif
#if YOLO_BOOT_SELFTEST
#error "Provide an authorized yolo_selftest_vectors.c before enabling the boot self-test"
#endif
