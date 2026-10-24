---
name: fpga-model-deployment
description: Design, debug, and verify neural-network inference on resource-constrained FPGA or Zynq systems when model operators, quantization, PS/PL data movement, timing, and board behavior must be reconciled.
---

# FPGA model deployment

Turn a trained inference graph into a measured FPGA implementation. Start from the user's network, target part, toolchain, PS/PL interfaces, task metric, and available evidence. Derive tile sizes, clocks, thresholds, and acceptance limits from those inputs.

## Working method

1. **Define the contract.** Record the exact model graph, input transformation, tensor layout, output decoding, quantization format, validation split, and intended timing window. Separate task accuracy from fixed-threshold operating results. Use [dataset_contract.md](references/dataset_contract.md), [quantization_contract.md](references/quantization_contract.md), and [tensor_contract.md](references/tensor_contract.md) when these contracts are changing or ambiguous.
2. **Map the graph to the machine.** Inventory unsupported operations and determine how each changes dependencies, memory traffic, and control flow. Compare a PL extension, PS execution, and graph restructuring using the same task and system boundary. Use [operator_mapping.md](references/operator_mapping.md) for the candidate matrix.
3. **Design data movement before scaling arithmetic.** Estimate tensor lifetime, external bytes, weight reuse, line-buffer footprint, bank conflicts, and the number of channel tiles. Choose an architecture whose compute parallelism can be supplied by its memory path. Use [pl_dataflow.md](references/pl_dataflow.md) to prepare the memory map and loop schedule.
4. **Validate at successive boundaries.** Compare a floating-point reference, integer reference, HLS buffers, routed design, and board output at matching semantic points. When a new operator or branch is added, check that the testbench actually exercises it and can detect a deliberate error; use [verification_coverage.md](references/verification_coverage.md). If the first board mismatch appears after C simulation succeeds, use [board_mismatch.md](references/board_mismatch.md) to narrow the cache, address, clock, or control boundary.
5. **Isolate the optimization mechanism.** State the predicted bottleneck and the observation that would disprove it. Keep independent changes separate; record coupled source and parameter changes as one candidate. Measure buffer equality, HLS schedule, routed resource/timing, and board latency at the boundaries relevant to that candidate. Use [optimization.md](references/optimization.md) to decide whether a local improvement has system value.
6. **Freeze the release.** Link model, quantization files, layer table, IP, bitstream, software, reports, and test vectors to one build identity. Use [release_identity.md](references/release_identity.md) when handing the design to another developer or preparing a submission.

## Evidence to return

- For a contract change, describe the input and output semantics, validation unit, quantization convention, and measurement window.
- For an architecture choice, compare the alternatives, expected resource and communication cost, task-metric effect, and the experiment used to choose among them.
- For a mismatch, name the compared buffers or reports, their build identity, the first observed difference, and the next discriminating check.
- For a release, link the model and generated hardware/software artifacts to one build identity.

Return only the evidence relevant to the request. Keep the general method in the skill; store project-specific chronology, fixed constants, and measured effects with that project's evidence.
