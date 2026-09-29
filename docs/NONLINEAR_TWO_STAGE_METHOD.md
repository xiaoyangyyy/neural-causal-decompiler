# Exact two-stage realization for trained phase-crossing ReLU rings

The earlier 8/32/64/128D trained nonlinear study certified a 2,250-state full-cube weighted grid at exact rational tolerance `17/100` for each of two frozen training seeds. The transition networks genuinely cross ReLU phases in both a state coordinate and a continuous control coordinate. This stage reduces the finite-state upper by separating the transient full-cube initial coverage from the smaller recurrent transition image.

For each of the eight frozen networks, the verifier first checks the previous accepted study hash, model hash, weighted-grid certificate, 81-state initial-output packing lower, and the exact rational state/control second-difference phase witnesses. The weighted certificate proves a unit-cube transition invariant, a global ReLU sensitivity relation, and all-horizon recurrent simulation. The new verifier uses the same frozen network as the transition system; it does not substitute the synthetic teacher.

The initial machine has three bins on coordinates `(0,1,2,3,d-1)` and one on the others, yielding `3^5=243` tagged initial states. Exact absolute-layer-weight sensitivity bounds prove that the output of every initial cell center differs from every state in its cell by at most `17/100`. For every continuous unit-cube action, the sensitivity of a step from a concrete initial cell to the step from its abstract center, including the 128-bin implicit action quantizer, plus recurrent half-bin error, lies inside the previously certified recurrent relation. This is an exact one-step handoff, independent of whether any ReLU phase switches.

An interval image of the *entire* state/action unit cube bounds every possible recurrent successor. Only recurrent grid cells intersecting this image are retained. In each of the eight cases, the five nontrivial active-coordinate index ranges have sizes `(4,3,3,3,2)`, giving `4*3^3*2=216` recurrent states. The proof checks that every initial-to-recurrent and recurrent-to-recurrent transition lands in this rectangle. `two_stage_initial`, `two_stage_output`, and `two_stage_step` are executable deterministic machine functions; the action alphabet remains the full continuous unit square, with internal 128-bin quantization.

Thus every frozen network has a `243+216=459`-state finite realization at epsilon `17/100` for every unit-cube initial state, every continuous unit-cube action word and all finite horizons. An independent 81-state packing lower from the original certificate gives interval **[81,459]**. This is a 79.6% reduction from the prior 2,250-state upper, with the same model, tolerance and full-cube domain.

This theorem covers only the eight frozen structured-feature, output-layer-trained synthetic traffic-inspired ReLU rings. It does not certify arbitrary phase-crossing networks or recover their symbolic causal mechanisms. The 216-cell recurrent rectangle may include unreachable cells, and 459 is not a proved global minimum.

Replay:

```powershell
python -m scripts.acceptance_nonlinear_two_stage --verify
```

Evidence: `validation/nonlinear_two_stage_acceptance.json`, eight certificates under `runs/nonlinear_two_stage_v1`, and the original frozen systems and weighted certificates under `runs/trained_nonlinear_global_v1`.
