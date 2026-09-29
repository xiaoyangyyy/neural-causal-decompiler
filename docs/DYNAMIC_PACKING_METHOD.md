# Exact one-step behavioral packing for finite neural realizations

A deterministic finite-state realization with observation error at most
`epsilon` cannot assign two concrete initial states to one abstract state
when their output traces, under the *same* admissible action word, differ
by more than `2*epsilon` at any time. The abstract transition is then the
same for both, so its output at that time cannot approximate both concrete
outputs. This elementary packing lemma applies to arbitrary frozen ReLU
networks; it does not require global affine dynamics, a chosen encoder,
or a particular abstract transition table.

`ncd.dynamic_packing` stores rational initial states as sparse coordinates
and a shared rational action word of up to 256 steps at any positive
declared binary64 tolerance. The independent verifier evaluates every
stored state and all action-prefix outputs with exact fractions. If it can
prove every hidden ReLU has fixed phase on the full unit input cube, it
extracts an exact affine evaluator; otherwise it directly evaluates the
ReLU network at the finite witness points. It checks every pair and records
the first separating time and exact minimum trace distance. A missing separation, model mismatch, or inconsistent metric or claim fails
replay. Changed points and actions must pass a fresh exact pairwise check.
A separate nonlinear regression case exercises the direct path.

For the frozen affine ring family, the witness uses five pairs
`(x_0,x_(d-1))`:

    (0,0), (1/2,0), (1,0), (1/10,1), (3/5,1).

Coordinates 1, 2 and 3 independently take values `0,1/2,1`; every other
coordinate is zero. This gives `5*3^3=135` initial states. The common
action word is one all-zero action vector. Across different triples,
initial outputs differ by at least `1/2`. Within one triple, initial
coordinate 0 separates every pair except `(0,0)` versus `(1/10,1)` and
`(1/2,0)` versus `(3/5,1)`. The frozen transition has effective
`A[0,0]=binary64(0.45)` and `A[0,d-1]=binary64(0.3)`; those critical pairs
separate in output 0 after one step by approximately `0.345`, strictly
above `2*binary64(0.17)`, approximately `0.34`.

The verifier does not infer separation from this hand calculation. It
replays all 9,045 point pairs from the serialized neural weights: 8,991
separate at time 0, and 54 first separate at time 1. The exact minimum
trace distance is

    62149674857712843 / 180143985094819840,

while the exact threshold is

    6124895493223875 / 18014398509481984.

Their positive difference is
`900719925474093 / 180143985094819840`. The same proof replays for the
8, 32, 64 and 128 dimensional frozen affine profiles. It raises the
128D lower bound from 81 to 135. The historical 40,824-state upper
certificate is independently replayed on exactly the same 128D model.
Its tolerance `17/100` is slightly smaller than the binary64 value used
for the new lower proof, so the upper remains valid. The resulting
certified interval is `[135,40824]`.

Counterfactual checks are deliberately narrower. Removing effective
feedback edge `A[0,d-1]` makes the critical pair's one-step output-0
distance about `0.045`; two trained nonlinear 128D models also leave
that particular pair below the threshold. These exact evaluations prove
that the *specified 135-point witness* fails there. They do not prove a
smaller optimal packing or a smaller finite realization for those models.
