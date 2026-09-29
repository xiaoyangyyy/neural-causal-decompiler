# Exact two-stage finite realization on the trained affine 128D network

The frozen trained affine 128D ReLU ring previously had a
27,216-state weighted coordinate-grid upper certificate at exact
rational tolerance `17/100`. That grid made all full-cube cells
available as initial abstract states. An initial state may occupy any
one of them, but **after one transition** the true and abstract
successors lie inside the network's proved full-cube transition image.
The new construction separates these roles.

The machine has two tagged state kinds:

1. **Initial:** a full-cube grid with three bins on coordinates
   `(0,1,2,3,127)` and one bin on every other coordinate. Its
   `3^5=243` states cover every initial state. The output at a state
   is the frozen network's exact observation of that cell's center.
2. **Recurrent:** the existing, independently certified weighted grid
   with bins `(16,6,6,6,8)` on those five coordinates and 128
   implicit bins per continuous action coordinate. Only indices whose cells intersect the certified coordinatewise
   transition-image enclosure are retained.
   The nontrivial index counts are `(14,4,4,4,5)`, giving
   `14*4^3*5=4,480` recurrent states.

The recurrent certificate proves a full-cube invariant and a
coordinate relation with radii `r`. For an initial grid cell, let
`h_init` be its halfwidth vector; let `h_rec` be recurrent grid
halfwidths, `S` the exact absolute-weight transition sensitivity,
and `U/(2m)` the exact action-binning error at `m=128`. The
independent checker replays the frozen weights and verifies

    output_sensitivity(h_init) <= 17/100,
    S*h_init + U/(2m) + h_rec <= r

coordinate by coordinate using fractions. The smallest handoff slack
is about `0.000476066893`, strictly positive. Therefore every
concrete state represented by an initial cell moves, under **every
continuous action**, into the relation of the single recurrent cell
chosen by the deterministic abstract transition.

The recurrent phase then uses the preexisting exact weighted
simulation invariant for every future action and time. Its state
space is closed: each recurrent center and every quantized action
are inside the unit cubes, so their frozen-network transition lies
inside the exact full-cube transition-image enclosure. Quantizing
that image always gives one of the retained recurrent indices.
The same reasoning places every initial-to-recurrent successor
inside this set.

`ncd.reachable_two_stage` provides executable initial selection,
output, and action-labelled transition functions. The abstract
state includes an `initial` or `recurrent` tag. All
`243+4480=4,723` states are represented implicitly; the
`128^5` action cells are never enumerated. Exact rational replay
checks the serialized model, historical recurrent certificate,
initial output bounds, handoff inequalities, transition-image
indices, and total state count. Finite action-word smoke traces
exercise the actual machine functions, while the proof covers
all real unit-domain actions and arbitrary horizons.

The 4,723-state upper certificate uses `epsilon=17/100`, slightly
stricter than the binary64 `0.17` of the existing 162-state
all-horizon packing lower. Thus the same frozen network's general
minimum-state interval is **[162,4723]**. This is an executable
deterministic finite realization, not merely a behavioral
trajectory cover. The interval remains wide; global minimality
and trained nonlinear 128D realizations are not established.
