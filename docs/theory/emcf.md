# The principle of EMCF unwrapping

This page explains EMCF (extended minimum cost flow, Pepe & Lanari 2006) unwrapping from the point of view of space x
time. It first writes the ideal problem, in which all points and all images are unwrapped together, then shows that the
time step and the space step of EMCF are two relaxations of it, how the phase closure correction brings the result back
to consistent image phases, and how moraine implements each step. It assumes that you know interferograms, phase
wrapping and minimum cost flow (MCF) unwrapping; how to run it is in [05 Phase unwrapping](../workflows/05_unwrap.md).

## Notation

**Subscripts are space.** $p, q$ are points; $pq$ is the edge from $p$ to $q$ of the Delaunay triangulation of the
points; $\tau$ is a triangle of the triangulation and $\partial\tau$ its three edges; $\sigma_\tau(pq) = +1$ if the
direction of edge $pq$ is the counterclockwise direction around $\tau$, otherwise $-1$.

**Superscripts are time.** $i, j, l$ are images; $ij$ is an image pair, the interferogram "image $i$ − image $j$";
$\theta$ is a loop of the network of image pairs, e.g. $(ij), (jl), (il)$; $\partial\theta$ are its pairs and
$\sigma_\theta(ij) = \pm 1$ is the direction of a pair in the loop (in the example $+1$ for $(ij)$ and $(jl)$, $-1$ for
$(il)$).

**Differences.** For a quantity $X_p^i$ of (point, image), two points as subscript mean the difference along the edge,
two images as superscript the difference of the images:

$$
X_{pq}^i = X_q^i - X_p^i, \qquad
X_p^{ij} = X_p^i - X_p^j, \qquad
X_{pq}^{ij} = X_q^i - X_p^i - X_q^j + X_p^j .
$$

$X_{pq}^{ij}$ is the gradient of interferogram $ij$ along edge $pq$, the **mixed difference**. Quantities of an edge only
(e.g. a cost $c_{pq}$) take the subscript $pq$ as well, quantities of an image pair only (e.g. a cost $c^{ij}$) the
superscript $ij$. $\mathcal{W}(x) = x - 2\pi \operatorname{round}(x / 2\pi)$ wraps a phase into $(-\pi, \pi]$.

**Data and unknowns.** $\psi_p^i \in (-\pi, \pi]$ is the wrapped phase of image $i$ at point $p$ after phase linking,
$N_p^i \in \mathbb{Z}$ are the unknown whole cycles, and the unwrapped phase is

$$
\Phi_p^i = \psi_p^i + 2\pi N_p^i .
$$

By the rule above, $\psi_{pq}^{ij}$ is the mixed difference computed from the wrapped phases (not wrapped again); the
observed gradient of the interferogram is $\mathcal{W}(\psi_{pq}^{ij})$. The whole cycles lost by wrapping,

$$
m_{pq}^{ij} = \frac{\psi_{pq}^{ij} - \mathcal{W}(\psi_{pq}^{ij})}{2\pi} \in \{-2, \dots, 2\} ,
$$

are known integers. After unwrapping, the gradient of interferogram $ij$ along edge $pq$ differs from the observed one by

$$
k_{pq}^{ij} = \frac{\Phi_{pq}^{ij} - \mathcal{W}(\psi_{pq}^{ij})}{2\pi} = m_{pq}^{ij} + N_{pq}^{ij}
$$

whole cycles.

## MCF of one interferogram

Unwrapping interferogram $ij$ alone, the unknowns are its whole cycles $K_p^{ij} \in \mathbb{Z}$ at every point (which
should equal $N_p^{ij}$); the change on an edge is $m_{pq}^{ij} + K_{pq}^{ij}$, and the problem is

$$
\min_{K^{ij}} \sum_{pq} c_{pq} \left| m_{pq}^{ij} + K_{pq}^{ij} \right| ,
$$

with $c_{pq} > 0$ the cost of one cycle on edge $pq$ and the sum over the edges of the triangulation. Every term holds
the difference of two unknowns, $K_q^{ij} - K_p^{ij}$: an L1 potential problem on a graph.

The same problem can be written with the changes $k_{pq}^{ij}$ as unknowns. The changed gradients must close around every
triangle; the $\psi_{pq}^{ij}$ around a triangle add up to 0, so the condition is

$$
\sum_{pq \in \partial\tau} \sigma_\tau(pq)\, k_{pq}^{ij} = -r_\tau^{ij}, \qquad
r_\tau^{ij} = \frac{1}{2\pi} \sum_{pq \in \partial\tau} \sigma_\tau(pq)\, \mathcal{W}(\psi_{pq}^{ij}) ,
$$

where $r_\tau^{ij}$ is the residue of interferogram $ij$ in triangle $\tau$. With every triangle a node (and an earth
node outside the convex hull), $k_{pq}^{ij}$ is the flow across edge $pq$ between the triangles on its two sides and the
condition is the conservation of flow with supplies $r_\tau^{ij}$: a minimum cost flow (Costantini 1998; for sparse
points Costantini & Rosen 1999). On a simply connected triangulation both forms are the same problem: every triangle
closes if and only if $k^{ij} - m^{ij}$ is the difference of an integer field.

The assumption is the Itoh condition: on most edges the true gradient is smaller than $\pi$, i.e. $k_{pq}^{ij} = 0$, and
the cycles jump on few edges of low cost. Every interferogram is unwrapped alone; the redundancy of the network of image
pairs is not used.

## Unwrapping as a problem of space x time

### The data block

The points lie in a plane and the images on the time axis: the cells $(p, i)$ make a block of 2D space x 1D time.
Unwrapping in 3D means choosing one integer $N_p^i$ for every cell of the block.

### Which differences are small

The block has three kinds of differences:

- **Spatial gradient of an image**, $\Phi_{pq}^i$ (one image, two neighbouring points). $\psi^i$ is relative to the
  reference image, and images far from it in time have accumulated large deformation: not necessarily small.
- **Interferometric phase at a point**, $\Phi_p^{ij}$ (one point, two images). An interferogram spans many cycles over a
  scene and has an arbitrary constant: not necessarily small.
- **Mixed difference**, $\Phi_{pq}^{ij}$ (two points x two images, a small square). The gradient of a short
  interferogram along a short edge: small.

```text
           point p           point q
image i   (p,i) ──────────── (q,i)
            │                  │
image j   (p,j) ──────────── (q,j)
```

Another way to see it is the gauge freedom. Replacing $N_p^i$ by $N_p^i + M^i + L_p$ ($M^i \in \mathbb{Z}$ the same
integer added to image $i$ at all points, $L_p \in \mathbb{Z}$ the same integer added to point $p$ in all images) leaves
the gradients of all interferograms unchanged, so the data cannot tell these solutions apart: they stand for the missing
reference point and the missing reference image. $N_{pq}^i$ changes by $L_q - L_p$ and $N_p^{ij}$ by $M^i - M^j$; only
$N_{pq}^{ij}$ does not change. A cost term that depends on the gauge makes the objective depend on an arbitrary choice:
**the costs belong to the mixed differences**.

With the costs on one-dimensional differences such as $\Phi_{pq}^i$ and $\Phi_p^{ij}$ (classical 3D unwrapping of a data
cube) every term is still the difference of two unknowns and network flow solves the problem exactly; but its
assumption, that these differences are small, does not hold for InSAR.

### Ideal 3D unwrapping

$$
\min_{N} \sum_{pq} \sum_{ij} c_{pq}^{ij} \left| m_{pq}^{ij} + N_{pq}^{ij} \right|
$$

$c_{pq}^{ij} > 0$ is the cost of one cycle on the mixed difference $(pq, ij)$; ideally it decreases with the probability
that the true gradient on this edge and pair is outside $(-\pi, \pi]$ (cheap on long edges, at points of low quality, for
long time spans). The sum runs over the edges of the triangulation and the pairs of the network.

The solution gives the whole cycles $N_p^i$ of every image: all interferograms are differences of the same $N$, so every
loop closes and the unwrapped phase of every image comes out directly; the reference point and the reference image fix
the gauge.

### Form with residues

With the changes $k_{pq}^{ij}$ as unknowns there are two families of constraints.

**Closure in space**: for every pair $ij$ and triangle $\tau$,
$\sum_{pq \in \partial\tau} \sigma_\tau(pq)\, k_{pq}^{ij} = -r_\tau^{ij}$, as for one interferogram.

**Closure in time**: for every edge $pq$ and loop $\theta$,

$$
\sum_{ij \in \partial\theta} \sigma_\theta(ij)\, k_{pq}^{ij} = -r_{pq}^\theta, \qquad
r_{pq}^\theta = \frac{1}{2\pi} \sum_{ij \in \partial\theta} \sigma_\theta(ij)\, \mathcal{W}(\psi_{pq}^{ij}) ,
$$

where $r_{pq}^\theta$ is the residue in time of loop $\theta$ on edge $pq$.

If the triangulation is simply connected and $\theta$ runs over the loops of the network, $k$ meets both families if and
only if $k = m + N_{pq}^{ij}$ for an integer field $N$. Proof: by the closure in space, $k^{ij} - m^{ij}$ is the spatial
difference of an integer field $K^{ij}$, zero at the reference point; on every edge $k - m$ closes in time, so its sums
along spatial paths do too, and $K_p^{ij}$ closes in time at every point: $K_p^{ij} = N_p^{ij}$. The form with residues
and the potential form above are the same problem.

The two kinds of residues are not independent. Exchanging the order of the sums for every triangle $\tau$ and loop
$\theta$ gives

$$
\sum_{ij \in \partial\theta} \sigma_\theta(ij)\, r_\tau^{ij} = \sum_{pq \in \partial\tau} \sigma_\tau(pq)\, r_{pq}^\theta .
$$

A residue of one interferogram cannot just end: it continues in the same triangle of another interferogram of the loop,
or as a residue in time on an edge of the triangle.

### Geometric picture

Let the network of image pairs be a planar triangulation (e.g. the Delaunay network of the images in the plane of time and
perpendicular baseline of the original paper). $S^*$ is the dual graph of the triangulation of the points: its nodes are
the triangles $\tau$ and an earth node, every edge $pq$ has a dual edge. $T^*$ is the dual graph of the triangulation of
the images: its nodes are the loops $\theta$ and the outer face, every pair $ij$ has a dual edge. In the product
$S^* \times T^*$:

| element of $S^* \times T^*$ | quantity |
|---|---|
| vertex $(\tau, \theta)$ | conservation of the residues |
| edges $(\tau, ij)$ and $(pq, \theta)$ | residues $r_\tau^{ij}$ and $r_{pq}^\theta$ |
| square $(pq, ij)$ | change $k_{pq}^{ij}$ |

The residues are closed lines (the conservation says that what flows into a vertex flows out), and the two families of
constraints say that the boundary of the surface $k$ is the residue lines. **Ideal 3D unwrapping is the surface of least
cost bounded by the residue lines.** In 2D the residues are points and the cuts are paths, and the minimum cost flow is
the cheapest set of paths; in 3D "points and paths" become "closed lines and surfaces".

### Why it is not a network flow

In 2D every term is the difference of two unknowns. In 3D every term $m_{pq}^{ij} + N_{pq}^{ij}$ holds four unknowns,
$N_q^i, N_p^i, N_q^j, N_p^j$; in the form with residues every $k_{pq}^{ij}$ appears in two constraints in space (the
triangles on both sides of $pq$) and in several in time (the loops through $ij$). The constraint matrix is not a network
matrix: the problem is no minimum cost flow but a general integer program. Already for examples of a few points and
images the matrix of $N \mapsto N_{pq}^{ij}$ is not totally unimodular and the linear programming relaxation has optima
of half-integers, so linear programming does not solve it exactly either. For tens of millions of points, solving the
ideal problem directly is out of reach.

## EMCF: two relaxations

The four unknowns of $N_{pq}^{ij}$ can be grouped as a difference of differences in two ways. Taking one level of
differences as independent unknowns turns the problem back into differences of two unknowns, which MCF solves.

Let $\mathcal{F}_S$ be the set of $k$ that close in space, $\mathcal{F}_T$ the set of $k$ that close in time, and
$J(k) = \sum_{pq} \sum_{ij} c_{pq}^{ij} \left| k_{pq}^{ij} \right|$. The ideal problem is
$\min_{\mathcal{F}_S \cap \mathcal{F}_T} J$; dropping either family of constraints is a relaxation:

$$
\min_{\mathcal{F}_S} J \le \min_{\mathcal{F}_S \cap \mathcal{F}_T} J, \qquad
\min_{\mathcal{F}_T} J \le \min_{\mathcal{F}_S \cap \mathcal{F}_T} J .
$$

### Every interferogram alone: no closure in time

Replacing $N_p^{ij}$ by an independent integer field $K_p^{ij}$ of every pair splits the problem into one MCF per
interferogram, as above (`mcf-pc`). Without loops in the network there is no closure in time, and unwrapping every
interferogram alone is the ideal problem; this is why errors cannot be found without redundancy.

### The time step: no closure in space

Grouped space first, then time:

$$
k_{pq}^{ij} = m_{pq}^{ij} + \underbrace{\left( N_q^i - N_p^i \right)}_{n_{pq}^i}
- \underbrace{\left( N_q^j - N_p^j \right)}_{n_{pq}^j} .
$$

$n_{pq}^i$ are the whole cycles of the gradient of image $i$ along edge $pq$. Taken as independent unknowns of every edge
(by the rule above $n_{pq}^{ij} = n_{pq}^i - n_{pq}^j$), they split the problem into one problem per edge $pq$:

$$
\min_{n_{pq}} \sum_{ij} c^{ij} \left| m_{pq}^{ij} + n_{pq}^{ij} \right| ,
$$

with $c^{ij} > 0$ the cost of changing pair $ij$ by one cycle. It is a potential problem on the graph of image pairs: the
nodes are the images, the edges the pairs. The constraint dropped is $n_{pq}^i = N_q^i - N_p^i$: around a triangle the
$n_{pq}^i$ add up to 0. If the network of the points is a tree (no triangles), the time step is the ideal problem.

Example: on an edge the true gradients of $(ij)$, $(jl)$ and $(il)$ are $0.6\pi$, $0.6\pi$ and $1.2\pi$. Wrapped, $(il)$
becomes $-0.8\pi$ and the residue in time is $r_{pq}^\theta = (0.6\pi + 0.6\pi + 0.8\pi) / 2\pi = 1$. Adding one cycle to
$(il)$ closes the loop, and so does subtracting one from $(ij)$ or $(jl)$: with unit costs the three are equally good. The
other loops decide (with every image paired with the next three, every pair is in several triangles of pairs), or the
costs (long pairs alias more easily and are cheaper to change). Short pairs have small gradients and hardly alias; the
time step uses them to fix the whole cycles of the long pairs on every edge, so that edges where a long pair has a
gradient above $\pi$ get the right value too.

The time step of moraine (`_emcf_temporal` in `moraine/api/unwrap/emcf.py`) writes the problem with the wrapped image
gradients $\mathcal{W}(\psi_{pq}^i)$; its unknowns differ from $n_{pq}^i$ by known integers (the cycles lost in wrapping
$\psi_{pq}^i$), so it is the same problem. The solver (`_l1_fit` in `moraine/api/unwrap/closure.py`) first integrates
integer potentials of the images along a spanning forest of the graph of pairs, so that the pairs of the forest are not
changed. If no pair has to change (every loop on this edge closes already, as on most edges), it stops. Otherwise it
solves the dual problem, a minimum cost circulation on the graph of pairs with capacities $c^{ij}$, by cancelling
negative cycles found by Bellman–Ford; the final shortest path distances are the optimal potentials. If the network of
pairs is a planar triangulation, this is the same problem as the MCF on the dual of that triangulation in the original
paper; moraine needs no triangulation of the pairs and takes any network (decision 0020).

`temporal_cost` chooses the costs: `constant` $c^{ij} = 1$; `length`

$$
c^{ij} = \operatorname{clip}\left( \operatorname{round}\left( 4\, \frac{\overline{\Delta t}}{\Delta t^{ij}} \right),\ 1,\ 20 \right),
$$

with $\Delta t^{ij} = |t^i - t^j|$ the time span of pair $ij$, $t^i$ the acquisition time of image $i$ and
$\overline{\Delta t}$ the median time span of the pairs. Short pairs are expensive, long pairs are changed first.

### The space step: relative to the time step

Grouped time first, then space:

$$
k_{pq}^{ij} = m_{pq}^{ij} + \underbrace{\left( N_q^i - N_q^j \right)}_{K_q^{ij}}
- \underbrace{\left( N_p^i - N_p^j \right)}_{K_p^{ij}} .
$$

With $K_p^{ij}$ independent for every pair this is unwrapping every interferogram alone. The space step of EMCF
(`_emcf_spatial`) does not measure the changes from the wrapped values but from the result $\hat n_{pq}^i$ of the time
step, once for every pair $ij$:

$$
\min_{K^{ij}} \sum_{pq} c_{pq} \left| K_{pq}^{ij} - \hat n_{pq}^{ij} \right| .
$$

In the implementation the residues of the gradients corrected by the time step,
$\mathcal{W}(\psi_{pq}^{ij}) + 2\pi \left( m_{pq}^{ij} + \hat n_{pq}^{ij} \right)$, are
$\hat r_\tau^{ij} = \sum_{pq \in \partial\tau} \sigma_\tau(pq)\, \hat n_{pq}^{ij}$; a minimum cost flow (successive
shortest paths, `moraine/api/unwrap/mcf.py`) removes them and the gradients are integrated over the triangles from the
first point $p_0$, whose value is $\psi_{p_0}^i - \psi_{p_0}^j$ (not wrapped).

`spatial_cost` chooses the costs: $c_{pq} = 1 + \operatorname{round}(99\, \eta_{pq})$ with $\eta_{pq} \in [0, 1]$ the
reliability of the edge. `length`: $\eta_{pq} = \min(1, \bar L / L_{pq})$, $L_{pq}$ the length of the edge and $\bar L$
the median length; `weight`: $\eta_{pq} = \min(w_p, w_q)$, $w_p \in [0, 1]$ the weight of a point (e.g. the temporal
coherence); `length+weight` the product of both; `constant` $c_{pq} = 1$. Reliable edges are expensive, so the cycles
jump on long edges and at points of low quality first; edges to the earth node cost `earth_cost` times more.

If the space step changes nothing ($K_{pq}^{ij} = \hat n_{pq}^{ij}$ on every edge), summing along the same integration
tree gives $K_p^{ij} = A_p^i - A_p^j$, with $A_p^i$ the sum of $\hat n_{pq}^i$ along the path from $p_0$: every loop
closes at every point. Loops can only open in the space step.

### The closure correction: back to consistent image phases

The result of the time step is in $\mathcal{F}_T$ and the space step projects it into $\mathcal{F}_S$. The loops still
close only if the changes of the space step, $f_{pq}^{ij} = K_{pq}^{ij} - \hat n_{pq}^{ij}$, can be written as
$F_{pq}^i - F_{pq}^j$ ($F_{pq}^i$ an integer change of image $i$ alone); unwrapping every interferogram on its own does not
ensure it. In the geometric picture, the time step chooses the surface strip by strip along $T^*$ (one strip per edge
$pq$) and the space step adds squares strip by strip along $S^*$ (one strip per pair); a square $(pq, ij)$ added in the
space step also has boundary on the edges $(pq, \theta)$ in time, and where the interferograms of a loop add different
squares, the closure in time breaks.

The phase closure correction (`unwrap-correct-closure-pc`, decisions 0021 and 0022) brings the result back into
$\mathcal{F}_S \cap \mathcal{F}_T$. At every point it solves

$$
\min_{N_p} \sum_{ij} \left| N_p^{ij} - K_p^{ij} \right| ,
$$

which gives the change $\mu_p^{ij} = N_p^{ij} - K_p^{ij}$ of every interferogram. In every region $R$ (a connected
component of the points without the edges longer than $4 \bar L$) every pair is shifted by the median $\bar\mu_R^{ij}$ of
its $\mu_p^{ij}$, because errors of spatial unwrapping shift whole regions; the points of regions of fewer than 30 points
take their own $\mu_p^{ij}$. A last solution at every point gives the whole cycles $N_p^i$ of the images and the phase
relative to the reference image $i_0$, $\Phi_p^{i i_0} = \Phi_p^i - \Phi_p^{i_0}$. The result meets both families of
constraints, but it is not necessarily the optimum of the ideal problem.

### Summary

| method | unknowns | every term | MCF | constraints met |
|---|---|---|---|---|
| ideal 3D | $N_p^i$ | four unknowns | no | both |
| every interferogram alone (`mcf-pc`) | $K_p^{ij}$, independent per pair | difference of two | yes | closure in space |
| time step | $n_{pq}^i$, independent per edge | difference of two | yes | closure in time |
| space step | $K_p^{ij}$, relative to the time step | difference of two | yes | closure in space |
| closure correction | $N_p^i$, independent per point | difference of two | yes | both, not necessarily optimal |

## Example: a residue of one image

Let image $i$ be noisy in triangle $\tau$, the only image with a residue there, the others smooth. Every pair with $i$
has $r_\tau^{ij} = \pm 1$, and no edge has a residue in time. In $T^*$ the dual edges of the pairs of $i$ make a loop
around image $i$, so the residue line is a closed line at $\tau$ that goes once around image $i$ in $T^*$.

- **Ideal solution**: the surface of least cost is "a path $\gamma$ in $S^*$ from $\tau$" x "this loop": every
  interferogram with $i$ jumps on the same path $\gamma$, and the loops still close.
- **Time step**: the pairs with $i$ do not contradict each other, and the time step changes no gradient.
- **Space step**: every $(ij)$ has a residue in $\tau$ and chooses its own cut $\gamma_j$. Where $\gamma_j \neq \gamma_l$,
  the loop $(ij) + (jl) - (il)$ is one cycle off at the points between the two cuts, which is left to the closure
  correction.

## Assumptions and limits

- The time step assumes that on every edge the cost weighted majority of the pairs has $\left| \Phi_{pq}^{ij} \right| < \pi$,
  the Itoh condition in time. Where most pairs alias (long edges across gaps, steep deformation gradients, strong
  atmosphere in one image), L1 follows the majority and changes the wrong ones.
- Changes of the form $n_{pq}^i - n_{pq}^j$ close every loop. If on an edge the true gradient of image $i$ relative to
  every other image is one cycle more than the wrapped one, $n = 0$ costs nothing and the time step does not see the
  error. If the error is on one edge, it leaves residues in the interferograms with $i$, which the space step handles; if
  it is consistent in space too (e.g. image $i$ one cycle off over a whole region), it is a surface without boundary,
  adding it only adds cost, and the ideal problem does not correct it either. On the synthetic data of the workflow guide
  0.4–9 % of the points without any misclosure were wrong: use the result together with a quality of the points such as
  the temporal coherence.
- Short pairs constrain the long ones: the network needs enough short pairs with small gradients.

## Relation to the original EMCF

- The original paper (Pepe & Lanari 2006) builds the network by a Delaunay triangulation of the images in the plane of
  time and perpendicular baseline; the time step is an MCF on the dual of this triangulation, once per spatial edge, and
  the space step starts from its result with one MCF per interferogram in the azimuth / range plane. moraine takes any
  network of pairs (without the perpendicular baseline; the example pairs every image with the next three) and writes the
  time step as a potential problem; on a planar triangulation both have the same optimum (decision 0020).
- The original paper unwraps multilooked interferograms. With $\varphi_p^{ij}$ the wrapped phase of the multilooked
  interferogram $ij$ at point $p$, the closure phase $\mathcal{W}(\varphi_p^{ij} + \varphi_p^{jl} - \varphi_p^{il})$ is in
  general not 0, and the residues in time include noise. The interferograms of moraine are made of the phase linked
  image phases: their closure phase is exactly 0 (modulo $2\pi$), the residues in time
  $r_{pq}^\theta = -\sum_{ij \in \partial\theta} \sigma_\theta(ij)\, m_{pq}^{ij}$ come from the wrapping of the gradients
  only, and $m_{pq}^{ij}$ is computed directly from the image phases.
- Where loops are still open after the two steps, moraine uses the separate phase closure correction (decision 0021),
  which also takes the results of unwrapping every interferogram alone.

## In moraine

- Commands: `image-pairs` makes the network of pairs, `emcf-pc` runs the time step and the space step,
  `unwrap-correct-closure-pc` corrects by phase closure and gives the phase of every image; `mcf-pc` unwraps every
  interferogram alone, `gamma-mcf-pt` with GAMMA's `mcf_pt`. Their arguments are in the
  [command line reference](../cli/pu.md), how to run them and the expected results in
  [05 Phase unwrapping](../workflows/05_unwrap.md).
- Code: `moraine/api/unwrap/emcf.py` (the two steps), `moraine/api/unwrap/closure.py` (the L1 potential problem on the
  graph of pairs and the closure correction), `moraine/api/unwrap/mcf.py` (minimum cost flow on the triangulation of the
  points).
- Decisions: [0012](../decisions/0012-network-emcf-inversion.md) (redundant networks and EMCF),
  [0013](../decisions/0013-own-delaunay-and-mcf.md) (Delaunay triangulation and successive shortest path MCF),
  [0020](../decisions/0020-emcf-any-image-pair-network.md) (any network of pairs, the units of the two steps, the costs),
  [0021](../decisions/0021-closure-correction-by-regions.md) (closure correction by regions),
  [0022](../decisions/0022-closure-correction-outputs-phase-time-series.md) (the phase of every image as output).

## References

- M. Costantini, "A novel phase unwrapping method based on network programming", IEEE Transactions on Geoscience and
  Remote Sensing 36(3), 1998.
- M. Costantini, P. A. Rosen, "A generalized phase unwrapping approach for sparse data", IGARSS 1999.
- A. Pepe, R. Lanari, "On the extension of the minimum cost flow algorithm for phase unwrapping of multitemporal
  differential SAR interferograms", IEEE Transactions on Geoscience and Remote Sensing 44(9), 2006.
- Z. Yunjun, H. Fattahi, F. Amelung, "Small baseline InSAR time series analysis: unwrapping error correction and noise
  reduction", Computers & Geosciences 133, 104331, 2019.
