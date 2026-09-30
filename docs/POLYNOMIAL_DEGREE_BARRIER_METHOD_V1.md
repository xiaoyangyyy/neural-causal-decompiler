# A finite-difference barrier for one frozen neural mechanism

The scoped claim concerns the frozen child checkpoint at runs/active_end_to_end_seed4993/worlds/n3_test_id_0/observational_graph/baseline/mechanism_0.pt (SHA-256 20b62f0bd0f6ba0e36ee678fbd7401e8425fdb8720e28c80b4b588382e671b61). Its ideal real-valued Tanh network is evaluated with the exact stored binary weights. The three-input domain is [-1,1]^3; the permitted pointwise error is S/100, where the checkpoint's training output scale is S=8528137/16777216.

Let f be this network and p any real polynomial of total degree at most six in its three inputs. On the rational line (x0,x1,x2)=(0,t,t), p(0,t,t) has degree at most six. At eight equally spaced values t_j=-37/40+j*11/40, j=0,...,7, its seventh finite difference is exactly zero:

    sum_{j=0}^7 (-1)^(7-j) C(7,j) p(0,t_j,t_j) = 0.

The binomial weights [-1,7,-21,35,-35,21,-7,1] have total absolute weight 128. If |f-p|<=S/100 throughout the box, then the absolute finite difference of f is at most 128*S/100 = 8528137/13107200.

The installed historical neural interval checker encloses each of the eight network outputs using exact rational arithmetic and rigorous Tanh enclosures. The independently replayed resulting seventh difference lies in

    [-159726122847194258199057/151115727451828646838272,
     -1277808982777554065575973/1208925819614629174706176].

This interval is strictly negative. Its absolute lower bound exceeds the permitted error contribution by

    12280679486372207340671901/30223145490365729367654400 > 0.

Consequently, every degree-at-most-six polynomial has normalized uniform error at least 1277808982777554065575973/78658180332266577194909696, about 0.0162451, strictly above the target 0.01. No such polynomial can meet the uniform error target. The proof excludes the whole family, without enumerating coefficients. In particular, a bounded search grammar consisting solely of polynomials of degree at most four cannot succeed for this target under this metric. It does not exclude nonpolynomial or piecewise programs, higher-degree polynomials, other networks or worlds, or a program meeting a weaker metric. It says nothing by itself about recovering the true structural equation, internal mechanism alignment, MDL minimality, uniqueness, or hardware floating-point behavior.

The protocol freezes the checkpoint, domain, points, checker source hashes, resource limits, and original claim count. The poly_degree_barrier checker is installed from a wheel into a separate environment together with the historical NCD 0.59 wheel. Prove and verify are separate Job-supervised processes (two threads, 8 GiB memory, 15-minute stage cap). The verifier independently reloads the bundled checkpoint and recomputes all eight rational enclosures and the separation. A strict acceptance gate checks both stage receipts, manifest, wheel/source identity, three adversarial tests, and the exact rational margin. The portable bundle contains its own checkpoint, protocol, certificate, and manifest.

This is a scoped conditional-mechanism search-family boundary relevant to R10.mechanism and to the polynomial grammar used by the R4 search method. It is not a theorem about the frozen discovery network named in R4.program_fidelity or about internal read/write alignment in R5; it closes no original R0-R13 atomic claim. The original ledger remains 0 proved, 2 refuted, 36 unresolved.