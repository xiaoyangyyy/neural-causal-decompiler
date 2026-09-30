# Strict Student5 second-moment tails in five historical SCM worlds

The accepted population-SCM proof established finite ideal second moments for 295 of the 300 frozen confirmation worlds. Its sufficient growth-exponent test left five Student5 worlds unresolved. This extension proves that each of those five has an **infinite observational second moment at a specified output node**. The two packages together classify all 300 historical observational worlds: 295 finite and five infinite. The earlier acceptance artifact is unchanged.

The proof interprets every JSON coefficient as its exact stored binary64 rational and executes the structural equations with mathematical real arithmetic. It applies to the ideal generator in which each node receives an independent continuous Student-t noise with five degrees of freedom and scale sigma = the stored value 0.35, normalized by kappa = sqrt(5/3). It does not describe a finite pseudorandom sample or floating hardware execution. All five worlds have observed scale one and no root-noise shift.

Fix one exogenous root noise U_k=t, and restrict every other exogenous noise to [-1,1]. A polynomial interval is a list of rational intervals [a_d,b_d] with the semantics

    X_j(t) = sum_d c_d(t) t^d,  a_d <= c_d(t) <= b_d,

for every admissible other-noise tuple and every real t. The coefficients are allowed to depend on t, which makes the representation valid for sine and Tanh terms: each is bounded by [-1,1] and contributes only a bounded degree-zero coefficient. Exact interval addition and convolution propagate linear, square and interaction terms in DAG order. The verifier checks the graph, each term's parent set, the fixed noise family and scale, and the observed scale.

For each selected output, the top interval at degree d >= 3 is strictly positive or strictly negative. Let m>0 be its minimum absolute magnitude and B the sum of all lower-degree coefficient absolute upper bounds. The exact integer T=max(1,ceil(2B/m)) then gives, for every t>=T on the other-noise event,

    |X_j(t)| >= m*t^d - B*t^(d-1) >= (m/2)*t^d.

The scaled Student5 density is

    f(u) = 8*kappa/(3*sigma*sqrt(5)*pi)
           * (1+kappa^2*u^2/(5*sigma^2))^(-3).

The checked range 1/4<=sigma<=1/2 and elementary inequalities 1<=kappa<=2, sqrt(5)<3 and pi<4 imply a rational minorant. Write alpha=(4/9)*(5/69)^3. Then f(u)>=alpha for |u|<=1 and f(t)>=alpha*t^(-6) for t>=1. Independence gives probability at least p=(2*alpha)^(n-1) for the other-noise event. Consequently, for every M>=T,

    E[X_j^2] >= p*alpha*(m/2)^2
                * (M^(2d-5)-T^(2d-5))/(2d-5).

Since 2d-5>=1, these explicit finite-cutoff lower bounds grow without limit. Thus the ideal observational second moment is infinite.

The verifier reads five exact unit ZIP archives from the previously accepted 300-world package. It checks each ZIP against the accepted package manifest, then checks each embedded world.json against its original unit manifest. It independently recomputes the rational coefficient intervals, leading separation, cutoff and tail constants; a candidate search or one failed candidate is not a proof. The accepted prior receipt and ledger bind the other 295 finite cases. The portable bundle includes all five archives, their witnesses and the three prior acceptance files, so replay needs no mutable historical run tree.

A direct observational corollary concerns quadratic transport. Each of the six accepted empirical-residual SCMs per affected world has finite support and hence a finite second moment. For any coupling of a divergent true coordinate X and an estimated coordinate Y, |X|^2 <= 2|X-Y|^2 + 2|Y|^2. A finite expected squared transport cost would contradict the proved infinite E[X^2]. Thus the extended quadratic coupling cost is infinite for the 30 historical true/estimated observational pairs. Finite-valued Wasserstein-2 is not asserted because the true law lies outside its usual finite-second-moment domain. This does not imply an infinite Wasserstein-1 cost; all 300 true worlds have a proved finite first moment.

This is a fixed-historical-world population statement. It makes no claim that every compatible intervention preserves infinite variance: clamping the heavy-tailed root or an intermediate node can remove the divergent path. It does not establish graph recovery, neural-program fidelity, noise-law inference from samples, a finite-valued Wasserstein recovery guarantee, or closure of an original R0-R13 atomic claim.