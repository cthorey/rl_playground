# Advanced policy gradient

Problems with genuine policy gradient
1. Gradient update are taken on a batch of data whose distribution change everytime the policy is improved. 
2. Not data efficient, have to use data recently collected by the policy. Importance sampling can help us reuse previous batch but very unstable. The variance of the estimator depends on the importance sampling ration which can cause problems.
2. Small change in parameters can translate to big update in policy spaces. What we want is an update rule that guaranty smooth variation in policy space.

## Relative performance of two policies

**Relative performance policy identity**: Expectation over new policy of the discounted advantage collected with the old policies.

Need to incorporate that into our objective.

**KL Divergence** Measure of how far apart two probability distribution are.

We end up with an optimization problem under constraint. Namely, optimizing the policy parameters under the constraint that that the KL divergence between the new and old one is less than a threshold. How do we do that ??

--> natural gradient **NPG**. Require the computation of the Hessian though. 

To save us from computing the inverse O(n3), we can use a few step of conjuguate gradient iteration to get a good estimate. Called **TNPG** Truncated Natural Policy Gradient. Another method --> **ACKTR** Actor critic with kroneker factor trust region. 


Small problems with natural policy gradient update
1. Not be robust to truss region size delta..
2. Because of quadratic approximation of the KL, KL-divergence constraint might be violated.

Here comes **TRPO**. Stop and check that the policy update satisfies:
1. Require improvement in surrogate
2. Enforce KL constraint.

Algo:
1. Compute the step 
2. Then for i=0 L :
    1. Compute proposed update.
    2. If one of the constraint is not respected, halve the step and go back to step1
    3. If both contraint are respected - break.

**PPO Proximal policy optimization**. Don't use natural gradient. Approx. enforce the KL constraint without computing the natural gradient. Two way to do that
1. Adaptive KL penalty
2. Clipped objective.

