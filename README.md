# Conditional Feature Importance revisited: Double Robustness, Efficiency and Inference

**Conditional Feature Importance (CFI)** (Strobl et al., 2008) was introduced to account for dependencies between a feature of interest and the remaining inputs. Despite its popularity, CFI has received little theoretical analysis, largely because the conditional sampling step has been treated as a purely practical issue.

In this work, we provide a theoretical foundation for **Conditional Permutation Importance (CPI)**, showing that it is a valid implementation of CFI. Under the conditional null hypothesis, we establish a **double robustness** property: as long as *either* the predictive model *or* the conditional sampler is valid, null features are correctly identified.

Under alternatives, we characterize the population target of CPI and connect it to the **Total Sobol Index (TSI)**. Building on this insight, we introduce **Sobol-CPI**, a generalization of CPI/CFI, prove its nonparametric efficiency, and propose a bias correction. Finally, we develop practical and theoretically justified **type-I error tests** and illustrate the results with numerical experiments.
