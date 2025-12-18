# Active Learning-Based Discovery of Surfactant Molecules

This repository implements an active learning framework integrated with coarse-grained molecular dynamics (CGMD) simulations to discover effective hydrocarbon-based surfactant alternatives to PFAS for firefighting applications. The goal is to identify surfactants that maximize interfacial transport resistance ($R_{int}$) to effectively inhibit fuel transport across oil-water interfaces.

## Project Overview

The search for eco-friendly alternatives to per- and polyfluoroalkyl substances (PFAS) is critical. This project leverages machine learning to navigate a vast chemical space of hydrocarbon surfactants, reducing the computational cost of evaluating candidates through iterative active learning rounds.

### Key Procedures

The workflow is divided into three main stages:

### Step 1: Chemical Space Design
- **Initial Generation:** Defined a chemical space of approximately 2.8 million single-tail hydrocarbon surfactants.
- **Screening & Filtering:** Reduced the space to a manageable size of **12,124 unique CG surfactants** by evaluating:
    - **Stability:** Ensuring the molecules are physically viable.
    - **Synthetic Feasibility:** Using scores like GAscore (geometry-based) and GASAscore (reaction-step based) to assess ease of synthesis.
- **Output:** A finalized set of candidate molecules ready for embedding and simulation.

### Step 2: RAE Embedding (Molecular Representation)
- **Model:** Developed a **Relational Autoencoder (RAE)** to generate compressed molecular graph representations.
- **Architecture:** 
    - Uses message-passing neural networks to process graph-based molecular data (nodes as Martini v3 beads, edges as chemical bonds).
    - Encodes molecules into a **16-dimensional latent space** (z-vectors).
- **Features:** Incorporates bead types, hydrogen bonding capabilities, charges, and interaction parameters (epsilon and sigma).

### Step 3: GPR and Bayesian Optimization (Active Learning)
- **Surrogate Model:** Employs **Gaussian Process Regression (GPR)** to map molecular embeddings (z-vectors) to their interfacial transport resistance ($R_{int}$).
- **Active Learning Loop:**
    - **Sampling:** Starts with an initial sampling (Round 0).
    - **Optimization:** Uses **Bayesian Optimization (BO)** to select the most informative next candidates to simulate.
    - **Strategies:** Compares constant sampling and adaptive sampling strategies across 4 rounds of active learning.
- **Goal:** Identify the best-performing surfactants while simulating only a small fraction (2.5%) of the total chemical space.

## Repository Structure

```text
.
├── data/                         # Datasets for training and evaluation
│   ├── features.csv              # Chemical features for the 12,124 surfactants
│   ├── Rint.csv                  # Calculated interfacial transport resistance values
│   └── Rint_with_features.csv    # Combined dataset of features and Rint results
├── Step1_chemical_space_design/  # Initial generation and filtering logic
├── Step2_RAE_embedding/          # Relational Autoencoder for molecular embeddings
├── Step3_GPR_Bayesian/           # Active learning rounds (GPR + Bayesian Optimization)
└── Surfactant_Discovery.pdf      # Detailed manuscript of the research
```

## Results

- Identified surfactants with an $R_{int}$ of up to **1641.5 s/m**, which is significantly higher than commercial alternatives like DTAB.
- Demonstrated that strategic manipulation of the surfactant head group is the primary driver for maximizing transport resistance.

## Citation

If you use this code or the findings from the manuscript in your research, please cite:

Wang, X., Zhang, H., Yu, X., Ham, S., Lattimer, B., & Qiao, R. (2025). *Active Learning-Based Discovery of Surfactant Molecules as Alkane Transport Inhibitors*.
