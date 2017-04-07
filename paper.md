---
title: "Implementation of Conditionally Accelerated Matrix Math"
author:
  - |
    | Joseph S. Haddad
    | The University of Akron
    | 302 E Buchtel Ave
    | Akron, OH, 44325, United States
    | jsh77@zips.uakron.edu
tags: [Parallel, Processing, OpenMP, CUDA, Network, Petascale, Machine, Learning, Genome]
abstract: |
  Still need an abstract.
references:
  - id: sourcecode
    title: Bayesian Learning source code
    type: webpage
    URL: "https://github.com/Timer/bayesian-learning"
  - id: firstpaper
    title: Analysis of Parallel Bayesian Network Learning
    publisher: "Proceedings of the 31st International Conference on Computers and Their Applications"
    issued:
      year: 2016
      isbn: "978–1–943436–02–6"
    page: 101-106
    author:
      - given: Joseph S.
        family: Haddad
      - given: Anthony
        family: Deeter
      - given: Zhong-Hui
        family: Duan
      - given: Timothy W.
        family: O’Neil
    type: article
  - id: cudainfo
    type: webpage
    title: "CUDA Parallel Computing Platform"
    URL: "http://www.nvidia.com/object/cuda_home_new.html"
  - id: cudaguide
    type: webpage
    title: "CUDA C Programming Guide"
    URL: "http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html"
  - id: cooper
    title: A Bayesian method for the induction of probabilistic networks from data
    publisher: Machine Learning
    issued:
      year: 1992
    volume: 9
    issue: 4
    page: 309-347
    author:
      - given: Gregory F.
        family: Cooper
      - given: Edward
        family: Herskovits
    type: article
  - id: altekar
    title: Parallel metropolis coupled Markov chain Monte Carlo for Bayesian phylogenetic inference
    publisher: Bioinformatics
    issued:
      year: 2004
    volume: 20
    issue: 3
    page: 407-415
    type: article
    author:
      - given: Gautam
        family: Altekar
      - given: Sandhya
        family: Dwarkadas
      - given: John P.
        family: Huelsenbeck
      - given: Fredrik
        family: Ronquist
  - id: misra
    title: "Parallel Bayesian network structure learning for genome-scale gene networks"
    publisher: International Conference for High Performance Computing, Networking, Storage and Analysis
    issued:
      year: 2014
    volume: SC14
    page: 461-472
    type: article
    author:
      - given: Sanchit
        family: Misra
      - given: Vasimuddin
        family: Md
      - given: Kiran
        family: Pamnany
      - given: Sriram P.
        family: Chockalingam
      - given: Yong
        family: Dong
      - given: Min
        family: Xie
      - given: Maneesha R.
        family: Aluru
      - given: Srinivas
        family: Aluru
  - id: sriram
    title: Predicting Gene Relations Using Bayesian Networks
    issued:
      year: 2011
    publisher: MS thesis, University of Akron
    URL: "https://etd.ohiolink.edu/pg_10?0::NO:10:P10_ETD_SUBID:47568"
    author:
      - given: Aparna
        family: Sriram
    type: thesis
  - id: pearl
    type: book
    title: Probabilistic inference in intelligent systems
    issued:
      year: 1998
    publisher: Morgan Kaufmann Publishers
    author:
      - given: Judea
        family: Pearl
  - id: korb
    type: book
    publisher: Chapman and Hall/CRC
    title: Bayesian artificial intelligence
    issued:
      year: 2003
    author:
      - given: Kevin
        family: Korb
      - given: Ann
        family: Nicholson
---

# Introduction
Inferring relations among genes requires a significant amount of data.
Bayesian networks may be used to correlate this data and extract relationships among the genes @sriram. We do not know what this relationship is, but we do know it has a high likelihood of existing.
These relationships can then be used to make testable hypotheses to determine how gene interactions influence life in organisms or humans. As a result, tests can be performed in the lab with more confidence and a reduced chance of wasting time and resources.

This concept has been applied to smaller data sets and shows promising results @sriram, however remains too slow to be applied to a larger problem.
It is our objective to decrease the runtime required to form a network which may reveal genetic interactions.
Bayesian network learning, however, is inherently slow because it is an NP-hard algorithm @cooper.
Search space reduction algorithms may be utilized to reduce the computational complexity.
K2 is a great example of a search space reduction algorithm, and is our algorithm of choice. However, it introduces a new problem. K2 restricts the parent hierarchy of genes within the network @cooper, and thus introduces bias in the computed relations.
To achieve high confidence in the generated networks, an abundance of Bayesian networks need to be computed using random search space restrictions. These random search space restrictions (or topologies) remove the bias and provide results which can be interpreted at various levels of confidence.

By eliminating one problem and introducing another, consensus networks enable the ability of parallelization by requiring multiple units of work rather than just one faster unit of work.
Other authors describe parallel implementations that can increase the speed of Bayesian network learning @altekar @misra.
However, no libraries exist which compute multiple Bayesian networks concurrently.
This project examines the value of Bayesian network learning within a parallel environment in order to reduce the time needed to generate consensus networks using many topological inputs.
This examination is performed through implementation of the said algorithm, exploring methods available such as OpenMP and MPI.

Results from running experiments with varying number of cores and machines are examined and it is found our parallelization has a positive impact. There are a couple caveats, however, such as the over provisioning of resources which leads to waste and potential introduction of latency from cluster parallelism.
When the resources are appropriate for the problem size, OpenMP and MPI substantially reduce the time to generate a consensus network. The reduction in runtime appears to be linear, more so after accounting for introduced latency and overhead.

This paper is an extension to the initial analysis performed on the algorithm and explains the thought processes behind the implementation. The preceding publication shows why the algorithm needs to be sped up, as an increase in samples causes linear growth of the problem and introduction of additional genes causes exponential growth of the problem @firstpaper.
After reading this paper, the reader should have a sense of why and how the parallelization was reasoned about and implemented to achieve optimal efficiency.

# Background
## Bayesian Networks
Bayesian networks capture qualitative relationships among variables within a directed acyclic graph (or DAG).
Nodes within the DAG represent variables, and edges represent dependencies between the variables @korb @pearl.
Bayesian networks have a search space which grows exponentially when introducing new nodes and not placing restrictions on the structure of the network.
This complication can be overcome by using the K2 algorithm. The K2 algorithm reduces the computational cost of learning by imposing restraints on parent node connections via topological ordering @cooper.
Here, a topology refers to a hierarchical structure of parenthood that the K2 algorithm will utilize to reduce overall computational complexity while scoring data relationships.
Restricting the parent ordering, however, creates an issue of bias, which is inherent within a constraint-based search space reduction @sriram.
Sriram @sriram proposed a solution to this issue by creating a consensus network, or the combination of multiple Bayesian networks derived from several topological inputs.
To eliminate the bias created by these restraints, many randomly generated topologies are used. By increasing the number of topological inputs, the consensus network has a greater chance of reflecting the true nature of the gene interactions with higher levels of confidence.

## CUDA
CUDA is a parallel computing platform and application programming interface (API) developed by NVIDIA @cudainfo.
CUDA allows software developers to utilize CUDA-enabled GPUs for general purpose processing (or GPGPU).
CUDA introduces a concept called kernels, which are extensions of C functions that, when called, are executed in parallel by CUDA threads instead of once like regular C functions @cudaguide.
The primary use case is when work is independent and many things need to be done in parallel (e.g. scaling a vector).
Due to the structure of threads on the GPU, operations such as branches or jumps are permitted but discouraged. This is because threads run in lockstep and when a branch happens, the branches are executed serially. This means threads are suspended and do not continue execution while the opposite branch is being explored. After the branch completes and the instructions converge, all threads resume running @cudaguide. This has many detrimental performance implications.
Knowing this, the GPU is best suited for vector-operations like scaling or other arithmetic which does not branch.
The memory for CUDA also resides on the GPU itself, which means before any kernels are executed memory must be copied to the GPU. Memory must then also be copied back to the host machine for use by the CPU @cudaguide. This adds a delay which may invalidate the benefits of CUDA for smaller workloads.

# Methodology
Testing was performed on the Blue Waters petascale machine at the University of Illinois at Urbana-Champaign. The facility is maintained by Cray and consists of 22,640 Cray XE6 machines and 3,072 XK7 machines, which are CPU-only and GPU-accelerated machines respectively. The XE6 machines consist of two 16 core AMD processors with 64 GBs of RAM. The XK7 machines consist of a single 16 core AMD processor with 32 GBs of RAM and a NVIDIA K20X GPU @bwinfo.

Cray XE6 machines were used to perform all tests utilizing purely synthetic data. OpenMP and MPI were implemented by the Cray Compiler, Cray C version 8.3.10.
The synthetic data is in the form of a gene-by-sample matrix consisting of the presence or absence of each gene within the sample.
This data was generated according to a model we defined.
We then ensured the result of the consensus network(s) matched our model to validate functionality and evaluate a degree of correctness for our algorithm.
Each test was run five times with the mean, standard deviation, and standard error calculated to measure runtime consistency.

The library being used to run the tests is available online @sourcecode. This library was implemented as described in this paper.

# Conclusion
By generating a consensus network out of many Bayesian networks, researchers may screen and infer new gene interactions. This allows researchers to feel more confident about testing hypotheses in the lab, such that their resources and time will not be wasted.

We have concluded that utilizing parallelization through means of OpenMP and MPI substantially reduces the time to generate a consensus network. However, as demonstrated in the graphs above, an increase in resources must be tailored to the problem at hand. Increasing the resources too significantly becomes detrimental, resulting in costly waste; see Table 2.

Future work may involve parallelizing the coalescing of consensus networks in effort to reduce the overhead introduced when increasing cluster parallelism.
Additionally, all matrix operations are currently done on a single-thread. These operations (in some cases) contain thousands of rows and columns being applied to an expensive mathematical function.
These operations are ideal for the GPU as it can perform the arithmetic across several thousand of threads simultaneously.
As such, the motivation for this is that CUDA (or other means of GPGPU acceleration) has the potential to speed the algorithm up by several orders of magnitude.

# Acknowledgments
This research was funded in part by a grant from the Choose Ohio First Bioinformatics scholarship.

The data, statements, and views within this paper are solely the responsibility of the authors.

\section{References}
