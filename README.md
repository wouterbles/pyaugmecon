# pyaugmecon
This is a <b>python/pyomo</b> implementation of the augmented epsilon-constraint (AUGMECON) method which can be used to solve multi-objective optimization problems, i.e., to return an approximation of the exact Pareto front. <br>
A [GAMS implementation](https://www.gams.com/latest/gamslib_ml/libhtml/gamslib_epscm.html) of the method was provided by the Author of the original paper. To the best of my knowledge, this is the first publicly available python implementation of the method.

<h4>Testing:</h4> 
To test the correctness of the proposed approach, the following optimization models have been tested both using the GAMS and the proposed implementations: <br>

- Sample bi-objective problem of the original paper

- Sample three-objective problem of the original implementation/presentation

- Multi-objective economic dispatch (minimize renewables and cost)

- Multi-objective short-term wind producer problem (maximize expected profit, maximize CVaR)

- Simple residential EMS (maximize profit, maximize self-consumption)

These models can be found in the <b>tests</b> folder.

> Despite the extensive testing, the pyAUGMECON implementation is provided without any warranty. Use it at your own risk!

<h4>Useful resources:</h4>

- The original paper is: <br>
G. Mavrotas, "Effective implementation of the ε-constraint method in Multi-Objective Mathematical Programming problems", <i>Applied Mathematics and Computation</i>, vol. 213, pp. 455-465, July 2009.

- The following PhD thesis is also very useful: <br>
https://www.chemeng.ntua.gr/gm/gmsite_gre/index_files/PHD_mavrotas_text.pdf <br>

- This presentation provides a quick overview: <br>
https://www.chemeng.ntua.gr/gm/gmsite_eng/index_files/mavrotas_MCDA64_2006.pdf

<h3>Modeling philosophy</h3>
The only piece of code that should be modified is the optimization model generator function (optModelGenerator()). This function should return and unsolved instance of the optimization problem. Essentially, the only difference in comparison with creating a single objective Pyomo optimization model is the fact that multiple objectives are defined using Pyomo's ObjectiveList().<br>
The rest of the process is automatically being taken care of by instantiating a MOOP() object. 

```python
a = MOOP(optModelGenerator())
print(a.payOffTable) # this prints the payOffTable
print(a.paretoSols) # this prints the unique Pareto optimal solutions
```

<h4>Notes:</h4>
- The choice of the objective function(s) to add as constraints affects the mapping of the Pareto front. Empirically, having one of the objective functions with the large range in the constraint set tends to result in a denser representation of the Pareto front. Nevertheless, despite the choice of which objectives to add in the constraint set, the solutions belong to the same Pareto front (obviously).
- If X grid points in GAMS, X+1 grid points are needed in pyAUGMECON in order to exactly replicate results