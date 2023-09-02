#bayesian network



!pip install pgmpy

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# Step 1: Define the Bayesian Network Structure (DAG)
model = BayesianNetwork([('A', 'C'), ('B', 'C'), ('C', 'D')])

# Step 2: Define Conditional Probability Distributions (CPDs)
cpd_A = TabularCPD(variable='A', variable_card=2, values=[[0.7], [0.3]])
cpd_B = TabularCPD(variable='B', variable_card=2, values=[[0.8], [0.2]])
cpd_C = TabularCPD(variable='C', variable_card=2, values=[[0.9, 0.6, 0.7, 0.1], [0.1, 0.4, 0.3, 0.9]],
                   evidence=['A', 'B'], evidence_card=[2, 2])
cpd_D = TabularCPD(variable='D', variable_card=2, values=[[0.95, 0.2], [0.05, 0.8]],
                   evidence=['C'], evidence_card=[2])

# Step 3: Add CPDs to the model
model.add_cpds(cpd_A, cpd_B, cpd_C, cpd_D)

# Step 4: Verify the model
assert model.check_model()

# Step 5: Perform Inference
inference = VariableElimination(model)

# Calculate P(D=1) given evidence A=0, B=1
result = inference.query(variables=['D'], evidence={'A': 0, 'B': 0})
print(result)