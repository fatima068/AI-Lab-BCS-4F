from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = DiscreteBayesianNetwork([
    ('Disease', 'Fever'),
    ('Disease', 'Cough'),
    ('Disease', 'Fatigue'),
    ('Disease', 'Chills')
])
cpd_disease = TabularCPD(
    variable='Disease',
    variable_card=2,
    values=[[0.3], [0.7]]  

  
cpd_fever = TabularCPD(
    variable='Fever',
    variable_card=2,
    values=[
        [0.1, 0.5],  # No
        [0.9, 0.5]   # Yes
    ],
    evidence=['Disease'],
    evidence_card=[2]
)

cpd_cough = TabularCPD(
    variable='Cough',
    variable_card=2,
    values=[
        [0.2, 0.4],
        [0.8, 0.6]
    ],
    evidence=['Disease'],
    evidence_card=[2]
)

cpd_fatigue = TabularCPD(
    variable='Fatigue',
    variable_card=2,
    values=[
        [0.3, 0.7],
        [0.7, 0.3]
    ],
    evidence=['Disease'],
    evidence_card=[2]
)

cpd_chills = TabularCPD(
    variable='Chills',
    variable_card=2,
    values=[
        [0.4, 0.6],
        [0.6, 0.4]
    ],
    evidence=['Disease'],
    evidence_card=[2]
)


model.add_cpds(cpd_disease, cpd_fever, cpd_cough, cpd_fatigue, cpd_chills)
assert model.check_model()
infer = VariableElimination(model)

# TASK 1
q1 = infer.query(
    variables=['Disease'],
    evidence={'Fever': 1, 'Cough': 1}
)

print("TASK 1: P(Disease | Fever, Cough)")
print(q1)

# TASK 2
q2 = infer.query(
    variables=['Disease'],
    evidence={'Fever': 1, 'Cough': 1, 'Chills': 1}
)

print("\nTASK 2: P(Disease | Fever, Cough, Chills)")
print(q2)

# TASK 3
q3 = infer.query(
    variables=['Fatigue'],
    evidence={'Disease': 0}  # Flu
)

print("\nTASK 3: P(Fatigue | Flu)")
print(q3)
