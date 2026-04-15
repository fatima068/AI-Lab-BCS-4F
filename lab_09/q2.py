from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = DiscreteBayesianNetwork([
    ('Intelligence', 'Grade'),
    ('StudyHours', 'Grade'),
    ('Difficulty', 'Grade'),
    ('Grade', 'Pass')
])
cpd_intelligence = TabularCPD('Intelligence', 2, [[0.7], [0.3]])
cpd_studyHours = TabularCPD('StudyHours', 2, [[0.6], [0.4]])
cpd_difficulty = TabularCPD('Difficulty', 2, [[0.4], [0.6]])

cpd_grade = TabularCPD(
    'Grade', 3,
    [
        [0.8,0.6,0.5,0.3,0.4,0.2,0.2,0.1],
        [0.15,0.3,0.3,0.4,0.4,0.5,0.4,0.3],
        [0.05,0.1,0.2,0.3,0.2,0.3,0.4,0.6]
    ],
    evidence=['Intelligence','StudyHours','Difficulty'],
    evidence_card=[2,2,2]
)

cpd_pass = TabularCPD(
    'Pass', 2,
    [
        [0.05, 0.20, 0.50],  # No
        [0.95, 0.80, 0.50]   # Yes
    ],
    evidence=['Grade'],
    evidence_card=[3]
)
model.add_cpds(cpd_intelligence, cpd_studyHours, cpd_difficulty, cpd_grade, cpd_pass)
assert model.check_model()
infer = VariableElimination(model)

#query1: P(Pass | StudyHours=Sufficient(0), Difficulty=Hard(0))
q1 = infer.query(
    variables=['Pass'],
    evidence={'StudyHours':0, 'Difficulty':0}  # Sufficient=0, Hard=0
)

print("P(Pass | Sufficient, Hard):")
print(q1)

#query2: P(Intelligence=High(0) | Pass=Yes(1))
q2 = infer.query(
    variables=['Intelligence'],
    evidence={'Pass':1}  # Yes
)

print("\nP(Intelligence | Pass=Yes):")
print(q2)
