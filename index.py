import numpy as np
import skfuzzy as fuzz
from skfuzzy import control

umidade = control.Antecedent(np.arange(0, 101, 1), 'umidade')
temperatura = control.Antecedent(np.arange(0, 51, 1), 'temperatura')
validade = control.Consequent(np.arange(0, 91, 1), 'validade')

umidade['baixa'] = fuzz.trapmf(umidade.universe, [0, 0, 20, 35])
umidade['media'] = fuzz.gaussmf(umidade.universe, 50, 5)
umidade['alta'] = fuzz.trapmf(umidade.universe, [65, 80, 100, 100])

temperatura['fria'] = fuzz.trapmf(temperatura.universe, [0, 0, 10, 20])
temperatura['ambiente'] = fuzz.gaussmf(temperatura.universe, 25, 5)
temperatura['quente'] = fuzz.trapmf(temperatura.universe, [30, 40, 50, 50])

validade['baixa'] = fuzz.trapmf(validade.universe, [0, 0, 10, 35])
validade['media'] = fuzz.gaussmf(validade.universe, 45, 5 )
validade['alta'] = fuzz.trapmf(validade.universe, [55, 80, 90, 90])

rule1 = control.Rule(temperatura['quente'] | umidade['alta'], validade['baixa'])
rule2 = control.Rule(temperatura['quente'] & umidade['baixa'], validade['media'])
rule3 = control.Rule(temperatura['fria'] & umidade['alta'], validade['media'])
rule4 = control.Rule(temperatura['fria'] & umidade['baixa'], validade['alta'])
rule5 = control.Rule(temperatura['ambiente'] | umidade['media'], validade['media'])

validation = control.ControlSystem([rule1, rule2, rule3, rule4, rule5])
validade_simulation = control.ControlSystemSimulation(validation)


inputUmidade = int(input("Umidade: "))
validade_simulation.input['umidade'] = inputUmidade

inputTemperatura = int(input("temperatura: "))
validade_simulation.input['temperatura'] = inputTemperatura

validade_simulation.compute()
dias = validade_simulation.output['validade']

print(f"A validade é de {round(dias)} dias")

temperatura.view(sim=validade_simulation)
umidade.view(sim=validade_simulation)
validade.view(sim=validade_simulation)

