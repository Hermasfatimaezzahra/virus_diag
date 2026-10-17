# virus_diag -Projet POO adapté à l'IA 

virus_diag/
├── data/       patient_data.csv
├── core/       dataset.py, model.py, logistic_regression.py, neural_network.py, optimizer.py
├── pipeline/   trainer.py, evaluator.py, predictor.py   <- ClinicalPredictor (exercice p.147)
├── utils/      preprocessing.py, metrics.py
├── app/        api.py, interface_clinique.py
└── main.py


Lancer : `pip install numpy`, puis `python generate_data.py` (une fois) et `python main.py`
(option : `python main.py --interactive` pour l'interface médecin) .

## Réponses
2. **Polymorphisme** appuye par l'abstraction `DiagnosticModel` : ClinicalPredictor appelle predict()
   sans connaître le type exact du modèle.
3. **Séparer Predictor et Trainer**  responsabilité unique ; sécurité clinique :le médecin ne peut pas
   modifier ni re-entraîner le modèle , modularite(changer d'algorithme sans casser l'application)et
   testabilité et maintenance