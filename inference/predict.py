from setfit import SetFitModel

model = SetFitModel.from_pretrained("gilleti/emotional-classification")

preds = model(["Ingen tech-dystopi slår människans inre mörker", "Ina Lundström: Jag har två Bruce-tatueringar"])

print(preds)
