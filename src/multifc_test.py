from multifc import *
import re

config = Config()
dev_dataset = MultiFC(mode="dev")
model = RobertaClassifier(config)
models_address = "./models/"
macro_f1s = []
micro_f1s = []
max_f1 = 0
for model_address in filter(lambda x:re.match("multifc_[0-9]+", x),os.listdir(path=models_address)):
  path = os.path.join(models_address, model_address)
  print(path)
  loaded = torch.load(path)
  model_weights = loaded["best_model_weights"]
  model.load_state_dict(model_weights)
  macro_f1, micro_f1 = multifc_evaluate(model, dev_dataset, config,verbose=False)
  if macro_f1 > max_f1:
    max_f1 =  macro_f1
    torch.save(loaded, os.path.join(models_address, f"multifc_best.pt"))
  micro_f1s.append(micro_f1)
  macro_f1s.append(macro_f1)
micro_f1s = np.array(micro_f1s)
macro_f1s = np.array(macro_f1s)
print("####","Average RESULT", "micro", f"{micro_f1s.mean():.3f}", "macro",f"{ macro_f1s.mean():.3f}")

loaded = torch.load(os.path.join(models_address, f"multifc_best.pt"))
weights = loaded["best_model_weights"]
model.load_state_dict(weights)
macro_f1, micro_f1 = multifc_evaluate(model, dev_dataset, config,verbose=True)

