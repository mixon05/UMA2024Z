import json
import numpy as np

LEARNING = "active"
IR = 256
with open(f"one_positive_class/rbf_active_learning_IR{IR}.json", "r", encoding="utf-8") as json_file:
    active = json.load(json_file)
    
with open(f"one_positive_class/rbf_passive_learning_IR{IR}.json", "r", encoding="utf-8") as json_file:
    passive = json.load(json_file)
    
    
if LEARNING == "active":
    true_labels = active["true labels"]
    active_models = active["models"]    
elif LEARNING == "passive":
    true_labels = passive["true labels"]
    active_models = passive["models"]
else:
    raise ValueError("Invalid LEARNING value")

best_score = (2, 0, 0) # score, index, threshold
confusion_matrix = {
    "tp": 0,
    "fp": 0,
    "tn": 0,
    "fn": 0
}

for index, model in enumerate(active_models[:int(len(active_models)/2)]):
    for threshhold in np.arange(0, 1.01, 0.01):
        tp = 0
        fp = 0
        tn = 0
        fn = 0
        for i, pred in enumerate(model["predictions"]):
            value = 0
            if pred > threshhold:
                value = 1
            if value == 1 and true_labels[i] == 1:
                tp += 1
            elif value == 1 and true_labels[i] == 0:
                fp += 1
            elif value == 0 and true_labels[i] == 1:
                fn += 1
            elif value == 0 and true_labels[i] == 0:
                tn += 1
            else:
                raise ValueError("invalid value")
        tpr = tp/(fn+tp)
        fpr = fp/(tn+fp)
        score = ((1-tpr)**2 + fpr**2)**(1/2)
        if score < best_score[0]:
            best_score = (score, index, threshhold)
            confusion_matrix["tp"] = tp
            confusion_matrix["fp"] = fp
            confusion_matrix["tn"] = tn
            confusion_matrix["fn"] = fn
            confusion_matrix["tpr"] = tpr
            confusion_matrix["fpr"] = fpr
            
    print(index)

print(best_score)
print(confusion_matrix)
