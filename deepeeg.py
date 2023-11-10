import pickle
with open("my_features.pkl", "rb") as f:
    feats = pickle.load(f)

from deepeeg import CreateModel
model,_ = CreateModel(feats, network="CNN")

from deepeeg import TrainTestVal
TrainTestVal(model,feats)
