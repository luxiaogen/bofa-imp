def get_model(model_name, args):
    name = model_name.lower()
    if name == "bofa":
        from models.bofa import Learner
        return Learner(args)
    else:
        assert 0
