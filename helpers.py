def unpack_data(data, model_index):
    return {
        "true_labels": data["true labels"],
        "model": data["models"][model_index],
        "predictions": data["models"][model_index]["predictions"],
        "training_instances": data["models"][model_index]["training instances"],
    }
