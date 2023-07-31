

def monitor_results(model, test_dataset):
    print("Evaluate model with test dataset:")
    history_eval = model.evaluate(test_dataset)
    for x in history_eval:
        print (x,':',history_eval[x])
