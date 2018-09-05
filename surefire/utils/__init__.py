def multi_task_loss(loss_fns, weights={}):
    def loss_fn(y_pred, y):
        return sum((loss_fns[k](v, y[k]) * weights.get(k, 1.0)) for k, v in y_pred.items())
    return loss_fn
