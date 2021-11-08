import numpy as np
import tensorflow as tf

import models


def logit(x):
    return np.log(x/(1-x))


def normlogit(y, M=0.997, m=0.003):
    #log (M) = -log (m) = 5.806
    y = y.clip(min=m, max=M)
    lM = -logit(m)
    return (logit(y) + lM) / (2*lM)


def ensemble_matrices(models, batch_gen, num_batches=None, logit=False):
    p = len(models)
    ATA = np.zeros((p,p))
    ATy = np.zeros(p)
    
    if num_batches is None:
        num_batches = len(batch_gen)
    
    for k in range(num_batches):
        (X, Y) = batch_gen[k]
        Y_pred = np.array([m.predict(X).ravel().astype(np.float64) for m in models])
        if logit:
            Y[0] = normlogit(Y[0])
            Y_pred = normlogit(Y_pred)
        N = Y_pred.shape[1]
        ATA_batch = Y_pred.dot(Y_pred.T) / N
        ATy_batch = Y_pred.dot(Y[0].ravel()) / N
        ATA = (k*ATA + ATA_batch) / (k+1)
        ATy = (k*ATy + ATy_batch) / (k+1)

        w = ensemble_weights(ATA, ATy, regularization=1e-4*np.diag(ATA).mean())
        print(w)

    return (ATA, ATy)


def ensemble_weights(ATA, ATy, regularization=0.0):
    ATA = ATA + regularization * np.eye(ATA.shape[0])
    return np.linalg.solve(ATA, ATy)


def ensemble_weights_sum1(ATA, ATy, regularization=0.0):
    p = ATA.shape[0]
    q = np.ones((p,1), dtype=ATA.dtype)
    B = np.vstack([
        np.hstack([ATA, -0.5*q]),
        np.hstack([q.T, [[0]]])
    ])
    c = np.hstack([ATy, 1])
    return np.linalg.solve(B, c)[:-1]


def logit_tf(x):
    return tf.math.log(x/(1-x))


def normlogit_tf(y, M=tf.constant(0.997), m=tf.constant(0.003)):
    #log (M) = -log (m) = 5.806
    y = tf.clip_by_value(y, m, M)
    lM = -logit_tf(m)
    return (logit_tf(y) + lM) / (2*lM)


def inv_logit_tf(x):
    exp_mx = tf.math.exp(-x)
    return tf.constant(1.0)/(tf.constant(1.0)+exp_mx)


def inv_normlogit_tf(y, M=tf.constant(0.997), m=tf.constant(0.003)):
    lM = -logit_tf(m)
    return inv_logit_tf(2*lM*y-lM)


def weighted_model(model_list, weights, output_var, logit=False):
    inputs = tf.keras.Input(shape=model_list[0].input_shape[1:])
    outputs = (m(inputs) for m in model_list)
    if logit:
        outputs = (normlogit_tf(o) for o in outputs)
    weighted_outputs = [
        tf.constant(float(w))*y for (w,y) in zip(weights,outputs)
    ]
    output = weighted_outputs[0]
    for out in weighted_outputs[1:]:
        output = output + out

    if logit:
        output = inv_normlogit_tf(output)

    output = tf.keras.layers.Lambda(lambda x: x, name=output_var)(output)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    models.compile_model(model, [output_var], optimizer='sgd')
    
    return model

def model_correlation(models, batch_gen):
    p = len(models)
    Ex = np.zeros(p+1)
    Ex2 = np.zeros((p+1,p+1))

    for k in range(len(batch_gen)):
        print("{}/{}".format(k,len(batch_gen)))
        (X, Y) = batch_gen[k]
        Y_pred = [m.predict(X).ravel().astype(np.float64) for m in models]
        Y_pred.append(Y[0].ravel().astype(np.float64))
        Y_pred = np.array(Y_pred)
        N = Y_pred.shape[1]
        Ex = (k*Ex + Y_pred.mean(axis=1)) / (k+1)
        Ex2 = (k*Ex2 + Y_pred.dot(Y_pred.T)/N) / (k+1)

        cov = Ex2 - np.outer(Ex,Ex)
        diag = np.diag(1.0/np.sqrt(np.diag(cov)))
        corr = np.matmul(diag, np.matmul(cov, diag))
        print(corr)

    return corr
