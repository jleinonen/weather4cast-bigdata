import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Layer
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D, UpSampling2D
from tensorflow.keras.layers import TimeDistributed, Lambda
from tensorflow.keras.optimizers import Adam

from blocks import res_block, ConvBlock, ResBlock
import datasets
import ensemble
from optimizers import AdaBeliefOptimizer
from rnn import CustomGateGRU, ConvGRU, ResGRU


file_dir = os.path.dirname(os.path.abspath(__file__))


def variable_activation(var_name):
    return Activation('sigmoid', name=var_name)


def rnn2_model(num_inputs=1,
    num_outputs=1,
    past_timesteps=4,
    future_timesteps=32,
    input_names=None,
    output_names=None
    ):

    past_in = Input(shape=(past_timesteps,None,None,num_inputs),
        name="past_in")
    inputs = [past_in]

    def conv_gate(channels, activation='sigmoid'):
        return Conv2D(channels, kernel_size=(3,3), padding='same',
            activation=activation)

    # recurrent downsampling
    block_channels = [32, 64, 128, 256]
    xt = past_in
    intermediate = []
    for channels in block_channels:        
        xt = res_block(channels, time_dist=True, stride=2)(xt)
        initial_state = Lambda(lambda y: tf.zeros_like(y[:,0,...]))(xt)
        xt = CustomGateGRU(
            update_gate=conv_gate(channels),
            reset_gate=conv_gate(channels),
            output_gate=conv_gate(channels, activation=None),
            return_sequences=True,
            time_steps=past_timesteps
        )([xt,initial_state])
        intermediate.append(
            Conv2D(channels, kernel_size=(3,3), 
                activation='relu', padding='same')(xt[:,-1,...])
        )

    # recurrent upsampling
    xt = Lambda(lambda y: tf.zeros_like(
        tf.repeat(y[:,:1,...],future_timesteps,axis=1)
    ))(xt)    
    for (i,channels) in reversed(list(enumerate(block_channels))):
        xt = CustomGateGRU(
            update_gate=conv_gate(channels),
            reset_gate=conv_gate(channels),
            output_gate=conv_gate(channels, activation=None),
            return_sequences=True,
            time_steps=future_timesteps
        )([xt,intermediate[i]])
        xt = TimeDistributed(UpSampling2D(interpolation='bilinear'))(xt)
        xt = res_block(block_channels[max(i-1,0)],
            time_dist=True, activation='relu')(xt)

    seq_out = TimeDistributed(Conv2D(num_outputs, kernel_size=(1,1),
        activation='sigmoid'))(xt)

    outputs = [
        Lambda(lambda x: x, name=name)(seq_out[...,i:i+1])
        for (i,name) in enumerate(output_names)
    ]

    model = Model(inputs=inputs, outputs=outputs)

    return model


def rnn3_model(num_inputs=1,
    num_outputs=1,
    past_timesteps=4,
    future_timesteps=32,
    input_names=None,
    output_names=None
    ):

    past_in = Input(shape=(past_timesteps,None,None,num_inputs),
        name="past_in")
    inputs = past_in

    for (i,name) in enumerate(input_names):
        if name == "crr_intensity":
            ip = past_in[...,i:i+1]
            ip = tf.math.log(tf.math.maximum(ip, tf.constant(0.0002)))
            past_in = tf.concat([past_in[...,:i], ip, past_in[...,i+1:]], axis=-1)

    block_channels = [32, 64, 128, 256]

    # recurrent downsampling    
    xt = past_in
    intermediate = []
    for channels in block_channels:        
        xt = ResBlock(channels, time_dist=True, stride=2)(xt)
        initial_state = Lambda(lambda y: tf.zeros_like(y[:,0,...]))(xt)
        xt = ResGRU(
            channels, return_sequences=True, time_steps=past_timesteps
        )([xt,initial_state])
        intermediate.append(ResBlock(channels)(xt[:,-1,...]))

    # recurrent upsampling
    xt = Lambda(lambda y: tf.zeros_like(
        tf.repeat(y[:,:1,...],future_timesteps,axis=1)
    ))(xt)    
    for (i,channels) in reversed(list(enumerate(block_channels))):
        xt = ResGRU(
            channels, return_sequences=True, time_steps=future_timesteps
        )([xt,intermediate[i]])
        xt = TimeDistributed(UpSampling2D(interpolation='bilinear'))(xt)
        xt = ResBlock(block_channels[max(i-1,0)], time_dist=True)(xt)

    seq_out = TimeDistributed(Conv2D(num_outputs, kernel_size=(1,1)))(xt)

    act = {} # can be used to define output-specific activations
    outputs = [
        act.get(name, Activation('sigmoid', name=name))(seq_out[...,i:i+1])
        for (i,name) in enumerate(output_names)
    ]

    model = Model(inputs=inputs, outputs=outputs)

    return model


def rnn4_model(num_inputs=1,
    num_outputs=1,
    past_timesteps=4,
    future_timesteps=32,
    input_names=None,
    output_names=None
    ):

    past_in = Input(shape=(past_timesteps,None,None,num_inputs),
        name="past_in")
    inputs = past_in

    for (i,name) in enumerate(input_names):
        if name == "crr_intensity":
            ip = past_in[...,i:i+1]
            ip = tf.math.log(tf.math.maximum(ip, tf.constant(0.0002)))
            past_in = tf.concat([past_in[...,:i], ip, past_in[...,i+1:]], axis=-1)

    block_channels = [32, 64, 128]

    # recurrent downsampling    
    xt = past_in
    intermediate = []
    for channels in block_channels:        
        xt = ResBlock(channels, time_dist=True, stride=2)(xt)
        initial_state = Lambda(lambda y: tf.zeros_like(y[:,0,...]))(xt)
        xt = ResGRU(
            channels, return_sequences=True, time_steps=past_timesteps
        )([xt,initial_state])
        intermediate.append(ResBlock(channels)(xt[:,-1,...]))

    # recurrent upsampling
    xt = Lambda(lambda y: tf.zeros_like(
        tf.repeat(y[:,:1,...],future_timesteps,axis=1)
    ))(xt)    
    for (i,channels) in reversed(list(enumerate(block_channels))):
        xt = ResGRU(
            channels, return_sequences=True, time_steps=future_timesteps
        )([xt,intermediate[i]])
        xt = TimeDistributed(UpSampling2D(interpolation='bilinear'))(xt)
        xt = ResBlock(block_channels[max(i-1,0)], time_dist=True)(xt)

    seq_out = TimeDistributed(Conv2D(num_outputs, kernel_size=(1,1)))(xt)

    act = {} # can be used to define output-specific activations
    outputs = [
        act.get(name, Activation('sigmoid', name=name))(seq_out[...,i:i+1])
        for (i,name) in enumerate(output_names)
    ]

    model = Model(inputs=inputs, outputs=outputs)

    return model


def rnn5_model(num_inputs=1,
    num_outputs=1,
    past_timesteps=4,
    future_timesteps=32,
    input_names=None,
    output_names=None
    ):

    past_in = Input(shape=(past_timesteps,None,None,num_inputs),
        name="past_in")
    inputs = past_in

    block_channels = [32, 64, 128, 256]

    # recurrent downsampling    
    xt = past_in
    intermediate = []
    for channels in block_channels:        
        xt = ConvBlock(channels, time_dist=True, stride=2)(xt)
        initial_state = Lambda(lambda y: tf.zeros_like(y[:,0,...]))(xt)
        xt = ConvGRU(
            channels, return_sequences=True, time_steps=past_timesteps
        )([xt,initial_state])
        intermediate.append(ConvBlock(channels)(xt[:,-1,...]))

    # recurrent upsampling
    xt = Lambda(lambda y: tf.zeros_like(
        tf.repeat(y[:,:1,...],future_timesteps,axis=1)
    ))(xt)    
    for (i,channels) in reversed(list(enumerate(block_channels))):
        xt = ConvGRU(
            channels, return_sequences=True, time_steps=future_timesteps
        )([xt,intermediate[i]])
        xt = TimeDistributed(UpSampling2D(interpolation='bilinear'))(xt)
        xt = ConvBlock(block_channels[max(i-1,0)], time_dist=True)(xt)

    seq_out = TimeDistributed(Conv2D(num_outputs, kernel_size=(1,1)))(xt)

    act = {} # can be used to define output-specific activations
    outputs = [
        act.get(name, Activation('sigmoid', name=name))(seq_out[...,i:i+1])
        for (i,name) in enumerate(output_names)
    ]

    model = Model(inputs=inputs, outputs=outputs)

    return model


def crr_combo_model(model_func=rnn2_model, **kwargs):
    num_inputs = kwargs.get("num_inputs", 4)
    past_timesteps = kwargs.get("past_timesteps", 4)
    past_in = Input(shape=(past_timesteps,None,None,num_inputs),
        name="past_in")
    inputs = past_in

    input_names = kwargs["input_names"]
    for (i,name) in enumerate(input_names):
        if name == "crr_intensity":
            crr = past_in[...,i:i+1]
            break

    dry_model = model_func(**kwargs)
    wet_model = model_func(**kwargs)

    dry_out = dry_model(inputs)
    wet_out = wet_model(inputs)

    has_rain = tf.math.reduce_any(
        crr > tf.constant(0.026), axis=(1,2,3,4), keepdims=True
    )
    output = tf.where(has_rain, wet_out, dry_out)
    output = Activation('linear', name="crr_intensity")(output)

    model = Model(inputs=inputs, outputs=output)
    # ugly hack that allows loading weights into these models later
    model._dry_model = dry_model
    model._wet_model = wet_model

    return model


def rounded_mse(y_true, y_pred):
    return tf.reduce_mean(
        tf.square(tf.math.round(y_pred)-y_true),
        axis=-1
    )


def logit(x):
    return tf.math.log(x/(1-x))


def normlogit_loss(y_true, y_pred, M=0.997, m=0.003):
    #log (M) = -log (m) = 5.806
    y_true = tf.clip_by_value(y_true, m, M)
    y_pred = tf.clip_by_value(y_pred, m, M)
    lM = -logit(m)
    y_true = (logit(y_true) + lM) / (2*lM)
    y_pred = (logit(y_pred) + lM) / (2*lM)

    return tf.reduce_mean(tf.square(y_pred-y_true), axis=-1)


variable_weights = {
    "temperature": 1.0/0.03163512,
    "crr_intensity": 1.0/0.00024158,
    "asii_turb_trop_prob": 1.0/0.00703378,
    "cma": 1.0/0.19160305
}


def compile_model(model, output_vars, optimizer='adabelief'):
    var_losses = {
        "asii_turb_trop_prob": normlogit_loss,
    }
    losses = [var_losses.get(v, 'mse') for v in output_vars]
    if len(output_vars) > 1:        
        weights = [variable_weights[v]/len(output_vars) for v in output_vars]
    else:
        weights = [1.0]

    if "cma" in output_vars:
        metrics = {"cma": rounded_mse}
    else:
        metrics = None

    if optimizer == "adam":
        optimizer = Adam()
    elif optimizer == "adabelief":
        optimizer = AdaBeliefOptimizer(epsilon=1e-14)

    if compile_model:
        model.compile(loss=losses, loss_weights=weights,
            optimizer=optimizer, metrics=metrics)


def init_model(batch_gen, model_func=rnn4_model, compile=True, 
    init_strategy=True, **kwargs):

    (past_timesteps, future_timesteps) = batch_gen.sequence_length
    (num_inputs, num_outputs) = batch_gen.num_vars

    input_vars = []
    for (_, variables) in batch_gen.variables[0].items():
        input_vars.extend(variables)
    
    output_vars = []
    for (_, variables) in batch_gen.variables[1].items():
        output_vars.extend(variables)

    if init_strategy and tf.config.list_physical_devices('GPU'):
        strategy = tf.distribute.MirroredStrategy()
    else:  # use default strategy
        strategy = tf.distribute.get_strategy()

    with strategy.scope():
        model = model_func(
            past_timesteps=past_timesteps,
            future_timesteps=future_timesteps,
            num_inputs=num_inputs,
            num_outputs=num_outputs,
            input_names=input_vars,
            output_names=output_vars,
            **kwargs
        )

        if compile:
            compile_model(model, output_vars)
    
    return model


def combined_model(models, output_names):
    past_in = Input(shape=models[0].input_shape[1:],
        name="past_in")
    outputs = [
        Layer(name=name)(model(past_in))
        for (model, name) in zip(models, output_names)
    ]
    comb_model = Model(inputs=[past_in], outputs=outputs)

    return comb_model


def combined_model_with_weights(
    batch_gen_valid,
    var_weights=[
        ("CTTH", "temperature", "../models/srnn_adabelief_2-temperature.h5", rnn4_model),
        ("CRR", "crr_intensity", "../models/srnn_adabelief_3-crr_intensity.h5", crr_combo_model),
        ("ASII", "asii_turb_trop_prob", "../models/srnn_adabelief_1-asii_turb_trop_prob.h5", rnn4_model),
        ("CMA", "cma", "../models/srnn_adabelief_1-cma.h5", rnn4_model)
    ],
):
    models = []
    output_vars = []
    if tf.config.list_physical_devices('GPU'):
        strategy = tf.distribute.MirroredStrategy()
    else:  # use default strategy
        strategy = tf.distribute.get_strategy()

    with strategy.scope():
        for (dataset, variable, weights, model_func) in var_weights:
            datasets.setup_univariate_batch_gen(batch_gen_valid, dataset, variable)
            model = init_model(batch_gen_valid, model_func=model_func, 
                compile=False, init_strategy=False)
            model.load_weights(weights)
            models.append(model)
            output_vars.append(variable)

        comb_model = combined_model(models, output_vars)
        compile_model(comb_model, output_vars)

    return comb_model


def ensemble_model_with_weights(
    batch_gen_valid, *, var_models, logit=False
):
    models = []
    output_vars = []
    if tf.config.list_physical_devices('GPU'):
        strategy = tf.distribute.MirroredStrategy()
    else:  # use default strategy
        strategy = tf.distribute.get_strategy()

    with strategy.scope():
        for (dataset, variable, model_list) in var_models:
            datasets.setup_univariate_batch_gen(batch_gen_valid, dataset, variable)

            var_model_list = []
            var_ensemble_weights = []

            for (model_weights, model_func, ensemble_weight) in model_list:            
                model = init_model(batch_gen_valid, model_func=model_func, 
                    compile=False, init_strategy=False)
                model.load_weights(model_weights)
                var_model_list.append(model)
                var_ensemble_weights.append(ensemble_weight)

            weighted_model = ensemble.weighted_model(
                var_model_list, var_ensemble_weights, variable,
                logit=(logit and (variable=="asii_turb_trop_prob"))
            )
            models.append(weighted_model)
            output_vars.append(variable)

        comb_model = combined_model(models, output_vars)
        compile_model(comb_model, output_vars)

    return comb_model


def train_model(model, batch_gen_train, batch_gen_valid,
    weight_fn="model.h5", monitor="val_loss", min_delta=0.0):

    fn = os.path.join(file_dir, "..", "models", weight_fn)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        fn, save_weights_only=True, save_best_only=True, mode="min",
        monitor=monitor
    )
    reducelr = tf.keras.callbacks.ReduceLROnPlateau(
        patience=3, mode="min", factor=0.2, monitor=monitor,
        min_delta=min_delta, verbose=1
    )
    earlystop = tf.keras.callbacks.EarlyStopping(
        patience=6, mode="min", restore_best_weights=True,
        monitor=monitor
    )

    callbacks = [checkpoint, reducelr, earlystop]

    model.fit(
        batch_gen_train,
        epochs=100,
        steps_per_epoch=len(batch_gen_train),
        validation_data=batch_gen_valid,
        callbacks=callbacks
    )
