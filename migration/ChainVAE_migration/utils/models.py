import tensorflow as tf
import tensorflow_probability as tfp
import os
import keras


tfk = keras
tfkl = keras.layers
tfpl = tfp.layers
tfd = tfp.distributions


class shared_NN(tfk.Model):
    def __init__(self, configs, **kwargs):
        super().__init__(**kwargs)

        self.configs=configs

        neurons = configs["neurons"]
        hidden_layers = configs["hidden_layers"]
        activation = configs["activation"]
        self.model_name = configs["name"]

        self.set_distributions()
        self.beta=tf.Variable(1.0,trainable=False)

        model_layers=[]

        if isinstance(neurons, list):
            if len(neurons) != hidden_layers:
                raise ValueError("Number of layers is not consistent: hidden_layers != len(neurons)")
            for i,n_i in enumerate(neurons):
                model_layers.append(tfkl.Dense(n_i, activation=activation, name=f"{self.model_name}_dense_{i}"))  

        else:
            for i in range(hidden_layers):
                model_layers.append(tfkl.Dense(neurons, activation=activation, name=f"{self.model_name}_dense_{i}"))
        
        model_layers+=self.distribution_layers

        self.model_layers=model_layers

    def set_distributions(self):
        distribution = self.configs["distribution"]
        output_shape = self.configs["output_shape"]

        if distribution == "normal":
            self.prior = tfd.Independent(tfd.Normal(loc=tf.zeros(output_shape), scale=1), reinterpreted_batch_ndims=1)
            self.distribution_layers=[
                tfkl.Dense(2*output_shape,activation=None,name=f"{self.model_name}_dense_dist_params"),
                tfpl.DistributionLambda(
                    lambda t: tfd.Independent(
                        tfd.Normal(
                            loc=t[...,:output_shape], 
                            scale=tf.math.sqrt(tf.math.exp(t[...,output_shape:]))), 
                        reinterpreted_batch_ndims=1),
                        name=f"{self.model_name}_laplace"
                )
            ]

        elif distribution == "laplace":
            self.prior = tfd.Independent(tfd.Laplace(loc=tf.zeros(output_shape), scale=1),reinterpreted_batch_ndims=1)
            self.distribution_layers=[
                tfkl.Dense(2*output_shape,activation=None,name=f"{self.model_name}_dense_dist_params"),
                tfpl.DistributionLambda(
                    lambda t: tfd.Independent(
                        tfd.Laplace(
                            loc=t[...,:output_shape], 
                            scale=tf.math.sqrt(tf.math.exp(t[...,output_shape:]))), 
                        reinterpreted_batch_ndims=1),
                        name=f"{self.model_name}_laplace"
                )
            ]
        
        elif distribution == "observation_normal":
            
            self.distribution_layers=[
                tfkl.Dense(tf.reduce_prod(output_shape),activation=None),
                tfpl.DistributionLambda(
                    lambda t: tfd.Independent(
                        tfd.Normal(
                            #loc=t[...,:latent_dims], 
                            #scale=tf.math.softplus(t[...,latent_dims:])), 
                            loc=t[...,:],
                            scale=0.4),
                            reinterpreted_batch_ndims=1),
                        name=f"{self.model_name}_observation_normal"

                )
            ]

        else:
            raise ValueError(f"Distribution {distribution} not implemented yet")

    def call(self, x):
        for layer in self.model_layers:
            x=layer(x)
        return x

    def model(self):
        input_shape=self.configs["input_shape"]
        inputs=tfk.Input(shape=input_shape, name=f"{self.model_name}_input")
        return tfk.Model(inputs=[inputs], outputs=self.call(inputs),name=self.model_name)

    
class VAE_model(tfk.Model):
    def __init__(self,configs):
        super().__init__()

        self.configs=configs
                    
        self.q_z1_x_model=shared_NN(configs["q_z1_x_configs"], name="q_z1_x")
        self.q_z2_z1_model=shared_NN(configs["q_z2_z1_configs"], name="q_z2_z1")
        self.p_z1_z2_model=shared_NN(configs["p_z1_z2_configs"], name="p_z1_z2")
        self.p_x_z1_model=shared_NN(configs["p_x_z1_configs"], name="p_x_z1")

        self.beta1=tf.Variable(1.0,trainable=False)
        self.beta2=tf.Variable(1.0,trainable=False)


    def call(self, x):
        q_z1_x=self.q_z1_x_model(x) 
        z1_sample=q_z1_x.sample()
        q_z2_z1=self.q_z2_z1_model(z1_sample)
        z2_sample=q_z2_z1.sample()
        p_z1_z2=self.p_z1_z2_model(z2_sample)
        p_x_z1=self.p_x_z1_model(z1_sample)


        nll=tf.reduce_mean(-p_x_z1.log_prob(x))
        self.add_loss(nll)
        self.add_metric(nll, name="nll", aggregation="mean")

        reg1=tf.reduce_mean(-q_z1_x.entropy())
        reg1+=tf.reduce_mean(-p_z1_z2.log_prob(z1_sample))
        self.add_loss(reg1*self.beta1)
        self.add_metric(reg1, name="reg1", aggregation="mean")
        

        kl2=tf.reduce_mean(tfp.distributions.kl_divergence(q_z2_z1, self.q_z2_z1_model.prior))
        self.add_loss(kl2*self.beta2) 
        self.add_metric(kl2, name="kl2", aggregation="mean")
        self.add_metric(self.beta1, name="beta1", aggregation="mean")
        self.add_metric(self.beta2, name="beta2", aggregation="mean")
        return p_x_z1


    def model(self):
        inputs=tfk.Input(shape=self.configs["q_z1_x_configs"]["input_shape"])
        return tfk.Model(inputs=[inputs], outputs=self.call(inputs))

def nested_model_summary(model,path=None):
    if path:
        with open(os.path.join(path,'summary.txt'),'w+') as f:
            for layer in model.layers:
                layer.model().summary(print_fn=lambda x: f.write(x + '\n'))
    
    else:
        for layer in model.layers:
            layer.model().summary()

def nested_model_visual(model,path):
    
    for layer in model.layers:
        file_path=os.path.join(path,layer.name+".png")
        tfk.utils.plot_model(layer.model(),expand_nested=True,show_shapes=True,to_file=file_path)
    

def get_dummy_model():
    x_dim=400
    z1_dim=450
    z2_dim=75
    activation="softplus"

    q_z1_x_configs={
        "name": "q_z1_x",
        "input_shape": x_dim,
        "neurons": 500,
        "hidden_layers": 3,
        "activation": activation,
        "output_shape": z1_dim,
        "distribution": "laplace",
    }

    q_z2_z1_configs={
        "name": "q_z2_z1",
        "input_shape": z1_dim,
        "neurons": 500,
        "hidden_layers": 3,
        "activation": activation,
        "output_shape": z2_dim,
        "distribution": "laplace",
    }

    p_x_z1_configs={
        "name": "p_x_z1",
        "input_shape": z1_dim,
        "neurons": 500,
        "hidden_layers": 3,
        "activation": activation,
        "output_shape": x_dim,
        "distribution": "observation_normal",
    }

    p_z1_z2_configs={
        "name": "p_z1_z2",
        "input_shape": z2_dim,
        "neurons": 500,
        "hidden_layers": 3,
        "activation": activation,
        "output_shape": z1_dim,
        "distribution": "laplace",
    }

    model_configs={
        "q_z1_x_configs": q_z1_x_configs,
        "q_z2_z1_configs": q_z2_z1_configs,
        "p_z1_z2_configs": p_z1_z2_configs,
        "p_x_z1_configs": p_x_z1_configs,
    }

    dummy_model=VAE_model(model_configs)

    return dummy_model
