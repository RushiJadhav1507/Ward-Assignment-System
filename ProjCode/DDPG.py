import math
from turtle import distance
import gym
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import globaldata
problem = "Pendulum-v1"
env = gym.make(problem)

num_states = env.observation_space.shape[0]
print("Size of State Space ->  {}".format(num_states))
num_actions = env.action_space.shape[0]
print("Size of Action Space ->  {}".format(num_actions))

upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]

print("Max Value of Action ->  {}".format(upper_bound))
print("Min Value of Action ->  {}".format(lower_bound))

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)
class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64,target_actor=None,gamma=0,target_critic=None,critic_model=None,critic_optimizer=None,actor_model=None,actor_optimizer=None):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        self.actor_optimizer=actor_optimizer
        self.gamma=gamma
        self.actor_model=actor_model
        self.critic_optimizer =critic_optimizer
        self.critic_model=critic_model
        # Num of tuples to train on.
        self.batch_size = batch_size
        self.target_critic=target_critic
        self.target_actor=target_actor
        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_state_batch, training=True)
            y = reward_batch + self.gamma * self.target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = self.critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.actor_model(state_batch, training=True)
            critic_value = self.critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)

class Test:
    # This update target parameters slowly
    # Based on rate `tau`, which is much less than one.
    def __init__(self,data=None):
        self.x_initial=data
    @tf.function
    def update_target(self,target_weights, weights, tau):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * tau + a * (1 - tau))
    def get_actor(self):
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(num_states,))
        out = layers.Dense(256, activation="relu")(inputs)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

        # Our upper bound is 2.0 for Pendulum.
        outputs = outputs * upper_bound
        model = tf.keras.Model(inputs, outputs)
        return model


    def get_critic(self):
        # State as input
        state_input = layers.Input(shape=(num_states))
        state_out = layers.Dense(16, activation="relu")(state_input)
        state_out = layers.Dense(32, activation="relu")(state_out)

        # Action as input
        action_input = layers.Input(shape=(num_actions))
        action_out = layers.Dense(32, activation="relu")(action_input)

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(256, activation="relu")(concat)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        return model
    def policy(self,state, noise_object):
        sampled_actions = tf.squeeze(self.actor_model(state))
        noise = noise_object()
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise

        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

        return [np.squeeze(legal_action)]
    
    def DistCalc(self,reward1,reward2):
        return reward1-reward2
    
    def process(self,data=globaldata.DF_admission_record):
        self.std_dev = 0.2
        self.x_initial=globaldata.DF_admission_record
        ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(self.std_dev) * np.ones(1))
        best_distance=0
        state_list=[]
        self.actor_model = self.get_actor()
        self.critic_model = self.get_critic()

        self.target_actor = self.get_actor()
        self.target_critic = self.get_critic()

        # Making the weights equal initially
        self.target_actor.set_weights(self.actor_model.get_weights())
        self.target_critic.set_weights(self.critic_model.get_weights())

        # Learning rate for actor-critic models
        critic_lr = 0.002
        actor_lr = 0.001

        self.critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
        self.actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

        total_episodes = 10
        # Discount factor for future rewards
        self.gamma = 0.99
        # Used to update target networks
        tau = 0.005

        prev_reward=0
        buffer = Buffer(50000, 64,self.target_actor,self.gamma,self.target_critic,self.critic_model,self.critic_optimizer,self.actor_model,self.actor_optimizer )

        # To store reward history of each episode
        ep_reward_list = []
        # To store average reward history of last few episodes
        avg_reward_list = []
        ep_q_list=[]
        # Takes about 4 min to train
        for ep in range(total_episodes):

            prev_state = env.reset()
            episodic_reward = 0

            while True:
 
                tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

                action = self.policy(tf_prev_state, ou_noise)
                # Recieve state and reward from environment.
                state, reward, done, info = env.step(action)

                buffer.record((prev_state, action, reward, state))
                episodic_reward += reward
                distance=self.DistCalc(prev_reward,reward)
                if distance>best_distance:
                    best_distance=distance
                    state_list.append(state)
                buffer.learn()
                self.update_target(self.target_actor.variables, self.actor_model.variables, tau)
                self.update_target(self.target_critic.variables, self.critic_model.variables, tau)

                # End this episode when `done` is True
                if done:
                    break

                prev_state = state

            ep_reward_list.append(episodic_reward)

           
            avg_reward = np.mean(ep_reward_list[-40:])
            #print(ep, abs(avg_reward))
            avg_reward_list.append(avg_reward)
        return ep_reward_list

    def getResult(self,input,str2,str1):
        inputlist = input.split(",") if ',' in input else [input]
        rewards=globaldata.rewards
        print("str1",str1)
        max,maxindx=0,1
        isfound=False
        calcrewards=0
        if str1==None or str1=="":
            for ind in globaldata.DF_Result.index:
                #diag=globaldata.DF_Result["Previous_History"][ind].lower()
                #print("diag",diag)
                tests=globaldata.DF_Result["Tests"][ind].lower()
                testlist = tests.split(",")
                print("tests",tests)
                for inputstr in inputlist:
                    print(inputstr)
                    if globaldata.rewards>0:
                        calcrewards=0
                    if inputstr.lower() in testlist:
                        calcrewards+=1
                        print(calcrewards)
                if len(inputlist)==calcrewards:
                    return (globaldata.DF_Result['Wards'][ind])
                elif calcrewards>max:
                    max=calcrewards
                    maxindx=ind
                    calcrewards=0
            if not isfound:
                return globaldata.DF_Result['Wards'][maxindx]
        else:
            for ind in globaldata.DF_Result.index:
                diag=None
                print(globaldata.DF_Result["Previous_History"][ind])
                if type(globaldata.DF_Result["Previous_History"][ind])==str:
                    diag=globaldata.DF_Result["Previous_History"][ind].lower()
                else: #not math.isnan(globaldata.DF_Result["Previous_History"][ind]):
                    diag=None
                print("diag",diag)
                tests=globaldata.DF_Result["Tests"][ind].lower()
                testlist = tests.split(",")
                if diag==None or diag=="": 
                    for inputstr in inputlist:
                        if globaldata.rewards>0:
                            calcrewards=0
                        if inputstr.lower() in testlist:
                            calcrewards+=1
                    if len(inputlist)==calcrewards:
                        return (globaldata.DF_Result['Wards'][ind])
                    elif calcrewards>max:
                        max=calcrewards
                        maxindx=ind
                elif diag==str1:
                    print("inputlist",inputlist)
                    for inputstr in inputlist:
                        if globaldata.rewards>0:
                            calcrewards=0
                        if inputstr.lower() in testlist:
                            calcrewards+=1
                            print("calcrewards",calcrewards)
                    if len(inputlist)==calcrewards:
                        return (globaldata.DF_Result['Wards'][ind])
                    elif calcrewards>max:
                        max=calcrewards
                        maxindx=ind
                        print(maxindx)
            if not isfound:
                return globaldata.DF_Result['Wards'][maxindx]
       
