
from keras.layers import Input, Dense, Embedding, LSTM
from keras.layers import AveragePooling1D, Reshape
from keras.models import Model

import gym
import gym_textgame

def getActionModel(in_size, hidden_size, action_size, object_size):
    x = Input(shape=(in_size,))
    y = Dense(hidden_size, activation="relu")(x)
    q_sa = Dense(action_size)(y)
    q_so = Dense(object_size)(y)
    return q_sa, q_so

def getReprModel(seq_length, vocab_size, embd_size, hidden_size):
    x = Input(shape=(seq_length,), dtype="int32")
    w_k = Embedding(output_dim=embd_size, input_dim=vocab_size, input_length=seq_length)(x)
    x_k = LSTM(hidden_size, return_sequences=True)(w_k)
    y = AveragePooling1D(pool_length=seq_length, stride=None)(x_k)
    v_s = Reshape((hidden_size,))(y) # remove 2 axis which is 1 caused by averaging
    return v_s

if __name__ == "__main_":
    # parameters
    epsilon = .4  # exploration
    epoch = 200
    max_memory = 500
    hidden_size = 100
    batch_size = 100

    env = gym.make("HomeWorld-v0")
    # action_space = Tuple(Discrete(5), Discrete(8))
    num_actions = env.action_space.spaces[0].n
    num_objects = env.action_space.spaces[1].n

    model = getModel(grid_size,hidden_size,num_actions)

    # Initialize experience replay object
    exp_replay = ExperienceReplay(max_memory=max_memory)

    # Train
    scores = []
    for e in range(epoch):
        loss = 0.
        env.reset()
        game_over = False
        # get initial input
        input_t = env.observe()

        while not game_over:
            input_tm1 = input_t
            # get next action
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, num_actions, size=1)
            else:
                q = model.predict(input_tm1)
                action = np.argmax(q[0])

            # apply action, get rewards and new state
            input_t, reward, game_over = env.act(action)
            if game_over:
                scores.append(reward)

            # store experience
            exp_replay.remember([input_tm1, action, reward, input_t], game_over)

            # adapt model
            inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)
            loss += model.train_on_batch(inputs, targets)
        print("Epoch {:03d}/{} | Loss {:.4f} | running avg {}".format(e, epoch, loss, scores[-1]))

    # Save trained model weights and architecture, this will be used by the visualization code
    model.save_weights("model.h5", overwrite=True)
    with open("model.json", "w") as outfile:
        json.dump(model.to_json(), outfile)
