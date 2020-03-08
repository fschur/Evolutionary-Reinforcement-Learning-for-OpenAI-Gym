import numpy as np
import random
import gym


"""
Implementation of Augmented-Random-Search for gym environments.
"""


def numpy_softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)


def compute_rewards_v1(genes, env, version, H, render, train, action_mode, mu, sigma):
    """
    Performs one episode in the environment for every given weights/gene.
    Returns the rewards and for version V2 the encountered states.
    """
    rewards = []
    states = []
    for gene in genes:
        done = False
        state = env.reset()
        rewards_sum = 0
        step = 0
        while not done and step <= H:
            step += 1

            if version == "V2":
                # In V2 the states/oberservations are normalized and saved
                if train:
                    states.append(state)
                state = np.matmul(np.diag(np.sqrt(sigma)), (state - mu))

            if action_mode == "continuous":
                action = np.matmul(gene, state)
            else:
                values = np.matmul(gene, state)
                probs = numpy_softmax(values)
                action = np.random.choice(len(values), p=probs)

            if render:
                env.render()

            state, reward, done, _ = env.step(action)
            rewards_sum += reward

        rewards.append(rewards_sum)
    rewards = np.array(rewards)

    if train:
        rewards = rewards / np.std(rewards)
        return rewards, states
    else:
        return rewards


def main(step_size=0.5, nu=0.5, H=np.inf, num_individuals=50, num_selected=50, measure_step=100, num_episodes=5000,
         render=True, env_name="CartPole-v1", version="V2", measure_repeat=100, seed=42):
    """
    :param step_size: How much the weights are adjusted with each iteration. Similar to learning rate.
    :param nu: How much the weights are mutated in order to find better policies. Increases exploration.
    :param H: Horizon (maximal number of steps taken per episode)
    :param num_individuals: Number of mutated weights generated each iteration.
    :param num_selected: Number of mutated weights used in the update step.
    :param measure_step: After "measure_step" many episodes in the environment the performance is measured.
    :param num_episodes: The number of total episodes in the environment. Increase this for longer training.
    :param render: Renders the environment once for every time the performance is measured.
    :param env_name: Name of the gym environment.
    :param version: Options: V1, V2. Version V2 will normalize the observations.
    :param measure_repeat: The number of episodes in the environment used to evaluate the performance.
    :param seed: random seed
    """

    assert (version == "V1" or version == "V2"), "Possible versions are: V1, V2"
    assert (num_selected <= num_individuals), "'num_selected' must be smaller or equal 'num_individuals'"

    env = gym.make(env_name)

    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)

    # check whether the environment has a continuous or discrete action space.
    if type(env.action_space) == gym.spaces.Discrete:
        action_mode = "discrete"
    elif type(env.action_space) == gym.spaces.Box:
        action_mode = "continuous"
    else:
        raise Exception("action space is not known")

    # Get number of actions for the discrete case and action dimension for the continuous case.
    if action_mode == "continuous":
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    state_dim = env.observation_space.shape[0]

    # Performance is measured with respect to the amount of episodes taken in the environment.
    # In every iteration "num_indiviuals" many episodes are performed. For that the variable
    # "measure_step" is adjusted.
    num_iter = num_episodes/(2*num_individuals)
    assert (float.is_integer(num_iter)), "'num_episodes' needs to be a multiple of 2*'num_individuals'"
    num_iter = int(num_iter)
    measure_step_adjusted = measure_step/(2*num_individuals)
    assert (float.is_integer(measure_step_adjusted)), "'measure_step' needs to be a multiple of 2*'num_individuals'"
    measure_step_adjusted = int(measure_step_adjusted)

    current_policy = np.zeros((action_dim, state_dim))
    if version == "V2":
        mu = np.zeros(state_dim)
        sigma = np.ones(state_dim)
        total_num_states = 0
    else:
        mu = None
        sigma = None

    performance = []
    for iteration in range(num_iter):
        # Mutates the linear policy weights randomly in both directions.
        translations = np.random.normal(0, 1, (num_individuals, action_dim, state_dim))
        policies_plus = current_policy + nu*translations
        policies_minus = current_policy - nu*translations

        # compute the performance of the mutated policies
        rewards_plus, states_plus = compute_rewards_v1(policies_plus, env, version, H, render=False, train=True, action_mode=action_mode, mu=mu, sigma=sigma)
        rewards_minus, states_minus = compute_rewards_v1(policies_minus, env, version, H, render=False, train=True, action_mode=action_mode, mu=mu, sigma=sigma)

        # update the policy with the most important mutated policies (measured by relative reward)
        relative_rewards = rewards_plus - rewards_minus
        best_idx = np.array(np.argpartition(relative_rewards, -num_selected)[-num_selected:])
        best_relative_rewards = relative_rewards[best_idx]
        std = np.std(best_relative_rewards)
        direction = (np.expand_dims(np.expand_dims(best_relative_rewards, axis=1), axis=1)*translations).sum(0)
        current_policy += (step_size/(num_selected*std))*direction

        # computes running mean and standard deviation of the encountered states/observations
        if version == "V2":
            states = np.array(states_minus + states_plus)
            num_new_states = len(states)
            num_old = total_num_states
            total_num_states += num_new_states
            mean = states.mean(0)
            sqrt_diff = (mean-mu)**2
            mu = (num_new_states*mean + num_old*mu)/total_num_states
            sigma += states.std(0) + sqrt_diff*num_new_states*num_old  / total_num_states

        # After "measure_step" many episodes performed in the environment the performance is measured.
        if iteration % measure_step_adjusted == 0:
            mean_rewards = np.mean([compute_rewards_v1([current_policy], env, version, render=(i==1)*render, H=np.inf, train=False, action_mode=action_mode, mu=mu, sigma=sigma) for i in range(measure_repeat)])
            performance.append([iteration*2*num_individuals, mean_rewards])
            print("Episode: ", performance[-1][0])
            print("rewards: ", performance[-1][1])


if __name__ == '__main__':
    main()