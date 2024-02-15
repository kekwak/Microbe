import resources.game_env as ge
import resources.neural_network as nn

import torch
import matplotlib.pyplot as plt

def setup_env():
    ge.Settings.unit_lines = 24
    ge.Settings.allow_coin_types = {1, 2, 3}
    ge.Settings.unit_border_should_stop = False

def load_nn():
    state_size = ge.Settings.unit_lines * coin_types
    action_size = len(possible_actions)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    q_network = nn.QNetwork(state_size, action_size, learning_rate, gamma, device).to(device)

    return q_network

def get_params():
    gamma = 0.95
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.995
    learning_rate = 0.0001
    num_episodes = 10000
    max_timesteps = 250

    possible_actions = [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]
    coin_types = len(ge.Settings.allow_coin_types)

    return gamma, epsilon, epsilon_min, epsilon_decay, \
        learning_rate, num_episodes, max_timesteps, \
        possible_actions, coin_types

def train_loop(epsilon):
    for episode in range(num_episodes):
        total_reward = 0

        game.restart_game()
        
        state = helper.get_state(game.unit)
        helper.set_old_score(
            game.unit.get_score()
        )

        for t in range(max_timesteps):
            action = q_network.choose_action(state, epsilon)
            old_state = state
            helper.execute_action(game.unit, action)

            game.root.update()

            new_state = helper.get_state(game.unit)
            helper.set_new_score(
                game.unit.get_score()
            )
            reward = helper.get_reward(game.unit, action)

            ms = len(list(filter(lambda coin: coin.coin_type == 3, game.unit.coins))) - \
                len(list(filter(lambda coin: coin.is_alive and coin.coin_type == 3, game.unit.coins)))
            
            q_network.train_step(old_state, action, reward, new_state)

            state = new_state
            helper.set_old_score(
                helper.get_new_score()
            )

            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
            
            total_reward += reward
        
        graph.update(total_reward)

        if (episode >= 125) and (total_reward >= max(graph.get_scores_list()) * 0.95):
            q_network.save_model(f'resources/train/q_network_model_{episode}-{total_reward:.2f}.pth')

        print(f'Episode {episode}, Total Reward: {total_reward:.4f}, Ms: {ms}')

    plt.savefig('final_learning_rate.png')


if __name__ == '__main__':
    setup_env()

    gamma, epsilon, epsilon_min, epsilon_decay, \
    learning_rate, num_episodes, max_timesteps, \
    possible_actions, coin_types = get_params()

    q_network = load_nn()

    # q_network.load_state_dict(torch.load(f'resources/models/'))
    
    game = ge.Game()
    helper = nn.Helper(possible_actions, coin_types)
    graph = nn.Graph()

    train_loop(epsilon)