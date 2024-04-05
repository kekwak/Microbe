import resources.game_env as ge
import resources.neural_network as nn
import torch
import time

def setup_env():
    ge.Settings.unit_lines = 24
    ge.Settings.allow_coin_types = {1, 2, 3}
    ge.Settings.unit_border_should_stop = False

    ge.Settings.screen_width = 900
    ge.Settings.screen_height = 600

    ge.Settings.coins_number = 85
    ge.Settings.coins_random_number = 15

def load_nn(modelname):
    state_size = ge.Settings.unit_lines * coin_types
    action_size = len(possible_actions)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    q_network = nn.QNetwork(state_size, action_size, 0, 0, device).to(device)

    q_network.load_state_dict(torch.load(f'resources/models/{modelname}', map_location=device))
    return q_network.eval()


if __name__ == '__main__':
    setup_env()

    modelname = 'q_network_model_441-709.92_alpha.pth'

    possible_actions = [(0, 0), (-1, 0), (0, 1), (1, 0), (0, -1)]
    coin_types = len(ge.Settings.allow_coin_types)

    max_timesteps = 750
    frames_delay = 1000 / 90

    q_network = load_nn(modelname)

    game = ge.Game()
    helper = nn.Helper(possible_actions, coin_types)

    window_alive = True
    while window_alive:
        game.restart_game()
        
        state = helper.get_state(game.unit)

        for t in range(max_timesteps):
            time_rec = time.time()

            action = q_network.choose_action(state)
            helper.execute_action(game.unit, action)

            game.root.update()

            state = helper.get_state(game.unit)

            dif = time.time() - time_rec
            if dif < frames_delay:
                time.sleep((frames_delay - dif) / 1000)