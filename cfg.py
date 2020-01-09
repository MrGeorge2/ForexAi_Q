DATAFRAME_NAME = 'EURUSD_m15_Ask_ready.csv'
NUMBER_OF_SAMPLES = 100

EPISODES = 5000
TICQTY_MAX = 55000
HOLD_REWARD = -0.1
REWARD_FOR_PIPS = 10_000
TIMES_FACTOR = 15

ACTION_DECODE = {
    0: 0,
    1: 0.5,
    2: 1,
}
