DATAFRAME_NAME = 'EURUSD_m15_Ask_ready.csv'
NUMBER_OF_SAMPLES = 200

EPISODES = 5000
TICQTY_MAX = 55000
HOLD_REWARD = -1.5
DIVIDE_PRICE_UNDER_LOCAL_MINIMA = 10
REWARD_FOR_PIPS = 10_000
TIMES_FACTOR = 5

ACTION_DECODE = {
    0: 0,
    1: 0.5,
    2: 1,
}
