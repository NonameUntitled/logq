from config import GSASRecExperimentConfig

config = GSASRecExperimentConfig(
    dataset_name='steam_time',
    sequence_length=200,
    embedding_dim=128,
    num_heads=1,
    max_batches_per_epoch=100,
    num_blocks=2,
    dropout_rate=0.5,
    negs_per_pos=256,
    path_to_cnt='./datasets/steam_time/item_cnt.pkl'
)
