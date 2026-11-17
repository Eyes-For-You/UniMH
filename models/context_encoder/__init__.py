from .eth_encoder import ETHEncoder


def build_context_encoder(config, use_pre_norm):
    if config.NAME != 'ETHEncoder':
        raise ValueError('This teacher-only package supports ETHEncoder.')
    return ETHEncoder(config=config, use_pre_norm=use_pre_norm)
