import argparse

def get_parser():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--tg_num', type = int, default = 403, help = 'oecd test guideline number (403, 412, 413)')
    parser.add_argument('--inhale_type', type = str, default = 'vapour', help = 'route of administration (vapour, aerosol)')
    parser.add_argument('--model', type = str, default = 'logistic', help = 'logistic, lda, qda, plsda, dt, rf, gbt, xgb, lgb, mlp')
    parser.add_argument('--num_run', type = int, default = 10, help = 'the number of run')
    
    return parser