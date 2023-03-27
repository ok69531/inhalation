import argparse

def get_parser():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--tg_num', type = int, default = 403, help = 'oecd test guideline number (403, 412, 413)')
    parser.add_argument('--inhale_type', type = str, default = 'vapour', help = 'route of administration (vapour, aerosol)')
    parser.add_argument('--model', type = str, default = 'logistic', help = 'logistic, lda, qda, plsda, dt, rf, gbt, xgb, lgb, mlp')
    parser.add_argument('--metric', type = str, default = 'f1', help = 'precision, recall, accuracy, f1, auc(only binary)')
    parser.add_argument('--num_run', type = int, default = 10, help = 'the number of run')
    parser.add_argument('--splitseed', type = int, default = 42)
    parser.add_argument('--neighbor', type = int, default = 5, help = 'the number of neighbors for SMOTE')
    parser.add_argument('--threshold', type = float, default = .5)
    
    return parser