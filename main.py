"""Train from the command line:  python main.py --model mobilenet"""
import argparse

from src import training as T


def cli():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["cnn", "mobilenet"], default="mobilenet")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--ft-epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--data-dir", default=None)
    a = p.parse_args()
    d = T.prepare_data(a.data_dir)
    if a.model == "cnn":
        rid, _ = T.train_cnn(d, a.epochs, a.batch_size)
    else:
        rid, _ = T.train_mobilenet(d, a.epochs, a.ft_epochs, a.batch_size)
    print("MLflow run id:", rid)


if __name__ == "__main__":
    cli()
