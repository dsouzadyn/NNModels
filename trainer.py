from models import model_iris as iris
from models import model_zoo as zoo


def train_iris_model():
    iris.train_iris_net()


def train_zoo_model():
    zoo.train_zoo_net()


if __name__ == '__main__':
    train_iris_model()
    train_zoo_model()
