from solution import test_lambdas, LogRegLearner,load
from draw import draw_decision


if __name__ == "__main__":
    # izpis tabele za različne vrednosti lambda pri prečnem preverjanju in testiranju na učnih podatkih
    test_lambdas()

    X, y = load('reg.data')

    learner = LogRegLearner(lambda_=0.0001)
    classifier = learner(X, y)
    draw_decision(X, y, classifier, 0, 1)

    learner = LogRegLearner(lambda_=0.000001)
    classifier = learner(X, y)
    draw_decision(X, y, classifier, 0, 1)

    learner = LogRegLearner(lambda_=0.1)
    classifier = learner(X, y)
    draw_decision(X, y, classifier, 0, 1)

