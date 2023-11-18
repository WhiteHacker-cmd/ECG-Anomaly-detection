from utils import plot_prediction, predict
import seaborn as sns
import matplotlib.pyplot as plt


def test_model(model, train_dataset, test_normal_dataset, test_anomaly_dataset, threshold=26):
    _, losses = predict(model, train_dataset)
    sns.distplot(losses, bins=50, kde=True)

    THRESHOLD = threshold

    predictions, pred_losses = predict(model, test_normal_dataset)
    sns.distplot(pred_losses, bins=50, kde=True)


    correct = sum(l <= THRESHOLD for l in pred_losses)
    print(f'Correct normal predictions: {correct}/{len(test_normal_dataset)}')


    anomaly_dataset = test_anomaly_dataset[:len(test_normal_dataset)]

    predictions, pred_losses = predict(model, anomaly_dataset)
    sns.distplot(pred_losses, bins=50, kde=True)



    correct = sum(l > THRESHOLD for l in pred_losses)
    print(f'Correct anomaly predictions: {correct}/{len(anomaly_dataset)}')


    fig, axs = plt.subplots(
    nrows=2,
    ncols=6,
    sharey=True,
    sharex=True,
    figsize=(22, 8)
    )

    for i, data in enumerate(test_normal_dataset[:6]):
        plot_prediction(data, model, title='Normal', ax=axs[0, i])

    for i, data in enumerate(test_anomaly_dataset[:6]):
        plot_prediction(data, model, title='Anomaly', ax=axs[1, i])

    fig.tight_layout();
     
  









